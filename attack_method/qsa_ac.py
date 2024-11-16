import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from collections import namedtuple
from functools import lru_cache

from torch_scatter import scatter_add
from torch_geometric.utils import k_hop_subgraph
from base_attack import BaseAttack
from deeprobust.graph import utils
import pickle
from scipy.sparse import csr_matrix
from tqdm import tqdm
from gnn_model import *


SubGraph = namedtuple('SubGraph', ['connect_edge_index',
                                   'self_loop', 'self_loop_weight',
                                   'connect_edge_weight',
                                   'edges_all'])

class QSAttack(BaseAttack):

    def __init__(self, model, victim_model, data, args, nnodes=None, attack_structure=True, attack_features=False, surrogate='sgc', device='cpu'):

        super(QSAttack, self).__init__(model=None, nnodes=nnodes,
                                       attack_structure=attack_structure, attack_features=attack_features, device=device)
        self.target_node = None
        self.surrogate = surrogate
        self.victim_model = victim_model
        self.device = device
        self.data = data
        self.args = args

        if surrogate == 'gcn':
            self.K = 2
            self.weight1 = model.gc1.weight.to(device)
            self.weight2 = model.gc2.weight.to(device)
            self.bias1 = model.gc1.bias.to(device)
            self.bias2 = model.gc2.bias.to(device)
            self.weight = [self.weight1, self.weight2]
            self.bias = [self.bias1, self.bias2]
            self.with_relu = model.with_relu

        elif surrogate == 'sgc':
            self.K = model.conv1.K
            W = model.conv1.lin.weight.to(device)
            b = model.conv1.lin.bias
            if b is not None:
                b = b.to(device)
            self.weight, self.bias = W, b
        self.edge_weight = None
        self.self_loop_weight = None
        self.last_grad = None

    @lru_cache(maxsize=1)
    def compute_XW(self):
        return F.linear(self.modified_features, self.weight)

    def attack(self, features, adj, labels, target_node, n_perturbations, direct=True, n_influencers=3, **kwargs):
        if sp.issparse(features):
            # to dense numpy matrix
            features = features.A
        if not torch.is_tensor(features):
            features = torch.tensor(features, device=self.device)
        if torch.is_tensor(adj):
            from scipy.sparse import csr_matrix
            adj = csr_matrix(adj.to('cpu').detach().numpy())
        self.modified_features = features.requires_grad_(bool(self.attack_features))
        target_label = torch.tensor([1.], dtype=labels.dtype)
        best_wrong_label = torch.tensor([0.], dtype=labels.dtype)
        self.selfloop_degree = torch.tensor(adj.sum(1).A1, device=self.device)
        self.target_label = target_label.to(self.device)
        self.best_wrong_label = best_wrong_label.to(self.device)
        self.n_perturbations = n_perturbations
        self.ori_adj = adj
        self.target_node = target_node
        self.direct = direct
        attacker_nodes = torch.where(torch.as_tensor(labels) != 2)[0]

        self.modified_adj = adj.copy()
        structure_perturbations = []
        feature_perturbations = []
        for t in range(n_perturbations):
            subgraph = self.get_subgraph(attacker_nodes)
            grad_value, grad_index, loss_list = self.compute_gradient(subgraph)
            if self.args.constrain:
                node_index = subgraph.connect_edge_index[1, grad_index]
                ptb_value = self.NASR_Change(node_index)
                ptb_diff = self.args.alpha * ptb_value
                score = loss_list - ptb_diff.to(loss_list.device)
                max_connect_edge_grad, max_connect_edge_idx = torch.max(score, dim=0)
                best_edge = subgraph.connect_edge_index[:, grad_index[max_connect_edge_idx]]
            else:
                max_connect_edge_grad, max_connect_edge_idx = torch.max(loss_list, dim=0)
                best_edge = subgraph.connect_edge_index[:, grad_index[max_connect_edge_idx]]
            u, v = best_edge.tolist()
            self.selfloop_degree[u] += 2.0
            self.selfloop_degree[v] += 2.0
            assert u != v
            structure_perturbations.append((u, v))
            modified_adj = self.modified_adj.tolil(copy=True)
            row, col = u, v
            modified_adj[row, col] += 1
            modified_adj[col, row] += 1
            modified_adj[row, row] += 1
            modified_adj[col, col] += 1
            modified_adj = modified_adj.tocsr(copy=False)
            modified_adj.eliminate_zeros()
            self.modified_adj = modified_adj
        self.modified_features = self.modified_features.detach().cpu().numpy()
        self.structure_perturbations = structure_perturbations
        self.feature_perturbations = feature_perturbations


    def NASR_Change(self, attacker_nodes):
        target_node = self.target_node
        neighbors = self.modified_adj[target_node].indices.tolist()
        attacker_nodes = attacker_nodes.to('cpu').numpy()
        neighbors.remove(target_node)
        adj_matrix = self.modified_adj.tolil(copy=True).A
        strengths_matrix = np.sum(adj_matrix, axis=1)
        strengths_target = np.sum(strengths_matrix[neighbors])
        neighbors_vector = adj_matrix[attacker_nodes]
        neighbors_mask = np.where(neighbors_vector != 0, 1., 0.)
        strengths_neighbors = strengths_matrix[np.newaxis, :].repeat(neighbors_mask.shape[0], axis=0)
        l1 = np.abs(np.sum(strengths_neighbors * neighbors_mask, axis=1) - strengths_target + strengths_matrix[target_node] - strengths_matrix[attacker_nodes])
        l2 = np.sum(np.power(strengths_neighbors * neighbors_mask + 1, -1), axis=1)
        l = l1 + l2
        l = F.normalize(torch.tensor(l), dim=0)
        return l

    def get_subgraph(self, attacker_nodes):
        target_node = self.target_node
        influencers = [target_node]
        attacker_nodes = attacker_nodes.to('cpu').numpy()
        attacker_nodes = np.setdiff1d(attacker_nodes, target_node)
        subgraph = self.subgraph_processing(influencers, attacker_nodes)
        return subgraph

    def subgraph_processing(self, influencers, attacker_nodes):
        row = np.repeat(influencers, len(attacker_nodes))
        col = np.tile(attacker_nodes, len(influencers))
        connect_edges = np.row_stack([row, col])

        connect_edges = torch.as_tensor(connect_edges, device=self.device)
        unique_nodes = attacker_nodes
        unique_nodes = torch.as_tensor(unique_nodes, device=self.device)
        self_loop = unique_nodes.repeat((2, 1))
        edges_all = torch.cat([connect_edges, connect_edges[[1, 0]], self_loop], dim=1)

        connect_edge_weight = self.modified_adj.A[connect_edges[0].to('cpu').numpy(), connect_edges[1].to('cpu').numpy()] + 1e-5
        connect_edge_weight = torch.tensor(connect_edge_weight).to(self.device).requires_grad_(bool(self.attack_structure))

        self_loop_weight = self.modified_adj.A[self_loop[0].to('cpu').numpy(), self_loop[1].to('cpu').numpy()]
        self_loop_weight = torch.tensor(self_loop_weight).to(self.device).requires_grad_(bool(self.attack_structure))

        connect_edge_index = connect_edges
        self_loop = self_loop

        subgraph = SubGraph(connect_edge_index=connect_edge_index,
                            self_loop=self_loop, edges_all=edges_all,
                            connect_edge_weight=connect_edge_weight,
                            self_loop_weight=self_loop_weight)
        return subgraph

    def SGCCov(self, x, edge_index, edge_weight):
        row, col = edge_index
        for _ in range(self.K):
            src = x[row] * edge_weight.view(-1, 1)
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
        return x

    def GCN(self, x, edge_index, edge_weight):
        row, col = edge_index
        for i in range(self.K):
            x = F.linear(input=x, weight=self.weight[i].transpose(1, 0))
            src = x[row] * edge_weight.view(-1, 1)
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
            x += self.bias[i]
            if i == 0 and self.with_relu:
                x = F.relu(x)
        return x

    def compute_gradient(self, subgraph, eps=5.0):
        connect_edge_weight = subgraph.connect_edge_weight
        self_loop_weight = subgraph.self_loop_weight
        weights = torch.cat([connect_edge_weight, connect_edge_weight,
                            self_loop_weight], dim=0)

        weights = self.gcn_norm(subgraph.edges_all, weights, self.selfloop_degree)
        if self.surrogate == 'sgc':
            logit_ori = self.SGCCov(self.compute_XW(), subgraph.edges_all, weights)
        elif self.surrogate == 'gcn':
            logit_ori = self.GCN(self.modified_features, subgraph.edges_all, weights)
        logit_ori = logit_ori[self.target_node]

        if self.surrogate == 'sgc' and self.bias is not None:
            logit_ori += self.bias

        logit_ori = logit_ori.unsqueeze(0)
        loss_ori = F.nll_loss(logit_ori, self.target_label) - F.nll_loss(logit_ori, self.best_wrong_label)

        connect_edge_grad, self_loop_grad = torch.autograd.grad(loss_ori, [connect_edge_weight, self_loop_weight], create_graph=False)
        # x_weight = torch.cat([self.compute_XW()[:self.target_node], self.compute_XW()[self.target_node+1:]], dim=0)
        top_k = self.args.topk #connect_edge_grad.shape[0]
        grad_value, grad_index = torch.topk(connect_edge_grad, k=top_k, largest=True)
        loss_list = torch.tensor([]).to(self.device)
        for i in range(grad_index.shape[0]):
            modified_adj = self.modified_adj.tolil(copy=True)
            edge_col_row = subgraph.connect_edge_index[:, grad_index[i]]
            u, v = edge_col_row.tolist()
            row, col = u, v
            modified_adj[row, col] += 1
            modified_adj[col, row] += 1
            modified_adj[row, row] += 1
            modified_adj[col, col] += 1
            modified_adj = modified_adj.tocsr(copy=False)
            modified_adj.eliminate_zeros()

            self.victim_model.eval()
            self.data.adj = torch.tensor(modified_adj.todense()).to(self.device)
            if self.args.victim_model == 'gcn':
                output = self.victim_model.predict(self.modified_features, self.data.adj)
            else:
                output = predict(self.victim_model, self.data, self.args)

            logit = output[self.target_node].unsqueeze(0)
            loss = F.nll_loss(logit, self.target_label) - F.nll_loss(logit, self.best_wrong_label)
            loss_list = torch.cat([loss_list, loss.unsqueeze(-1)])
            torch.cuda.empty_cache()
        # index = loss_list.index(max(loss_list))
        return grad_value, grad_index, loss_list

    @ staticmethod
    def gcn_norm(edge_index, weights, degree):
        row, col = edge_index.to('cpu').numpy()
        inv_degree = torch.pow(degree, -0.5)
        normed_weights = weights * inv_degree[row] * inv_degree[col]
        return normed_weights