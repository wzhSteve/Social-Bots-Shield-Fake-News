import torch
import torch.nn as nn
from torch_sparse import spmm  # require the newest torch_sprase
import torch.nn.functional as F
import numpy as np
import argparse
import torch.nn as nn
import sys, os
sys.path.append(os.getcwd())
from Process.load_graph import *
from tqdm import tqdm
from torch_geometric.nn import MLP
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP


from torch_scatter import scatter_max, scatter_add


def softmax(src, index, num_nodes=None):
    """
        sparse softmax
    """
    num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out

class GCNDecor(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, n_classes, mlp_width):
        super().__init__()

        self.adj_transform = MLP([3, mlp_width, 2]) # replace DCSBM
        # self.adj_transform = torch.nn.DataParallel(MLP([3, mlp_width, 2]), device_ids=[0, 1, 2, 3])
        self.softmax = nn.Softmax(dim = -1)
        self.weight1 = nn.Linear(in_dim, hid_dim, bias=False)
        self.weight2 = nn.Linear(hid_dim, n_classes, bias=False)
        self.dropout = nn.Dropout(0.5)

    def eig_power(self, A, eps=1e-15):
        v0 = np.random.rand(A.shape[0])
        vec_norm = np.linalg.norm(v0)
        v0 = torch.tensor(v0 / vec_norm, dtype=torch.float).to(device)
        v0[torch.isinf(v0)] = 0.

        uk = v0
        flag = 1
        val_old = 0.
        n = 0
        while flag:
            n = n + 1
            vk = torch.matmul(A, uk)
            val = vk[torch.argmax(torch.abs(vk))]
            uk = vk / val
            t = torch.abs(val - val_old)
            if (t < eps):
                flag = 0
            val_old = val
            # print(np.asarray(uk).flatten(), val)
        # print('max eigenvalue:', val)
        # print('eigenvector:', np.asarray(uk).flatten())
        # print('iteration:', n)
        return val

    def mid_filter_3(self, adj, alpha=1):
        """Middle normalize adjacency matrix."""
        # add self-loop and normalization also affects performance a lot

        rowsum = torch.sum(adj, dim=1)
        D_row = torch.pow(rowsum, -0.5).flatten()
        D_row[torch.isinf(D_row)] = 0.
        D_row = torch.diag(D_row)
        colsum = torch.sum(adj, dim=0)
        D_col = torch.pow(colsum, -0.5).flatten()
        D_col[torch.isinf(D_col)] = 0.
        D_col = torch.diag(D_col)
        DAD = adj.mm(D_col).transpose(0, 1).mm(D_row).transpose(0, 1)

        I = torch.eye(DAD.shape[0], DAD.shape[1]).to(device)
        L = I - DAD
        max_ev = 2#self.eig_power(L)
        mid_point = max_ev / 2.
        return torch.matmul(0.5 * I - DAD, I + DAD) #I - torch.matmul(L, L)

    def forward(self, data):
        indices = torch.nonzero(data.adj.view(-1, 1), as_tuple=True)[0].cpu().detach().numpy()
        adj_flattened = torch.zeros(data.adj.view(-1, 1).shape[0]).to(data.adj.device)
        x, adj = data.x, torch.cat((data.adj.view(-1, 1)[indices], data.xdeg.view(-1, 1)[indices], data.ydeg.view(-1, 1)[indices]), 1)

        adj_mask = self.softmax(self.adj_transform(adj))[:, 1] # 0&1 has no effect  co-engagement matrix C

        adj_flattened[indices] = adj_mask
        adj_mask = adj_flattened.reshape(data.adj.shape[0], data.adj.shape[1])

        adj = data.adj #* adj_mask
        adj = adj + torch.eye(*adj.shape).to(data.adj.device)
        # print('adj before norm: {}'.format(self.eig_power(adj)))
        # adj = self.mid_filter_3(adj)

        rowsum = torch.sum(adj, dim=1)
        D_row = torch.pow(rowsum, -0.5).flatten()
        D_row[torch.isinf(D_row)] = 0.
        D_row = torch.diag(D_row)
        colsum = torch.sum(adj, dim=0)
        D_col = torch.pow(colsum, -0.5).flatten()
        D_col[torch.isinf(D_col)] = 0.
        D_col = torch.diag(D_col)
        adj = adj.mm(D_col).transpose(0, 1).mm(D_row).transpose(0, 1) # D^(-1/2)AD^(-1/2) 2-L Low-pass
        # print('adj norm: {}'.format(self.eig_power(adj)))

        support = self.weight1(x)
        output = torch.mm(adj, support)
        hid = self.dropout(output)
        support = self.weight2(hid)
        output = torch.mm(adj, support)

        return output


class MidGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.4, nlayer=2, alpha=0.5):
        super(MidGCN, self).__init__()
        self.alpha = alpha
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([
            GraphConvolution(nfeat if i == 0 else nhid, nhid, with_bias=False)
            for i in range(nlayer - 1)
        ])
        self.out_layer = GraphConvolution(nfeat if nlayer == 1 else nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.nrom = nn.BatchNorm1d(nhid)

    def mid_filter_2(self, adj, alpha):
        """Middle normalize adjacency matrix."""
        # add self-loop and normalization also affects performance a lot
        rowsum = torch.sum(adj, dim=1)
        D_row = torch.pow(rowsum, -0.5).flatten()
        D_row[torch.isinf(D_row)] = 0.
        D_row = torch.diag(D_row)
        colsum = torch.sum(adj, dim=0)
        D_col = torch.pow(colsum, -0.5).flatten()
        D_col[torch.isinf(D_col)] = 0.
        D_col = torch.diag(D_col)
        DAD = adj.mm(D_col).transpose(0, 1).mm(D_row).transpose(0, 1)
        I = torch.eye(adj.shape[0], adj.shape[1]).to(device)
        L = I - DAD
        return torch.matmul(alpha * I - DAD, I + DAD)  # middle filter: (L - (1 - alpha)) * (2I - L)

    def forward(self, data):
        x, adj = data.x, data.adj
        adj = self.mid_filter_2(adj, self.alpha)
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.relu(x)
        x = self.dropout(x)
        x = self.out_layer(x, adj)
        x = torch.log_softmax(x, dim=-1)
        return x

# class GCN(nn.Module):
#
#     def __init__(self, nfeat, nhid, nclass, dropout=0.4, nlayer=2, with_relu=True, with_bias=True):
#         super(MidGCN, self).__init__()
#         self.with_relu = with_relu
#         assert nlayer >= 1
#         self.hidden_layers = nn.ModuleList([
#             GraphConvolution(nfeat if i == 0 else nhid, nhid, with_bias=with_bias)
#             for i in range(nlayer - 1)
#         ])
#         self.out_layer = GraphConvolution(nfeat if nlayer == 1 else nhid, nclass)
#
#         self.dropout = nn.Dropout(p=dropout)
#         self.dropout_rate = dropout
#         self.relu = nn.ReLU(True)
#         self.nrom = nn.BatchNorm1d(nhid)
#
#     def normalize(self, adj):
#         """Middle normalize adjacency matrix."""
#         # add self-loop and normalization also affects performance a lot
#         rowsum = torch.sum(adj, dim=1)
#         D_row = torch.pow(rowsum, -0.5).flatten()
#         D_row[torch.isinf(D_row)] = 0.
#         D_row = torch.diag(D_row)
#         colsum = torch.sum(adj, dim=0)
#         D_col = torch.pow(colsum, -0.5).flatten()
#         D_col[torch.isinf(D_col)] = 0.
#         D_col = torch.diag(D_col)
#         DAD = adj.mm(D_col).transpose(0, 1).mm(D_row).transpose(0, 1)
#         return DAD
#
#     def forward(self, data):
#         x, adj = data.x, data.adj
#         adj = self.normalize(adj)
#         for i, layer in enumerate(self.hidden_layers):
#             # x = self.dropout(x)
#             x = layer(x, adj)
#             if self.with_relu:
#                 x = self.relu(x)
#         x = self.dropout(x)
#         x = self.out_layer(x, adj)
#         x = torch.log_softmax(x, dim=-1)
#         return x

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
    

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活

        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout=0.5, alpha=0.5, n_heads=8):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, data):
        x, adj = data.x, data.adj
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定


class DeepGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, nlayer=2, alpha=0.5):
        super(DeepGCN, self).__init__()
        self.alpha = alpha
        assert nlayer >= 1
        self.hidden_layers = nn.ModuleList([
            GraphConv(nfeat if i == 0 else nhid, nhid, bias=False)
            for i in range(nlayer - 1)
        ])
        self.out_layer = GraphConv(nfeat if nlayer == 1 else nhid, nclass)

        self.dropout = nn.Dropout(p=dropout)
        self.dropout_rate = dropout
        self.relu = nn.ReLU(True)
        self.nrom = nn.BatchNorm1d(nhid)
        # self.coe1 = nn.Parameter(torch.ones(2485, 2485))
        # self.coe2 = nn.Parameter(torch.ones(2485, 2485))

    def forward(self, data):
        x, adj = data.x, data.adj
        for i, layer in enumerate(self.hidden_layers):
            x = self.dropout(x)
            x = layer(x, adj)
            x = self.relu(x)

        x = self.dropout(x)
        x = self.out_layer(x, adj)
        x = torch.log_softmax(x, dim=-1)
        return x

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.cuda.manual_seed_all(seed)

criterion = nn.CrossEntropyLoss()


def train(net, optimizer, data):
    net.train()
    optimizer.zero_grad()
    output = net(data)
    output, labels = output[data.train_mask], data.y[data.train_mask]

    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    output = output.argmax(dim=1)
    acc = accuracy_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
    precision, recall, fscore, _ = score(labels.detach().cpu().numpy(), output.detach().cpu().numpy(),
                                         average='macro')
    return loss, acc, precision, recall, fscore


def val(net, optimizer, data):
    net.eval()
    with torch.no_grad():
        output = net(data)
    output, labels = output[data.val_mask], data.y[data.val_mask]
    loss_val = criterion(output, labels)
    output = output.argmax(dim=1)
    acc = accuracy_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
    precision, recall, fscore, _ = score(labels.detach().cpu().numpy(), output.detach().cpu().numpy(),
                                         average='macro')

    return loss_val, acc, precision, recall, fscore


def test(net, optimizer, data):
    net.eval()
    with torch.no_grad():
        output = net(data)
    output, labels = output[data.test_mask], data.y[data.test_mask]
    loss_test = criterion(output, labels)
    output = output.argmax(dim=1)

    acc = accuracy_score(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
    precision, recall, fscore, _ = score(labels.detach().cpu().numpy(), output.detach().cpu().numpy(),
                                         average='macro')

    return loss_test, acc, precision, recall, fscore


def predict(net, data, args):
    net.eval()
    with torch.no_grad():
        output = net(data)
    return output


def fit(net, data, args):

    test_accs = []
    prec_all, rec_all, f1_all = [], [], []

    OUT_PATH = "attack_method/results/"
    if args.gcn_model != None:
        checkpoint_file = OUT_PATH + "{}_gcn_{}_trained.pkl".format(args.dataset, args.gcn_model)
    else:
        checkpoint_file = OUT_PATH + "{}_{}_trained.pkl".format(args.dataset, args.victim_model)
    iterations = args.iters
    if args.retrain:
        print("-----------------Start Training Process-----------------")
        for i in range(iterations):
            print(len(data.train_mask))
            print(len(data.val_mask))
            print(len(data.test_mask))

            optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-6)
            net.train()

            early_stopping = args.patience
            best_acc = 0
            best_loss = 1e10
            import time
            s = time.time()

            for epoch in range(args.epochs):
                train_loss, train_acc, train_prec, train_recall, train_f1 = train(net, optimizer, data)
                val_loss, val_acc, val_prec, val_recall, val_f1 = val(net, optimizer, data)
                test_loss, test_acc, test_prec, test_recall, test_f1 = test(net, optimizer, data)

                print('Epoch %d: train loss %.3f train acc: %.3f, val loss: %.3f val acc %.3f | test acc %.3f.' %
                      (epoch, train_loss, train_acc, val_loss, val_acc, test_acc))

                if best_acc < val_acc:
                    best_acc = val_acc
                    torch.save(net.state_dict(), checkpoint_file)

            e = time.time()
            print("avr epoch time:", (e - s) / args.epochs)
            # pick up the best model based on val_acc, then do test
            net.load_state_dict(torch.load(checkpoint_file))

            test_loss, test_acc, test_prec, test_recall, test_f1 = test(net, optimizer, data)

            print(['Global Test Accuracy:{:.4f}'.format(test_acc),
                   'Precision:{:.4f}'.format(test_prec),
                   'Recall:{:.4f}'.format(test_recall),
                   'F1:{:.4f}'.format(test_f1)])
            # print("-----------------End of Iter {:03d}-----------------".format(iter))

            test_accs.append(test_acc)
            prec_all.append(test_prec)
            rec_all.append(test_recall)
            f1_all.append(test_f1)

        print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
            sum(test_accs) / iterations, sum(prec_all) / iterations, sum(rec_all) / iterations, sum(f1_all) / iterations))
        # with open('results.txt', 'a') as f:
        #     f.write("{} Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(checkpoint_file,
        #     sum(test_accs) / iterations, sum(prec_all) / iterations, sum(rec_all) / iterations, sum(f1_all) / iterations))

    print("-----------------Start Test-----------------")
    net.load_state_dict(torch.load(checkpoint_file))
    test_loss, test_acc, test_prec, test_recall, test_f1 = test(net, None, data)

    print("Victim_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
        test_acc, test_prec, test_recall, test_f1))




