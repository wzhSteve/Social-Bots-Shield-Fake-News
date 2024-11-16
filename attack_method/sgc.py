"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from torch_scatter import scatter_add

class SGC(torch.nn.Module):

    def __init__(self, args, nfeat, nclass, K=3, cached=True, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None):

        super(SGC, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        # self.conv1 = SGConv(nfeat,
        #         nclass, bias=with_bias, K=K, cached=cached)

        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.weight = Parameter(torch.FloatTensor(nfeat, nclass))
        self.bias = Parameter(torch.FloatTensor(nclass))
        self.reset_parameters()
        OUT_PATH = "attack_method/results/"
        self.checkpoint_file = OUT_PATH + "{}_sgc_surrogate_trained.pkl".format(args.dataset)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # def forward(self, data):
    #     x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
    #     x = self.conv1(x, edge_index, edge_weight)
    #     return F.log_softmax(x, dim=1)

    def forward(self, data):
        x, edge_index, edge_weight, degree = data.x, data.edge_index, data.edge_weight, data.degree
        edge_weight = self.gcn_norm(edge_index, edge_weight, degree)
        x = torch.mm(x, self.weight)
        row, col = edge_index
        for _ in range(2):
            src = x[row] * edge_weight.view(-1, 1)
            x = scatter_add(src, col, dim=-2, dim_size=x.size(0))
        return F.log_softmax(x + self.bias, dim=1)

    @staticmethod
    def gcn_norm(edge_index, weights, degree):
        row, col = edge_index
        inv_degree = torch.pow(degree, -0.5)
        normed_weights = weights * inv_degree[row] * inv_degree[col]
        return normed_weights

    def initialize(self):
        """Initialize parameters of SGC.
        """
        # self.conv1.reset_parameters()
        # self.weight.reset_parameters()
        # self.bias.reset_parameters()

    def fit(self, pyg_data, train_iters=200, initialize=True, verbose=False, patience=500, **kwargs):
        # self.device = self.conv1.weight.device
        # if initialize:
        #     self.initialize()

        self.data = pyg_data
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training SGC model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                # weights = deepcopy(self.state_dict())
                torch.save(self.state_dict(), self.checkpoint_file)
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        # self.load_state_dict(weights)
        self.load_state_dict(torch.load(self.checkpoint_file))
        torch.cuda.empty_cache()

    def test(self):
        self.eval()
        self.load_state_dict(torch.load(self.checkpoint_file))
        test_mask = self.data.test_mask
        labels = self.data.y
        with torch.no_grad():
            output = self.forward(self.data)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        self.eval()
        self.load_state_dict(torch.load(self.checkpoint_file))
        with torch.no_grad():
            output = self.forward(self.data)
        return output


