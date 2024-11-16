import torch
import numpy as np
from gcn import GCN
# from gcn import GCN
from qsa_ac import QSAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset, Dpr2Pyg
from sgc import SGC
import torch.nn as nn
import argparse
from tqdm import tqdm
from deeprobust.graph.utils import accuracy
import pickle
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from gnn_model import *
from collections import OrderedDict
import os

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'], help='dataset')
parser.add_argument('--subset', type=str, default='None', choices=['None', 'subset1', 'subset2', 'subset3'], help='dataset')

parser.add_argument('--victim_model', type=str, default='gcn', choices=['gcn', 'decor', 'gat', 'mid-gcn', 'gprgnn', 'prognn'])
parser.add_argument('--surrogate', type=str, default='gcn', choices=['gcn', 'sgc'])
parser.add_argument('--gcn_relu', action='store_true', default=False, help='gcn with relu')
parser.add_argument('--retrain', action='store_true', default=False, help='retrain the victim models')
parser.add_argument('--constrain', action='store_true', default=False, help='add constrain')
parser.add_argument('--topk', type=int, default=50, help='topk selected from surrogate model.')
parser.add_argument('--alpha', type=float, default=1, help='topk selected from surrogate model.')
parser.add_argument('--m', type=int, default=1, help='topk selected from surrogate model.')

parser.add_argument('--iters', default=5, type=int)
parser.add_argument('--patience', type=int, default=20, help='Early stope')
parser.add_argument('--epochs', type=int,  default=300, help='Number of epochs to train.')

# Mid-GCN
parser.add_argument('--mid_alpha', type=float, default=0.2, help='weight of l1 norm') # 0.2 for political


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load dataset
if args.dataset == "politifact":
    features = pickle.load(open('data/news_features/' + args.dataset + '_bert_raw_768d.pkl', 'rb'))
    graph_dict = pickle.load(open('data/user_news_graph/weighted/' + args.dataset + '_un_relations_t3_raw.pkl', 'rb'))

if args.subset != 'None':
    mask_dict = pickle.load(open('data/temp_splits/' + args.dataset + '_' + args.subset + '.pkl', 'rb'))
    idx_train, idx_val, idx_test, labels, features, A_un = mask_dict['train_mask'], mask_dict['val_mask'], mask_dict['test_mask'], mask_dict['label'], mask_dict['features'], mask_dict['A_un']
    idx_unlabeled = np.union1d(idx_val, idx_test)
    labels = np.array(labels)
    A_un = torch.Tensor(A_un)
else:
    mask_dict = pickle.load(open('data/temp_splits/' + args.dataset + '_split.pkl', 'rb'))
    idx_train, idx_val, idx_test = mask_dict['train_mask'], mask_dict['val_mask'], mask_dict['test_mask']
    idx_unlabeled = np.union1d(idx_val, idx_test)
    labels = np.array(mask_dict['label'])
    A_un = torch.Tensor(graph_dict['A_un'])

s = round(A_un.shape[1] / 100)
A_un_new = torch.where(A_un < s, A_un, torch.tensor(s, dtype=A_un.dtype))
eng_adj = A_un_new.transpose(0, 1).matmul(A_un_new)

A_un_thres1 = torch.where(A_un < 1, A_un, torch.tensor(1., dtype=A_un.dtype))
co_adj = A_un_thres1.transpose(0, 1).matmul(A_un_thres1)
adj = co_adj.to(device)

features = features.to(device)
labels = torch.tensor(labels, dtype=torch.int64).to(device)

fake_news = torch.sum(labels).cpu().numpy()
total_news = labels.shape[0]
real_news = total_news - fake_news
user = torch.sum(A_un_thres1, dim=1)
user = torch.sum(torch.where(user > torch.tensor(0., dtype=A_un.dtype), torch.tensor(1., dtype=A_un.dtype), torch.tensor(0., dtype=A_un.dtype)))

print("{}: Real News {}, Fake News {}, Users {}".format(args.subset, real_news, fake_news, user))

class Data():
    def __init__(self):
        self.A = None

if args.subset == 'None':
    file_path = r'data/user_news_graph/weighted/{}_edge_index.pkl'.format(args.dataset)
else:
    file_path = r'data/user_news_graph/weighted/{}_edge_index_{}.pkl'.format(args.dataset, args.subset)
t = os.path.exists(file_path)
if os.path.exists(file_path):
    print("Read existing data!")
    weighted_data = pickle.load(open(file_path, 'rb'))
    data = weighted_data['data']
else:
    print("Generate new data!")
    data = Data()
    data.x = features
    data.adj = adj.clone().detach().to(device)
    data.edge_index = torch.nonzero(data.adj).t().to(device)
    data.edge_weight = torch.tensor(
        data.adj[data.edge_index[0].to('cpu').numpy(), data.edge_index[1].to('cpu').numpy()]).to(device)
    data.train_mask = idx_train
    data.val_mask = idx_val
    data.test_mask = idx_test
    data.y = torch.tensor(labels, dtype=torch.int64).to(device)
    data.degree = adj.sum(1)
    # decor
    xdeg, ydeg = eng_adj.sum(0), eng_adj.sum(1)
    xdeg = xdeg.view(-1, 1)
    xdeg, ydeg = xdeg.repeat(1, eng_adj.shape[0]), ydeg.repeat(eng_adj.shape[1], 1)
    data.xdeg = xdeg.to(device)
    data.ydeg = ydeg.to(device)

    data_ = {'data': data}
    with open(file_path, 'wb') as file:
        pickle.dump(data_, file)

args.gcn_model = None
# train victim model
mlp_width = 16 if args.dataset == 'politifact' else 8
if args.victim_model == 'gcn':
    args.gcn_model = 'victim'
    victim_model = GCN(args=args, nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                device=device)
    victim_model = victim_model.to(device)
    iterations = 5
    test_accs = []
    prec_all, rec_all, f1_all = [], [], []
    criterion = nn.CrossEntropyLoss()
    label = labels[idx_test]
    if args.retrain:
        for i in range(iterations):
            victim_model.fit(features, adj, labels, idx_train, idx_val)
            # net.test(data.test_mask)
            output = victim_model.output
            output = output[idx_test]
            loss_val = criterion(output, label)
            output = output.argmax(dim=1)
            test_acc = accuracy_score(label.detach().cpu().numpy(), output.detach().cpu().numpy())
            test_prec, test_recall, test_f1, _ = score(label.detach().cpu().numpy(), output.detach().cpu().numpy(),
                                                       average='macro')
            print(['Global Test Accuracy:{:.4f}'.format(test_acc),
                   'Precision:{:.4f}'.format(test_prec),
                   'Recall:{:.4f}'.format(test_recall),
                   'F1:{:.4f}'.format(test_f1)])
            test_accs.append(test_acc)
            prec_all.append(test_prec)
            rec_all.append(test_recall)
            f1_all.append(test_f1)
        print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
            sum(test_accs) / iterations, sum(prec_all) / iterations, sum(rec_all) / iterations, sum(f1_all) / iterations))
    output = victim_model.predict(features, adj)
    labels_test = output.argmax(1)
    test_acc = accuracy_score(label.detach().cpu().numpy(), labels_test[idx_test].detach().cpu().numpy())
    test_prec, test_recall, test_f1, _ = score(label.detach().cpu().numpy(), labels_test[idx_test].detach().cpu().numpy(),
                                               average='macro')
    print(['Victim Test Accuracy:{:.4f}'.format(test_acc),
           'Precision:{:.4f}'.format(test_prec),
           'Recall:{:.4f}'.format(test_recall),
           'F1:{:.4f}'.format(test_f1)])
    test_accs.append(test_acc)
    prec_all.append(test_prec)
    rec_all.append(test_recall)
    f1_all.append(test_f1)

elif args.victim_model == 'decor':
    victim_model = GCNDecor(768, 64, 2, mlp_width).to(device)
    fit(victim_model, data, args)
elif args.victim_model == 'mid-gcn':
    victim_model = MidGCN(nfeat=data.x.shape[1], nclass=data.y.max().item()+1, nhid=16, alpha=args.mid_alpha).to(device)
    fit(victim_model, data, args)
elif args.victim_model == 'gat':
    victim_model = GAT(n_feat=data.x.shape[1], n_class=data.y.max().item() + 1, n_hid=16, dropout=0.2).to(device)
    fit(victim_model, data, args)


if args.surrogate == 'gcn':
    args.gcn_model = 'surrogate'
    surrogate = GCN(args=args, nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    device=device)
    surrogate = surrogate.to(device)

    iterations = 5
    test_accs = []
    prec_all, rec_all, f1_all = [], [], []
    criterion = nn.CrossEntropyLoss()
    label = labels[idx_test]
    # if args.retrain:
    for i in range(iterations):
        surrogate.fit(features, adj, labels, idx_train, idx_val)
        output = surrogate.output
        output = output[idx_test]
        loss_val = criterion(output, label)
        output = output.argmax(dim=1)
        test_acc = accuracy_score(label.detach().cpu().numpy(), output.detach().cpu().numpy())
        test_prec, test_recall, test_f1, _ = score(label.detach().cpu().numpy(), output.detach().cpu().numpy(),
                                                   average='macro')
        print(['Global Test Accuracy:{:.4f}'.format(test_acc),
               'Precision:{:.4f}'.format(test_prec),
               'Recall:{:.4f}'.format(test_recall),
               'F1:{:.4f}'.format(test_f1)])
        test_accs.append(test_acc)
        prec_all.append(test_prec)
        rec_all.append(test_recall)
        f1_all.append(test_f1)

    print("Total_Test_Accuracy: {:.4f}|Prec_Macro: {:.4f}|Rec_Macro: {:.4f}|F1_Macro: {:.4f}".format(
        sum(test_accs) / iterations, sum(prec_all) / iterations, sum(rec_all) / iterations, sum(f1_all) / iterations))
    output = surrogate.predict(features, adj)
    labels_test = output.argmax(1)
    test_acc = accuracy_score(label.detach().cpu().numpy(), labels_test[idx_test].detach().cpu().numpy())
    test_prec, test_recall, test_f1, _ = score(label.detach().cpu().numpy(), labels_test[idx_test].detach().cpu().numpy(),
                                               average='macro')
    print('Surrogate Test Accuracy:{:.4f}'.format(test_acc),
           'Precision:{:.4f}'.format(test_prec),
           'Recall:{:.4f}'.format(test_recall),
           'F1:{:.4f}'.format(test_f1))
    test_accs.append(test_acc)
    prec_all.append(test_prec)
    rec_all.append(test_recall)
    f1_all.append(test_f1)

    labels_test[idx_train] = labels[idx_train].clone().detach()
    labels_test[idx_val] = labels[idx_val].clone().detach()

torch.cuda.empty_cache()

def weighted_global_assortativity_value(target_node, adj_matrix, modified_adj):
    modified_adj = torch.tensor(modified_adj.todense()).to(device)
    strengths_matrix = torch.sum(modified_adj, dim=1)
    # sum of all weighted edges
    H = torch.sum(adj_matrix)
    H_m = torch.sum(modified_adj)

    target_modified = modified_adj
    edge_index = torch.nonzero(target_modified).t()
    w_e = torch.tensor(modified_adj[edge_index[0].to('cpu').numpy(), edge_index[1].to('cpu').numpy()]).to(device)
    s_e = strengths_matrix[edge_index[0].to('cpu').numpy()] - w_e
    t_e = strengths_matrix[edge_index[1].to('cpu').numpy()] - w_e

    U_w = torch.sum(0.5 * w_e * (s_e + t_e)) / H_m
    U_w_2 = torch.sum(0.5 * w_e * (s_e * s_e + t_e * t_e)) / H_m
    sigma_w = torch.sqrt(U_w_2 - U_w * U_w)
    r_d = torch.sum(w_e * (s_e - U_w) * (t_e - U_w)) / (H_m * sigma_w * sigma_w)

    target_matrix = adj_matrix
    edge_index = torch.nonzero(target_matrix).t()
    strengths_matrix = torch.sum(adj_matrix, dim=1)

    w_e = torch.tensor(adj_matrix[edge_index[0].to('cpu').numpy(), edge_index[1].to('cpu').numpy()]).to(device)
    s_e = strengths_matrix[edge_index[0].to('cpu').numpy()] - w_e
    t_e = strengths_matrix[edge_index[1].to('cpu').numpy()] - w_e

    U_w = torch.sum(0.5 * w_e * (s_e + t_e)) / H
    U_w_2 = torch.sum(0.5 * w_e * (s_e * s_e + t_e * t_e)) / H
    sigma_w = torch.sqrt(U_w_2 - U_w * U_w)
    r_w = torch.sum(w_e * (s_e - U_w) * (t_e - U_w)) / (H * sigma_w * sigma_w)

    perturb = (r_d - r_w)/r_w
    # print('t_ptb: %s' % (torch.sum(modified_adj - adj_matrix)))
    return torch.abs(perturb), torch.sum(modified_adj - adj_matrix)

def weighted_local_assortativity_value(target_node, adj_matrix, modified_adj):
    mod_adj = modified_adj
    modified_adj = torch.tensor(modified_adj.todense()).to(device)

    modified = modified_adj - adj_matrix
    strengths_modified = torch.sum(modified, dim=1)
    node_index = torch.nonzero(strengths_modified).t()
    node_list = [int(node_index[0, i]) for i in range(node_index.shape[-1])]
    node_list = mod_adj[node_list].indices
    node_list = list(OrderedDict.fromkeys(node_list))

    # weighted assortativity coefficient modified
    strengths_matrix = torch.sum(modified_adj, dim=1)
    H_m = torch.sum(modified_adj)
    node_adj = modified_adj[node_list]
    w_e = node_adj
    s_e = strengths_matrix[node_list].unsqueeze(1).repeat(1, node_adj.shape[1]) - w_e
    t_e = strengths_matrix.unsqueeze(0).repeat(node_adj.shape[0], 1) - w_e

    w_e_all = modified_adj
    s_e_all = strengths_matrix.unsqueeze(1).repeat(1, node_adj.shape[1]) - w_e_all
    t_e_all = strengths_matrix.unsqueeze(0).repeat(modified_adj.shape[0], 1) - w_e_all


    U_w = torch.sum(0.5 * w_e_all * (s_e_all + t_e_all)) / H_m
    U_w_2 = torch.sum(0.5 * w_e_all * (s_e_all * s_e_all + t_e_all * t_e_all)) / H_m
    sigma_w = torch.sqrt(U_w_2 - U_w * U_w)
    r_d = torch.sum(w_e * (s_e - U_w) * (t_e - U_w), dim=1) / (H_m * sigma_w * sigma_w)

    # weighted assortativity coefficient ori_adj
    strengths_matrix = torch.sum(adj_matrix, dim=1)
    H = torch.sum(adj_matrix)
    node_adj = adj_matrix[node_list]
    neighbors_mask = torch.where(node_adj != 0, torch.tensor(1., dtype=node_adj.dtype).to(node_adj.device), node_adj)
    w_e = node_adj
    s_e = strengths_matrix[node_list].unsqueeze(1).repeat(1, node_adj.shape[1]) - w_e
    t_e = strengths_matrix.unsqueeze(0).repeat(node_adj.shape[0], 1) - w_e

    w_e_all = adj_matrix
    s_e_all = strengths_matrix.unsqueeze(1).repeat(1, node_adj.shape[1]) - w_e_all
    t_e_all = strengths_matrix.unsqueeze(0).repeat(adj_matrix.shape[0], 1) - w_e_all

    U_w = torch.sum(0.5 * w_e_all * (s_e_all + t_e_all)) / H
    U_w_2 = torch.sum(0.5 * w_e_all * (s_e_all * s_e_all + t_e_all * t_e_all)) / H
    sigma_w = torch.sqrt(U_w_2 - U_w * U_w)
    r_w = torch.sum(w_e * (s_e - U_w) * (t_e - U_w), dim=1) / (H * sigma_w * sigma_w)

    perturb = torch.sum(torch.abs((r_d - r_w) / r_w))
    # print('t_ptb: %s' % (torch.sum(modified_adj - adj_matrix)))
    return perturb, torch.sum(modified_adj - adj_matrix)

def select_nodes_margin(node_list):
    victim_model.eval()
    if args.victim_model == 'gcn':
        output = victim_model.predict(features, adj)
    else:
        output = predict(victim_model, data, args)

    margin_dict = {}
    for idx in node_list:
        margin = classification_margin(output[idx], labels[idx])
        # if margin < 0:  # only keep the nodes correctly classified
        #     continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-70:]]
    other = [x for x, y in sorted_margins[10: -70]]
    other = np.random.choice(other, 20, replace=False).tolist()
    return high + low + other

def select_nodes():
    # remove isolation
    adj_diag = torch.diag(adj)
    adj_ = adj - torch.diag_embed(adj_diag)
    strengths = torch.sum(adj_, dim=1)
    edge_index = torch.nonzero(strengths).t()
    new_list = [int(edge_index[0, i]) for i in range(edge_index.shape[-1])]

    idx_select = []
    for idx in idx_test:
        if idx in new_list and labels[idx] == 1:
            idx_select.append(idx)
    return idx_select


def single_test(adj, features, target_node):
    victim_model.eval()
    data.adj = torch.tensor(adj.todense()).to(device)
    if args.victim_model == 'gcn':
        output = victim_model.predict(features, adj)
    else:
        output = predict(victim_model, data, args)

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

def degree_update(modified_adj, data):
    modified_adj = torch.Tensor(modified_adj.todense()).to(device)
    adj_changes = modified_adj - adj
    s = round(adj_changes.shape[1] / 100)

    add_user_cnt = torch.max(adj_changes)
    add_user_eng = torch.tensor([]).to(device)
    for i in range(int(add_user_cnt.detach().to('cpu').numpy())):
        subgraph_changes = torch.where(adj_changes >= i + 1, 1, 0)
        sj = torch.sum(subgraph_changes)
        index = torch.nonzero(subgraph_changes)
        for n in range(index.shape[0]):
            if index[n][0] >= index[n][1]:
                continue
            row_idx = index[n][0]
            col_idx = index[n][1]

            t = subgraph_changes[row_idx][col_idx]
            subgraph_changes.data[row_idx][col_idx] -= 1
            tmp = torch.zeros((1, A_un.shape[1])).to(device)
            tmp.data[:, row_idx] += torch.randint(low=1, high=s + 1, size=(1,)).to(device)
            tmp.data[:, col_idx] += torch.randint(low=1, high=s + 1, size=(1,)).to(device)
            add_user_eng = torch.cat((add_user_eng, tmp), dim=0)
    A_un_change = torch.cat((A_un.to(device), add_user_eng), dim=0)
    adj_update = A_un_change.transpose(0, 1).matmul(A_un_change)
    # degrees
    xdeg, ydeg = adj_update.sum(0), adj_update.sum(1)
    xdeg = xdeg.view(-1, 1)
    xdeg, ydeg = xdeg.repeat(1, adj_update.shape[0]), ydeg.repeat(adj_update.shape[1], 1)
    data.xdeg = xdeg
    data.ydeg = ydeg

def multi_test_evasion():
    cnt = 0
    l_ptb = []
    g_ptb = []
    t_ptb = []
    if args.subset == 'None':
        file_path_node = r'data/user_news_graph/weighted/{}_node_list.pkl'.format(args.dataset)
    else:
        file_path_node = r'data/user_news_graph/weighted/{}_node_list_{}.pkl'.format(args.dataset, args.subset)
    if os.path.exists(file_path_node):
        print("Read existing node list!")
        node = pickle.load(open(file_path_node, 'rb'))
        node_list = node['node']
    else:
        print("Generate new node list!")
        node = select_nodes()
        node_list = select_nodes_margin(node)
        node_ = {'node': node_list}
        with open(file_path_node, 'wb') as file:
            pickle.dump(node_, file)
    # node_list = select_nodes()
    # node_list = select_nodes_margin(node_list)
    num = len(node_list)
    print('=== [Evasion] Attacking %s nodes respectively ===' % num)
    n_p = 0
    n_temp = 0
    for target_node in tqdm(node_list):
        n_perturbations = min(max(1 * args.m, int(0.2 * args.m * adj[target_node, target_node])), int(user/100.))
        n_p += n_perturbations
        # print('n_perturbations : %.1f' % n_perturbations)
        model = QSAttack(surrogate, victim_model, data=data, args=args, surrogate=args.surrogate, attack_structure=True, attack_features=False, device=device)
        model = model.to(device)
        adj_ = adj.clone().detach()
        model.attack(features, adj_, labels, target_node, n_perturbations, direct=True)
        modified_adj = model.modified_adj

        modified_features = model.modified_features
        if args.victim_model == 'decor':
            degree_update(modified_adj, data)
        l_pt, t_pt = weighted_local_assortativity_value(target_node, adj, modified_adj)
        g_pt, t_pt = weighted_global_assortativity_value(target_node, adj, modified_adj)
        l_ptb.append(l_pt)
        g_ptb.append(g_pt)
        t_ptb.append(np.sum(modified_adj.todense() - adj.to('cpu').numpy()))
        acc = single_test(modified_adj, modified_features, target_node)
        n_temp += 1
        if acc == 0:
            cnt += 1
        # print('temp misclassification rate : %.12f' % (cnt / n_temp))
    print('MIS: %.12f' % (cnt / num))
    print('LSAPR: %.12f' % (sum(l_ptb) / len(l_ptb)))
    # print('global_ptb: %.12f' % (sum(g_ptb) / len(g_ptb)))
    # print('t_ptb: %.12f' % (sum(t_ptb)))
    # print('n_p: %.12f' % (n_p))

    import csv
    f = open("qsa_ac_result.csv", 'a', newline='')
    writer = csv.writer(f)
    writer.writerow(
        [args.dataset] + [args.subset] + [args.victim_model] + ['misclassification rate : %.12f' % (cnt / num)] + [
            'local_ptb: %.12f' % (sum(l_ptb) / len(l_ptb))] + ['global_ptb: %.12f' % (sum(g_ptb) / len(g_ptb))])
    f.close()

if __name__ == '__main__':
    multi_test_evasion()
