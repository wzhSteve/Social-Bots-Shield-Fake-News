import pickle 
import torch
import pandas as pd
import numpy as np
# from torch_geometric.data import Data

import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['Arial']#如果要显示中文字体，则在此处设为：SimHei
# plt.rcParams['axes.unicode_minus']=False#显示负号
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data():
    def __init__(self):
        self.A = None


def smooth(input):
    remain_size = 4
    window_size = 16
    input_len = input.shape[0]
    input_cp = np.copy(input, order='K', subok=False)
    if remain_size < window_size//2:
        padding_front = np.repeat(input[0], window_size//2 - remain_size)
        input_cp = np.concatenate((padding_front, input_cp))
        padding_behind = np.repeat(input[-1], window_size//2)
        input_cp = np.concatenate((input_cp, padding_behind))
        for i in range(input_len):
          if i < remain_size:
            continue
          else:
            input[i] = np.mean(input_cp[i: i+window_size])


def eig_power(A, eps=1e-15):
    v0 = np.random.rand(A.shape[0])
    vec_norm = np.linalg.norm(v0)
    v0 = torch.tensor(v0 / vec_norm, dtype=torch.float)
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


def draw_picture(input_list, e_val_list, str_list):
    """
    :param input:
    :param str: 标题
    :return:
    """
    # plt.clf()
    color_list = ['black', 'red', 'blue', 'green', 'gray', 'orange']
    x_list = []
    for i in range(len(input_list)):
        input_list[i] = input_list[i][:, 0]
        smooth(input_list[i])
        x_list.append(e_val_list[i][:].tolist())

    plt.figure(figsize=(10, 8))
    #plt.grid(linestyle="--")  # 设置背景网格线为虚线
    ax = plt.gca()
    for i in range(len(input_list)):
        plt.plot(x_list[i], input_list[i], color=color_list[i], label=str_list[i], linewidth=1.5)
    #plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    #plt.yticks([])
    #plt.title("true pred", fontsize=12, fontweight='bold')  # 默认字体大小为12
    #plt.xlabel("pred_len", fontsize=13, fontweight='bold')
    #plt.ylabel("MSE", fontsize=13, fontweight='bold')
    #plt.xlim(0, 3)  # 设置x轴的范围
    #plt.ylim(0.2, 1.2)
    plt.legend(loc=1, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()

    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.show()
    # plt.savefig('figure/' +str + '.pdf')
    # plt.cla()


def eigen_cal(adj):
    rowsum = torch.sum(adj, dim=1)
    D_row = torch.pow(rowsum, -0.5).flatten()
    D_row[torch.isinf(D_row)] = 0.
    D_row = torch.diag(D_row)
    colsum = torch.sum(adj, dim=0)
    D_col = torch.pow(colsum, -0.5).flatten()
    D_col[torch.isinf(D_col)] = 0.
    D_col = torch.diag(D_col)
    DAD = adj.mm(D_col).transpose(0, 1).mm(D_row).transpose(0, 1)

    print('bigest eigen value of DAD: {}'. format(eig_power(DAD)))

    I = torch.eye(DAD.shape[0], DAD.shape[1])
    L = I - DAD

    print('bigest eigen value of L: {}'.format(eig_power(L)))
    adj_np = L.cpu().numpy()

    # # calculate the rank of the adj
    # rank = np.linalg.matrix_rank(adj_np)
    # print('The number of the connected subgraphs in adj is: {}'.format(L.shape[0] - rank))

    e_val, e_vec = np.linalg.eig(adj_np)
    idx = e_val.argsort()
    e_val = e_val[idx]
    e_vec = e_vec[:, idx]
    return e_val, e_vec


def draw_degree(adj, news_features, label, args):

    true_mask, fake_mask = veracity_mask(label)
    # degree distribution
    x_np = news_features.cpu().numpy()
    diag_vector = torch.diag(adj)
    diag_adj = torch.diag_embed(diag_vector)
    adj = adj - diag_adj
    # fake_adj = adj[fake_mask, :]
    # true_adj = adj[true_mask, :]
    # fake_degree = degreeDistribution(fake_adj)
    # true_degree = degreeDistribution(true_adj)
    #
    # str_list = ['true', 'fake']
    # color_list = ['green', 'red']
    # plot_degree([true_degree, fake_degree], color_list, str_list)

    # adjacent nodes' degree
    # adj = torch.where(adj < 1, adj, torch.tensor(1., dtype=adj.dtype).to(adj.device))
    fake_adj = adj[fake_mask, :][:, fake_mask]
    true_adj = adj[true_mask, :][:, true_mask]
    tf_adj = adj[true_mask, :][:, fake_mask]
    ft_adj = tf_adj.T
    fake_degree = degreeDistribution(fake_adj)
    true_degree = degreeDistribution(true_adj)
    tf_degree = degreeDistribution(tf_adj)
    ft_degree = degreeDistribution(ft_adj)
    if len(fake_degree) > len(true_degree):
        flag = 1
    else:
        flag = 0
    l = min(len(fake_degree), len(true_degree))
    same_degree = [0] * l
    for i in range(l):
        same_degree[i] = fake_degree[i] + true_degree[i]
    if flag == 1:
        same_degree += fake_degree[l:]
    else:
        same_degree += true_degree[l:]

    if len(tf_degree) > len(ft_degree):
        flag = 1
    l = min(len(tf_degree), len(ft_degree))
    different_degree = [0] * l
    for i in range(l):
        different_degree[i] = tf_degree[i] + ft_degree[i]
    if flag == 1:
        different_degree += tf_degree[l:]
    else:
        different_degree += ft_degree[l:]


    str_list = ['same', 'different']
    color_list = ['green', 'red']
    plot_degree([same_degree, different_degree], color_list, str_list)

    str_list = ['t_same', 'f_same']
    color_list = ['green', 'red']
    plot_degree([true_degree, fake_degree], color_list, str_list)

    str_list = ['t_different', 'f_different']
    color_list = ['green', 'red']
    plot_degree([tf_degree, ft_degree], color_list, str_list)

    j = 0


def veracity_mask(label):
    true_mask = []
    fake_mask = []
    for i in range(len(label)):
        if label[i] == 0:
            true_mask.append(i)
        else:
            fake_mask.append(i)
    return true_mask, fake_mask


def degreeDistribution(adj_matrix):
    degrees_matrix = torch.sum(adj_matrix, dim=1)
    degrees_matrix = torch.where(degrees_matrix != 0., torch.log(degrees_matrix), torch.tensor(0., dtype=degrees_matrix.dtype))
    max_degree = torch.max(degrees_matrix).to('cpu').numpy()
    distribution = [0] * (int(max_degree) + 1)
    for i in range(degrees_matrix.shape[0]):
        distribution[int(degrees_matrix[i])] += 1
    return distribution

def plot_degree(degree_dist, color_list, str_list):
    plt.figure(figsize=(10, 8))
    # plt.grid(linestyle="--")  # 设置背景网格线为虚线
    for i in range(len(degree_dist)):
        plt.plot(range(len(degree_dist[i])), degree_dist[i], color=color_list[i], label=str_list[i], linewidth=1.5)
    # plt.xticks(x, group_labels, fontsize=12, fontweight='bold')  # 默认字体大小为10
    # plt.yticks([])
    # plt.title("true pred", fontsize=12, fontweight='bold')  # 默认字体大小为12
    # plt.xlabel("pred_len", fontsize=13, fontweight='bold')
    # plt.ylabel("MSE", fontsize=13, fontweight='bold')
    # plt.xlim(0, 3)  # 设置x轴的范围
    # plt.ylim(0.2, 1.2)
    plt.legend(loc=1, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()

    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
    plt.show()


# def weighted_assortativity_value(adj_matrix):
#     # adj_matrix = torch.where(adj_matrix != 0., adj_matrix, torch.tensor(0., dtype=adj_matrix.dtype))
#     # remove self-loop
#     adj_diag = torch.diag(adj_matrix)
#     adj_matrix = adj_matrix - torch.diag_embed(adj_diag)
#     # adj_matrix = torch.where(adj_matrix > 10., adj_matrix, torch.tensor(0., dtype=adj_matrix.dtype, device=device))
#
#     strengths_matrix = torch.sum(adj_matrix, dim=1)
#     max_strengths = torch.max(strengths_matrix).to('cpu').numpy()
#     distribution = [0] * (int(max_strengths) + 1)
#     N = adj_matrix.shape[0]
#     # strength probability
#     P_s = {}
#     for i in range(strengths_matrix.shape[0]):
#         key = int(strengths_matrix[i])
#         distribution[key] += 1
#         if key in P_s.keys():
#             P_s[key] += 1
#         else:
#             P_s[key] = 1
#     for key in P_s.keys():
#         P_s[key] = P_s[key]/N
#
#     # # draw the distribution of weighted strenghts
#     # str_list = ['weighted strenghts']
#     # color_list = ['green', 'red']
#     # plot_degree(distribution, color_list, str_list)
#
#     # sum of all weighted edges
#     H = torch.sum(adj_matrix)
#     # M = torch.sum(torch.where(adj_matrix == 0., adj_matrix, torch.tensor(1., dtype=adj_matrix.dtype, device=device)))
#
#     edge_index = torch.nonzero(adj_matrix).t()
#     edge_weight = torch.zeros(edge_index.shape[1])
#     s_e = torch.zeros(edge_index.shape[1]).to(device)
#     t_e = torch.zeros(edge_index.shape[1]).to(device)
#     for i in range(edge_index.shape[1]):
#         row_idx = edge_index[0][i]
#         col_idx = edge_index[1][i]
#         tmp = adj_matrix[row_idx][col_idx]
#         s_e[i] = strengths_matrix[row_idx] - tmp
#         t_e[i] = strengths_matrix[col_idx] - tmp
#         edge_weight[i] = tmp
#     w_e = edge_weight.to(device)
#
#     U_w = torch.sum(0.5 * w_e * (s_e + t_e)) / H
#     U_w_2 = torch.sum(0.5 * w_e * (s_e * s_e + t_e * t_e)) / H
#     sigma_w = torch.sqrt(U_w_2 - U_w * U_w)
#     r_w = torch.sum(w_e * (s_e - U_w) * (t_e - U_w)) / (H * sigma_w * sigma_w)
#     return r_w
def weighed_assortativity_value(adj_matrix):
    H = torch.sum(adj_matrix)

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


def load_graph_decor(args, u_thres = 3):
    news_features = pickle.load(open('../data/news_features/' + args.dataset_name + '_bert_raw_768d.pkl', 'rb'))
    graph_dict = pickle.load(
        open('../data/user_news_graph/weighted/' + args.dataset_name + '_un_relations_t3_raw.pkl', 'rb'))
    A_un = torch.Tensor(graph_dict['A_un']).to(device)

    if args.ptb_rate == 0:
        attack_adj = None
        A_attack = None
    elif args.adj_only:
        A_attack = None
        graph_dict = pickle.load(
            open('../data/attacked_adj/' + args.dataset_name + '_{}_adj_{}.pkl'.format(args.attack, args.ptb_rate),
                 'rb'))
        attack_adj = graph_dict['mod_adj'].to(device)
    else:
        graph_dict = pickle.load(
            open('../data/attacked_adj/' + args.dataset_name + '_{}_adj_{}.pkl'.format(args.attack, args.ptb_rate),
                 'rb'))
        attack_adj = graph_dict['mod_adj'].to(device)

    mask_dict = pickle.load(open('../data/temp_splits/' + args.dataset_name + '_split.pkl', 'rb'))
    train_mask, val_mask, test_mask = mask_dict['train_mask'], mask_dict['val_mask'], mask_dict['test_mask']
    y_label = mask_dict['label']

    # DECOR thresholds the maximum number of engagement between a certain user and a certain article at 1% of num_news.
    # original
    s = round(A_un.shape[1] / 100)
    A_un_new = torch.where(A_un < s, A_un, torch.tensor(s, dtype=A_un.dtype).to(device))
    adj = A_un_new.transpose(0, 1).matmul(A_un_new)

    # degrees
    xdeg, ydeg = adj.sum(0), adj.sum(1)
    xdeg = xdeg.view(-1, 1)
    xdeg, ydeg = xdeg.repeat(1, adj.shape[0]), ydeg.repeat(adj.shape[1], 1)

    # if A_attack != None:
    #     sa = round(A_un.shape[1] / 100)
    #     A_un_new = torch.where(A_un < s, A_un, torch.tensor(s, dtype=A_un.dtype).to(device))
    #     adj = A_un_new.transpose(0, 1).matmul(A_un_new)
    #
    #     # degrees
    #     xdeg, ydeg = adj.sum(0), adj.sum(1)
    #     xdeg = xdeg.view(-1, 1)
    #     xdeg, ydeg = xdeg.repeat(1, adj.shape[0]), ydeg.repeat(adj.shape[1], 1)

    # co-engagement
    A_un_thres1 = torch.where(A_un < 1, A_un, torch.tensor(1., dtype=A_un.dtype).to(device))
    adj_thres1 = A_un_thres1.transpose(0, 1).matmul(A_un_thres1)

    r_w = weighted_assortativity_value(adj_thres1.to(device))
    print("r_w: {}".format(r_w))
    # draw_degree(adj_thres1.to('cpu'), news_features, y_label, args)

    data = Data()
    data.news_features = news_features.to(device)
    data.adj = adj_thres1.to(device)
    data.attack_adj = attack_adj
    data.xdeg = xdeg.to(device)
    data.ydeg = ydeg.to(device)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.y_label = torch.tensor(y_label, dtype=torch.int64).to(device)
    return data
