import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from gcn import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='gossipcop', choices=['politifact', 'gossipcop'], help='dataset')

args = parser.parse_args()

device = torch.device("cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# load dataset

news_features = pickle.load(open('data/news_features/' + args.dataset + '_bert_raw_768d.pkl', 'rb'))
graph_dict = pickle.load(open('data/user_news_graph/weighted/' + args.dataset + '_un_relations_t3_raw.pkl', 'rb'))
mask_dict = pickle.load(open('data/temp_splits/' + args.dataset + '_train_test_mask_80_20_temp.pkl', 'rb'))

train_mask, test_mask = mask_dict['train_mask'], mask_dict['test_mask']
y_train, y_test = mask_dict['y_train'], mask_dict['y_test']
A_un = torch.Tensor(graph_dict['A_un'])

mask = train_mask.tolist()
y_train = y_train.tolist()
y_test = y_test.tolist()
label = []
n_tr = 0
n_te = 0
for i in range(A_un.shape[1]):
    if mask[i]:
        label.append(y_train[n_tr])
        n_tr = n_tr + 1
    else:
        label.append(y_test[n_te])
        n_te = n_te + 1

a = [i for i in range(A_un.shape[1])]
random_mask = np.random.choice(a, A_un.shape[1], replace=False)

random_label = np.array(label)[random_mask].tolist()
random_features = news_features[random_mask]
random_A_un = A_un[:, random_mask]

dataset_1_len = int(0.33 * len(random_label))
dataset_2_len = int(0.33 * len(random_label))
dataset_3_len = len(random_label) - dataset_1_len - dataset_2_len

label_1 = random_label[:dataset_1_len]
label_2 = random_label[dataset_1_len:dataset_1_len+dataset_2_len]
label_3 = random_label[dataset_1_len+dataset_2_len:]
feature_1 = random_features[:dataset_1_len]
feature_2 = random_features[dataset_1_len:dataset_1_len+dataset_2_len]
feature_3 = random_features[dataset_1_len+dataset_2_len:]

A_un_1 = random_A_un[:, :dataset_1_len]
A_un_2 = random_A_un[:, dataset_1_len:dataset_1_len+dataset_2_len]
A_un_3 = random_A_un[:, dataset_1_len+dataset_2_len:]

idx_1 = [i for i in range(dataset_1_len)]
idx_2 = [i for i in range(dataset_2_len)]
idx_3 = [i for i in range(dataset_3_len)]

random_mask_1 = np.random.choice(idx_1, dataset_1_len, replace=False)
random_mask_2 = np.random.choice(idx_2, dataset_2_len, replace=False)
random_mask_3 = np.random.choice(idx_3, dataset_3_len, replace=False)

train_end_1 = int(0.2 * dataset_1_len)
val_end_1 = int(0.3 * dataset_1_len)

train_end_2 = int(0.2 * dataset_2_len)
val_end_2 = int(0.3 * dataset_2_len)

train_end_3 = int(0.2 * dataset_3_len)
val_end_3 = int(0.3 * dataset_3_len)

train_mask_1 = random_mask_1[:train_end_1]
val_mask_1 = random_mask_1[train_end_1:val_end_1]
test_mask_1 = random_mask_1[val_end_1:]

train_mask_2 = random_mask_2[:train_end_2]
val_mask_2 = random_mask_2[train_end_2:val_end_2]
test_mask_2 = random_mask_2[val_end_2:]

train_mask_3 = random_mask_3[:train_end_3]
val_mask_3 = random_mask_3[train_end_3:val_end_3]
test_mask_3 = random_mask_3[val_end_3:]

data_1 = {'train_mask': train_mask_1, 'val_mask': val_mask_1, 'test_mask': test_mask_1, 'label': label_1, 'features': feature_1, 'A_un': A_un_1}
data_2 = {'train_mask': train_mask_2, 'val_mask': val_mask_2, 'test_mask': test_mask_2, 'label': label_2, 'features': feature_2, 'A_un': A_un_2}
data_3 = {'train_mask': train_mask_3, 'val_mask': val_mask_3, 'test_mask': test_mask_3, 'label': label_3, 'features': feature_3, 'A_un': A_un_3}
with open(r'data/temp_splits/{}_subset1.pkl'.format(args.dataset), 'wb') as file:
    pickle.dump(data_1, file)
with open(r'data/temp_splits/{}_subset2.pkl'.format(args.dataset), 'wb') as file:
    pickle.dump(data_2, file)
with open(r'data/temp_splits/{}_subset3.pkl'.format(args.dataset), 'wb') as file:
    pickle.dump(data_3, file)
