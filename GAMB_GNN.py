import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import pandas as pd
import torch_scatter
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import svm
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import log_loss, mean_squared_error, f1_score, roc_auc_score, accuracy_score, recall_score
from sklearn.preprocessing import label_binarize
import math
import openpyxl
def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


def normalization(adjacency):
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    L = d_hat.dot(adjacency).dot(d_hat).tocoo()
    indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values = torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, L.shape)
    return tensor_adjacency



class DDDataset(object):

    def __init__(self, Data_index):
        sparse_adjacency_1, all_node_features_1, sparse_adjacency_2, all_node_features_2, sparse_adjacency_3, \
        all_node_features_3, sparse_adjacency_4, all_node_features_4, graph_indicator, graph_labels, num_nodes, \
        graph_num, gene_nums = self.read_data_diff(Data_index)
        self.sparse_adjacency_1 = sparse_adjacency_1.tocsr()
        self.all_node_features_1 = all_node_features_1
        self.sparse_adjacency_2 = sparse_adjacency_2.tocsr()
        self.all_node_features_2 = all_node_features_2
        self.sparse_adjacency_3 = sparse_adjacency_3.tocsr()
        self.all_node_features_3 = all_node_features_3
        self.sparse_adjacency_4 = sparse_adjacency_4.tocsr()
        self.all_node_features_4 = all_node_features_4
        self.graph_indicator = graph_indicator
        self.graph_labels = graph_labels
        self.num_nodes = num_nodes
        self.graph_num = graph_num
        self.gene_nums = gene_nums

    def __getitem__(self, index):
        mask = self.graph_indicator == index
        all_node_features = self.all_node_features[mask]
        graph_indicator = self.graph_indicator[mask]
        graph_labels = self.graph_labels[index]
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, all_node_features, graph_indicator, graph_labels

    def __len__(self):
        return len(self.graph_labels)

    def read_data_diff(self, data_index):
        type_index = [0, 1, 2, 3]
        data_names = ['ALL2_200',
                      'ALL4_200',
                      'Leukaemia_200',
                      'Myeloma_200',
                      'Prostate_200',
                      'CNS_200'
                      ]

        dataset_names = ['ALL2',
                         'ALL4',
                         'Leukaemia',
                         'Myeloma',
                         'Prostate',
                         'CNS'
                         ]
        edge_types = ["Co-expression", "Co-localization", "Genetic Interactions", "Physical Interactions"]
        original_data_path = "..\\data\\Microarray_data\\microarray_data.xlsx"
        gene_relat_path = "..\\Network_data\\Final_data_200\\Final_" + data_names[
            data_index] + ".xlsx"
        gene_num_path = "..\\Network_data\\Final_data_200\\" + dataset_names[data_index] + "_num.xlsx"
        name_to_num_path = "..\\Network_data\\Final_data_200\\name_to_num.xlsx"
        original_data = pd.read_excel(original_data_path, sheet_name=dataset_names[data_index], header=None)
        name_to_num = pd.read_excel(name_to_num_path, sheet_name=data_names[data_index])
        graph_labels = list(original_data.iloc[:, 0])
        graph_labels = np.array(graph_labels, dtype=np.int64)
        graph_num = len(graph_labels)
        print("graph_num: ", graph_num)
        top_200 = list(name_to_num['Gene_name'])
        gene_nums = list(name_to_num['Gene_num'])
        num_nodes = len(list(name_to_num['Gene_name']))
        print("num_nodes: ", num_nodes)
        graph_indicator = graph_indicator_generate(graph_num, num_nodes)
        for i in range(len(type_index)):
            gene_relat = pd.read_excel(gene_relat_path, sheet_name=edge_types[type_index[i]])
            gene_num = pd.read_excel(gene_num_path, sheet_name=edge_types[type_index[i]])
            gene1_num = list(gene_num['Gene 1'])
            gene2_num = list(gene_num['Gene 2'])
            gene_name1 = list(gene_relat['Gene 1'])
            gene_name2= list(gene_relat['Gene 2'])
            gene_weight = list(gene_relat['Weight'])
            adjacency_list = subgraph_generate(graph_num, gene1_num, gene2_num, num_nodes)
            if i == 0:
                sparse_adjacency_1 = sp.coo_matrix((np.ones(len(adjacency_list)),
                                                    (adjacency_list[:, 0], adjacency_list[:, 1])),
                                                   shape=(num_nodes * graph_num, num_nodes * graph_num), dtype=np.float32)
                all_node_features_1 = node_feature_generate(num_nodes, top_200, gene_name1, gene_name2, gene_weight,
                                                            graph_num, name_to_num, original_data)
            if i == 1:
                sparse_adjacency_2 = sp.coo_matrix((np.ones(len(adjacency_list)),
                                                    (adjacency_list[:, 0], adjacency_list[:, 1])),
                                                   shape=(num_nodes * graph_num, num_nodes * graph_num), dtype=np.float32)
                all_node_features_2 = node_feature_generate(num_nodes, top_200, gene_name1, gene_name2, gene_weight,
                                                            graph_num, name_to_num, original_data)
            if i == 2:
                sparse_adjacency_3 = sp.coo_matrix((np.ones(len(adjacency_list)),
                                                    (adjacency_list[:, 0], adjacency_list[:, 1])),
                                                   shape=(num_nodes * graph_num, num_nodes * graph_num), dtype=np.float32)
                all_node_features_3 = node_feature_generate(num_nodes, top_200, gene_name1, gene_name2, gene_weight,
                                                            graph_num, name_to_num, original_data)
            if i == 3:
                sparse_adjacency_4 = sp.coo_matrix((np.ones(len(adjacency_list)),
                                                    (adjacency_list[:, 0], adjacency_list[:, 1])),
                                                   shape=(num_nodes * graph_num, num_nodes * graph_num), dtype=np.float32)
                all_node_features_4 = node_feature_generate(num_nodes, top_200, gene_name1, gene_name2, gene_weight,
                                                            graph_num, name_to_num, original_data)

        return sparse_adjacency_1, all_node_features_1, sparse_adjacency_2, all_node_features_2, sparse_adjacency_3, \
               all_node_features_3, sparse_adjacency_4, all_node_features_4, graph_indicator, graph_labels, num_nodes, \
               graph_num, gene_nums



def subgraph_generate(graph_num, gene1_num, gene2_num, num_nodes):
    adjacency_list = []
    graph_indicator = []
    for subgraph_num in range(graph_num):
        for i in range(len(gene1_num)):
            temp = []
            temp.append(gene1_num[i] + subgraph_num * num_nodes)
            temp.append(gene2_num[i] + subgraph_num * num_nodes)
            adjacency_list.append(temp)
        for node in range(num_nodes):
            graph_indicator.append(subgraph_num)

    adjacency_list = np.array(adjacency_list)

    return adjacency_list

def graph_indicator_generate(graph_num, num_nodes):
    graph_indicator = []
    for subgraph_num in range(graph_num):
        for node in range(num_nodes):
            graph_indicator.append(subgraph_num)

    graph_indicator = np.array(graph_indicator, dtype=np.int64)


    return graph_indicator


def node_feature_generate(num_nodes, top_200, gene_name1, gene_name2, gene_weight, graph_num, df4, df1):
    node_features = []
    for i in range(num_nodes):
        temp = [0 for _ in range(num_nodes)]
        node_features.append(temp)
    for i in range(num_nodes):
        begin_gene = top_200[i]
        for j in range(len(gene_name1)):
            if begin_gene == gene_name1[j]:
                for k in range(num_nodes):
                    if gene_name2[j] == top_200[k]:
                        node_features[i][k] = gene_weight[j]
                        break
            if begin_gene == gene_name2[j]:
                for k in range(num_nodes):
                    if gene_name1[j] == top_200[k]:
                        node_features[i][k] = gene_weight[j]
                        break
    for i in range(num_nodes):
        for j in range(num_nodes):
            if node_features[i][j] != 0:
                node_features[j][i] = node_features[i][j]

    all_node_features = []
    for i in range(graph_num):
        all_node_features += node_features
    all_node_features = np.array(all_node_features)

    name_to_num = list(df4['Gene_num'])
    for subgraph_num in range(graph_num):
        for gene_num in range(len(name_to_num)):
            gene_expression = df1.iloc[subgraph_num, name_to_num[gene_num]]
            all_node_features[subgraph_num * num_nodes + gene_num] = [i * gene_expression for i in
                                                                      all_node_features[
                                                                          subgraph_num * num_nodes + gene_num]]
    return all_node_features


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'




def global_max_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_max(x, graph_indicator, dim=0, dim_size=num)[0]


def global_avg_pool(x, graph_indicator):
    num = graph_indicator.max().item() + 1
    return torch_scatter.scatter_mean(x, graph_indicator, dim=0, dim_size=num)


def filter_adjacency(adjacency, mask):
    device = adjacency.device
    mask = mask.cpu().numpy()
    indices = adjacency.coalesce().indices().cpu().numpy()
    num_nodes = adjacency.size(0)
    row, col = indices
    maskout_self_loop = row != col
    row = row[maskout_self_loop]
    col = col[maskout_self_loop]
    sparse_adjacency = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(num_nodes, num_nodes), dtype=np.float32)
    filtered_adjacency = sparse_adjacency[mask, :][:, mask]
    return normalization(filtered_adjacency).to(device)


def top_rank(attention_score, gene_num):
    original_index = []
    sorted_index = []
    for i in range(len(attention_score)):
        original_index.append(i)
        sorted_index.append(attention_score[i][1])
    attention_score = np.array(attention_score)
    attention_score = tensor_from_numpy(attention_score, DEVICE)
    mask = attention_score.new_empty((0,), dtype=torch.bool)
    graph_mask = attention_score.new_zeros((NODE_NUM,),
                                           dtype=torch.bool)
    keep_graph_node_num = gene_num

    graph_mask[original_index[:keep_graph_node_num]] = True
    for i in range(SAMPLE_NUM):
        mask = torch.cat((mask, graph_mask))

    return mask

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, gene_num, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.gene_num = gene_num
        self.activation = activation
        self.attn_gcn = GraphConvolution(input_dim, 1)

    def forward(self, adjacency_1, adjacency_2, adjacency_3, adjacency_4, w_1, w_2, w_3, w_4, input_feature,
                input_feature_1, input_feature_2, input_feature_3, input_feature_4, graph_indicator):
        attn_score_1 = self.attn_gcn(adjacency_1, input_feature_1)
        attn_score_2 = self.attn_gcn(adjacency_2, input_feature_2)
        attn_score_3 = self.attn_gcn(adjacency_3, input_feature_3)
        attn_score_4 = self.attn_gcn(adjacency_4, input_feature_4)
        attn_score = w_1 * attn_score_1 + w_2 * attn_score_2 + w_3 * attn_score_3 + w_4 * attn_score_4
        attn_score = attn_score.squeeze()
        score_n = attn_score.tolist()
        avag_score = gene_score(score_n)
        mask = top_rank(avag_score, self.gene_num)
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1)
        mask_graph_indicator = graph_indicator[mask]
        mask_adjacency_1 = filter_adjacency(adjacency_1, mask)
        mask_adjacency_2 = filter_adjacency(adjacency_2, mask)
        mask_adjacency_3 = filter_adjacency(adjacency_3, mask)
        mask_adjacency_4 = filter_adjacency(adjacency_4, mask)
        return hidden, mask_graph_indicator, mask_adjacency_1, mask_adjacency_2, mask_adjacency_3, mask_adjacency_4, avag_score


def gene_score(score_list):
    result = []
    for i in range(NODE_NUM):
        temp = []
        for j in range(SAMPLE_NUM):
            temp.append(score_list[i + NODE_NUM * j])
        result.append(temp)
    node_score = []
    for i in range(len(result)):
        temp = []
        temp.append(mean(result[i]))
        temp.append(int(i))
        node_score.append(temp)
    node_score.sort(key=lambda x: x[0], reverse = True)
    return node_score


def relation_score(w_1, w_2, w_3, w_4):
    w_1 = w_1.squeeze().tolist()
    w_2 = w_2.squeeze().tolist()
    w_3 = w_3.squeeze().tolist()
    w_4 = w_4.squeeze().tolist()

    r_weight = []
    r_weight.append(mean(w_1))
    r_weight.append(mean(w_2))
    r_weight.append(mean(w_3))
    r_weight.append(mean(w_4))
    return r_weight





class relation_weight(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False):
        super(relation_weight, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, input_feature):
        output = torch.mm(input_feature, self.weight)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'



def tor_sum(W):
    ans = W.tolist()
    ans_list = []
    for i in range(len(ans)):
        ans_list.append(ans[i][0])
    ans_min = min(ans_list)
    ans_max = max(ans_list)
    for i in range(len(ans)):
        ans[i][0] = (ans[i][0] - ans_min) / (ans_max - ans_min)
    ans = np.array(ans)
    return ans


class ModelC(nn.Module):
    def __init__(self, input_dim, hidden_dim, gene_num, num_classes=2):
        super(ModelC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.gene_num = gene_num

        self.gcn1_1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn1_2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn1_3 = GraphConvolution(hidden_dim, hidden_dim)
        self.rw_learn_1 = relation_weight(hidden_dim * 3, 1)

        self.gcn2_1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2_2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn2_3 = GraphConvolution(hidden_dim, hidden_dim)
        self.rw_learn_2 = relation_weight(hidden_dim  * 3, 1)

        self.gcn3_1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn3_2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3_3 = GraphConvolution(hidden_dim, hidden_dim)
        self.rw_learn_3 = relation_weight(hidden_dim * 3, 1)

        self.gcn4_1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn4_2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn4_3 = GraphConvolution(hidden_dim, hidden_dim)
        self.rw_learn_4 = relation_weight(hidden_dim * 3, 1)

        self.pool = SelfAttentionPooling(hidden_dim * 3, gene_num)
        self.fc1 = nn.Linear(hidden_dim * 3 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, adjacency_1, input_feature_1, adjacency_2, input_feature_2, adjacency_3, input_feature_3, adjacency_4, input_feature_4, graph_indicator):
        gcn1_1 = F.relu(self.gcn1_1(adjacency_1, input_feature_1))
        gcn1_2 = F.relu(self.gcn1_2(adjacency_1, gcn1_1))
        gcn1_3 = F.relu(self.gcn1_3(adjacency_1, gcn1_2))

        gcn2_1 = F.relu(self.gcn2_1(adjacency_2, input_feature_2))
        gcn2_2 = F.relu(self.gcn2_2(adjacency_2, gcn2_1))
        gcn2_3 = F.relu(self.gcn2_3(adjacency_2, gcn2_2))

        gcn3_1 = F.relu(self.gcn3_1(adjacency_3, input_feature_3))
        gcn3_2 = F.relu(self.gcn3_2(adjacency_3, gcn3_1))
        gcn3_3 = F.relu(self.gcn3_3(adjacency_3, gcn3_2))

        gcn4_1 = F.relu(self.gcn4_1(adjacency_4, input_feature_4))
        gcn4_2 = F.relu(self.gcn4_2(adjacency_4, gcn4_1))
        gcn4_3 = F.relu(self.gcn4_3(adjacency_4, gcn4_2))

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        gcn_feature_1 = torch.cat((gcn1_1, gcn1_2, gcn1_3), dim=1)
        w_1 = self.rw_learn_1(gcn_feature_1)
        w_1 = tensor_from_numpy(tor_sum(w_1), DEVICE)
        w_1 = w_1.to(torch.float32)

        gcn_feature_2 = torch.cat((gcn2_1, gcn2_2, gcn2_3), dim=1)
        w_2 = self.rw_learn_2(gcn_feature_2)
        w_2 = tensor_from_numpy(tor_sum(w_2), DEVICE)
        w_2 = w_2.to(torch.float32)

        gcn_feature_3 = torch.cat((gcn3_1, gcn3_2, gcn3_3), dim=1)
        w_3 = self.rw_learn_3(gcn_feature_3)
        w_3 = tensor_from_numpy(tor_sum(w_3), DEVICE)
        w_3 = w_3.to(torch.float32)

        gcn_feature_4 = torch.cat((gcn4_1, gcn4_2, gcn4_3), dim=1)
        w_4 = self.rw_learn_4(gcn_feature_4)
        w_4 = tensor_from_numpy(tor_sum(w_4), DEVICE)
        w_4 = w_4.to(torch.float32)

        fenmu = torch.exp(w_1)  + torch.exp(w_2) + torch.exp(w_3) + torch.exp(w_4)

        w_1 = torch.divide(torch.exp(w_1), fenmu)
        w_2 = torch.divide(torch.exp(w_2), fenmu)
        w_3 = torch.divide(torch.exp(w_3), fenmu)
        w_4 = torch.divide(torch.exp(w_4), fenmu)


        new_W = relation_score(w_1, w_2, w_3, w_4)

        final_gcn_feature = torch.mul(w_1, gcn_feature_1) + torch.mul(w_2, gcn_feature_2) + torch.mul(w_3, gcn_feature_3) + torch.mul(w_4, gcn_feature_4)
        pool, pool_graph_indicator, pool_adjacency_1, pool_adjacency_2, pool_adjacency_3, pool_adjacency_4, node_score = \
            self.pool(adjacency_1, adjacency_2, adjacency_3, adjacency_4, w_1, w_2, w_3, w_4, final_gcn_feature, gcn_feature_1,
                      gcn_feature_2, gcn_feature_3, gcn_feature_4,
                                                               graph_indicator)

        readout = torch.cat((global_avg_pool(pool, pool_graph_indicator),
                             global_max_pool(pool, pool_graph_indicator)), dim=1)


        fc1 = F.relu(self.fc1(readout))
        fc2 = F.relu(self.fc2(fc1))
        logits = self.fc3(fc2)
        return logits, new_W, node_score


dataset_names = ['ALL2',
                 'ALL4',
                 'Leukaemia',
                 'Myeloma',
                 'Prostate',
                 'CNS',
              ]



data_index = 0

dataset = DDDataset(data_index)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


adjacency_1 = dataset.sparse_adjacency_1
normalize_adjacency_1 = normalization(adjacency_1).to(DEVICE)
node_features_1 = tensor_from_numpy(dataset.all_node_features_1, DEVICE)
node_features_1 = node_features_1.to(torch.float32)

adjacency_2 = dataset.sparse_adjacency_2
normalize_adjacency_2 = normalization(adjacency_2).to(DEVICE)
node_features_2 = tensor_from_numpy(dataset.all_node_features_2, DEVICE)
node_features_2 = node_features_2.to(torch.float32)

adjacency_3 = dataset.sparse_adjacency_3
normalize_adjacency_3 = normalization(adjacency_3).to(DEVICE)
node_features_3 = tensor_from_numpy(dataset.all_node_features_3, DEVICE)
node_features_3 = node_features_3.to(torch.float32)

adjacency_4 = dataset.sparse_adjacency_4
normalize_adjacency_4 = normalization(adjacency_4).to(DEVICE)
node_features_4 = tensor_from_numpy(dataset.all_node_features_4, DEVICE)
node_features_4 = node_features_4.to(torch.float32)

graph_indicator = tensor_from_numpy(dataset.graph_indicator, DEVICE)
graph_indicator = graph_indicator.to(torch.int64)


INPUT_DIM = node_features_1.size(1)
NUM_CLASSES = 2
EPOCHS = 200    # @param {type: "integer"}

global EP
global NODE_NUM
global SAMPLE_NUM
global GENE_NUMS
NODE_NUM = dataset.num_nodes
SAMPLE_NUM = dataset.graph_num
GENE_NUMS = dataset.gene_nums
print("GENE_NUMS: ", GENE_NUMS)
EP = 0

def metric_mark(data_y, data_preds, data_prob):
    binary_data_prob = data_prob[:, 1]
    mul_data_prob = data_prob
    result_set = list()
    score = accuracy_score(data_y, data_preds)
    if len(set(data_y)) > 2:
        y_one_hot = label_binarize(data_y, np.arange(len(set(data_y))))
        AUC = roc_auc_score(y_one_hot, mul_data_prob, average='micro')
    else:
        AUC = roc_auc_score(data_y, binary_data_prob)
    F1_score = f1_score(data_y, data_preds, labels=None, pos_label=1, average='binary', sample_weight=None)
    Recall_score = recall_score(data_y, data_preds, labels=None, pos_label=1, average='binary', sample_weight=None)
    result_set.append(score)
    result_set.append(AUC)
    result_set.append(Recall_score)
    result_set.append(F1_score)
    return result_set

def train():
    loss_history = []
    val_acc_history = []
    model.train()
    final_w = []
    final_s = []
    max_rate = 0
    max_ans = []
    for epoch in range(EPOCHS):
        logits, node_w, node_s = model(normalize_adjacency_1, node_features_1, normalize_adjacency_2, node_features_2,
                                       normalize_adjacency_3, node_features_3, normalize_adjacency_4, node_features_4,
                                       graph_indicator)
        loss = criterion(logits[train_index], train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = torch.eq(
            logits[train_index].max(1)[1], train_label).float().mean()
        test_ans = test(test_index)
        val_acc = test_ans[0]
        if test_ans[0] > max_rate:
            final_w = node_w
            final_s = node_s
            max_rate = test_ans[0]
            max_ans = test_ans
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))
    return loss_history, val_acc_history, final_w, final_s, max_ans


def test(test_index):
    model.eval()
    with torch.no_grad():
        logits, node_w, node_s = model(normalize_adjacency_1, node_features_1, normalize_adjacency_2, node_features_2,
                                       normalize_adjacency_3, node_features_3, normalize_adjacency_4, node_features_4,
                                       graph_indicator)
        test_logits = logits[test_index]
        prob_score = test_logits.tolist()
        score = []
        for i in range(len(prob_score)):
            temp = []
            s1 = prob_score[i][0]
            s2 = prob_score[i][1]
            e_sum = math.exp(s1) + math.exp(s2)
            s1 = math.exp(s1) / e_sum
            s2 = math.exp(s2) / e_sum
            temp.append(s1)
            temp.append(s2)
            score.append(temp)
        score = np.array(score)
        test_ans = metric_mark(test_label.tolist(), test_logits.max(1)[1].tolist(), score)

    return test_ans


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')
    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()
