import copy
import random

import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torch_scatter import scatter
from sklearn.metrics import average_precision_score as apscore
from sklearn.metrics import roc_auc_score as auroc

def Task1MLPEvaluator(model, X, v1, v2, e1, e2, label) :

    with torch.no_grad() :
        model.eval()
        pred = model(X, v1, v2, e1, e2).squeeze(-1)
        pred = pred.to('cpu').detach().numpy()
        label = label.to('cpu').numpy()
        return apscore(label, pred), auroc(label, pred)


def Task2MLPEvaluator(model, X, ind1, ind2, label):
    with torch.no_grad():
        model.eval()
        pred = model(X, ind1, ind2).squeeze(-1)
        pred = pred.to('cpu').detach().numpy()
        label = label.to('cpu').numpy()
        return apscore(label, pred), auroc(label, pred)

class Task1MLP(nn.Module) :

    def __init__(self, in_dim, hidden_dim, num_layers = 2,
                 device = 'cuda:0', aggregation_type = 'inner_dot') :

        super(Task1MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear_bucket = torch.nn.ModuleList()
        self.agg_type = aggregation_type
        self.dropouts = torch.nn.Dropout(p = 0.5)
        self.device = device
        self.pooling = 'sum'

        self.linear_bucket.append(nn.Linear(in_dim, hidden_dim))

        for i in range(num_layers - 1) :
            self.linear_bucket.append(nn.Linear(hidden_dim, hidden_dim))

        if self.agg_type == 'hadamard' :
            self.last_linear = nn.Linear(hidden_dim, 1)

        elif self.agg_type == 'concat' :
            self.last_linear = nn.Linear(int(2*hidden_dim), 1)

        elif self.agg_type == 'abs_sub' :
            self.last_linear = nn.Linear(int(hidden_dim), 1)

        else :
            None

    def aggregate_Z_for_E(self, Z, V_map1, V_map2, E_map1, E_map2):

        Zv1 = Z[V_map1]
        Zv2 = Z[V_map2]

        Ie1 = torch.tensor(E_map1).to(self.device)
        Ie2 = torch.tensor(E_map2).to(self.device)

        if self.pooling in ['sum', 'mean']:
            Ze1 = scatter(Zv1, Ie1, dim=0, reduce=self.pooling)
            Ze2 = scatter(Zv2, Ie2, dim=0, reduce=self.pooling)
        elif self.pooling in ['maxmin']:
            Ze1 = scatter(Zv1, Ie1, dim=0, reduce='max') - scatter(Zv1, Ie1, dim=0, reduce='min')
            Ze2 = scatter(Zv2, Ie2, dim=0, reduce='max') - scatter(Zv2, Ie2, dim=0, reduce='min')
        return Ze1, Ze2


    def forward(self, x, V_map1, V_map2, E_map1, E_map2):

        for layer in self.linear_bucket :

            x = layer(x)
            x = torch.relu(x)
            x = self.dropouts(x)

        Ze1, Ze2 = self.aggregate_Z_for_E(x, V_map1, V_map2, E_map1, E_map2)

        if self.agg_type == 'abs_sub':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear(torch.abs(Ze1 - Ze2))).squeeze(-1)
        elif self.agg_type == 'hadamard':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear((Ze1 * Ze2))).squeeze(-1)
        elif self.agg_type == 'concat':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear(torch.hstack([Ze1 , Ze2]))).squeeze(-1)
        else:
            pred = torch.sigmoid(torch.sum(Ze1 * Ze2, 1))  ## Inner Product

        return pred

class Task2MLP(nn.Module) :

    def __init__(self, in_dim, hidden_dim, num_layers = 2,
                 device = 'cuda:0', aggregation_type = 'inner_dot') :

        super(Task2MLP, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.linear_bucket = torch.nn.ModuleList()
        self.agg_type = aggregation_type
        self.dropouts = torch.nn.Dropout(p = 0.5)

        self.linear_bucket.append(nn.Linear(in_dim, hidden_dim))

        for i in range(num_layers - 1) :
            self.linear_bucket.append(nn.Linear(hidden_dim, hidden_dim))

        if self.agg_type == 'hadamard' :
            self.last_linear = nn.Linear(hidden_dim, 1)

        elif self.agg_type == 'concat' :
            self.last_linear = nn.Linear(int(2*hidden_dim), 1)

        elif self.agg_type == 'abs_sub' :
            self.last_linear = nn.Linear(int(hidden_dim), 1)

        else :
            None

    def forward(self, x, ind1, ind2) :

        for layer in self.linear_bucket :

            x = layer(x)
            x = torch.relu(x)
            x = self.dropouts(x)

        if self.agg_type == 'abs_sub':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear(torch.abs(x[ind1] - x[ind2]))).squeeze(-1)
        elif self.agg_type == 'hadamard':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear((x[ind1] * x[ind2]))).squeeze(-1)
        elif self.agg_type == 'concat':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear(torch.hstack([x[ind1] , x[ind2]]))).squeeze(-1)
        else:
            pred = torch.sigmoid(torch.sum(x[ind1] * x[ind2], 1))  ## Inner Product

        return pred

def Task1MLPTrainer(model, X, seed, dataloader,
                               lr=0.001, epoch=100, w_decay=1e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    criterion = torch.nn.BCELoss()
    dataloader.create_data(seed)
    train_v1, train_v2 = dataloader.train_V_map
    train_e1, train_e2 = dataloader.train_E_map
    train_labels = dataloader.train_labels

    valid_v1, valid_v2 = dataloader.valid_V_map
    valid_e1, valid_e2 = dataloader.valid_E_map
    valid_labels = dataloader.valid_labels

    test_v1, test_v2 = dataloader.test_V_map
    test_e1, test_e2 = dataloader.test_E_map
    test_labels = dataloader.test_labels

    val_ap = 0

    for ep in tqdm(range(epoch)):

        model.train()
        optimizer.zero_grad()
        outs = model(X, train_v1, train_v2, train_e1, train_e2)
        loss = criterion(outs, train_labels)
        loss.backward()
        optimizer.step()
        

        if ep % 10 == 0:
            cur_val_ap, cur_val_au = Task1MLPEvaluator(model, X, valid_v1, valid_v2,
                                                                  valid_e1, valid_e2, valid_labels)
            

            if cur_val_ap > val_ap:
                params = copy.deepcopy(model.state_dict())
                val_ap = cur_val_ap

    model.load_state_dict(params)
    test_ap, test_auroc = Task1MLPEvaluator(model, X, test_v1, test_v2,
                                                       test_e1, test_e2, test_labels)

    return model, test_ap, test_auroc

def Task2MLPTrainer(model, X,
                               dataloader,
                               device='cuda:0',
                               lr=0.001,
                               epoch=100,
                               w_decay=1e-6,
                               training_batch_size=50000,
                               seed=0,
                               valid_size=100000,
                               test_size=100000,
                               training_process=True):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=w_decay)
    criterion = torch.nn.BCELoss()
    ap_score = 0

    test_idx1, test_idx2, test_label = dataloader.partial_pair_loader(batch_size=test_size,
                                                                      device='cpu', mode='test', seed=seed)

    valid_idx1, valid_idx2, valid_label = dataloader.partial_pair_loader(batch_size=valid_size,
                                                                         device='cpu', mode='valid', seed=seed)

    valid_score = 0


    for ep in tqdm(range(epoch), disable=not training_process):

        batch_data = dataloader.train_batch_loader(batch_size=(training_batch_size),
                                                   device='cpu')
        model.train()

        for ind1, ind2, labels in batch_data:
            optimizer.zero_grad()
            outs = model(X, ind1, ind2)
            loss = criterion(outs, labels.to(device))
            loss.backward()
            optimizer.step()

        if ep % 10 == 0:
            valid_ap, valid_auroc = Task2MLPEvaluator(model, X, valid_idx1, valid_idx2, valid_label)

            if valid_ap > valid_score:
                valid_score = valid_ap
                params = copy.deepcopy(model.state_dict())

    model.load_state_dict(params)
    test_ap, test_auroc = Task2MLPEvaluator(model, X, test_idx1, test_idx2, test_label)

    return model, test_ap, test_auroc

def Task1SSLBatchEvaluator(model, X, v1, v2, e1, e2, label, device):
    batch_v_dict, batch_v, batch_e, batch_label = create_edge_batch(v1, v2,
                                                                    e1, e2,
                                                                    label, device)
    total_pred = []
    total_label = []
    with torch.no_grad():
        model.eval()
        for cur_b in range(len(batch_v_dict)):
            cur_X = X[batch_v_dict[cur_b]].to(device)
            cur_V1, cur_V2 = batch_v[cur_b][0], batch_v[cur_b][1]
            cur_E1, cur_E2 = batch_e[cur_b][0], batch_e[cur_b][1]
            cur_label = batch_label[cur_b]

            cur_pred = list(model(cur_X, cur_V1, cur_V2, cur_E1, cur_E2).to('cpu').detach().squeeze(-1).numpy())
            total_pred.extend(cur_pred)
            total_label.extend(list(cur_label.to('cpu').numpy()))
            del cur_X

    return apscore(total_label, total_pred), auroc(total_label, total_pred)



def create_edge_batch(v1, v2, e1, e2, label, device):
    label = label.to('cpu').numpy()
    prev_e1, prev_e2 = 0, 0
    prev_indptr1, prev_indptr2 = 0, 0

    e_1_dict = dict()
    e_2_dict = dict()

    for i in range(len(v1)):
        cur_e = e1[i]
        if cur_e != prev_e1:
            e_1_dict[prev_e1] = v1[prev_indptr1:i]
            prev_indptr1 = i
            prev_e1 = cur_e
    e_1_dict[prev_e1] = v1[prev_indptr1:]

    for i in range(len(v2)):
        cur_e = e2[i]
        if cur_e != prev_e2:
            e_2_dict[prev_e2] = v2[prev_indptr2:i]
            prev_indptr2 = i
            prev_e2 = cur_e
    e_2_dict[prev_e2] = v2[prev_indptr2:]

    batch_indexer = np.arange(int(e1[-1] + 1))
    batch_indexer = np.random.permutation(batch_indexer)

    if label.shape[0] > 1000000:
        n_bb = 50
    else:
        n_bb = 10

    N_B = int(batch_indexer.shape[0] / n_bb)
    N_START = 0
    batch_splitter = [0]
    for i in range(n_bb):
        N_START += N_B
        batch_splitter.append(N_START)
    batch_splitter.append(int(N_B + 1))

    BATCH_V_MAPPER = []
    BATCH_V_SET = []
    BATCH_E_SET = []
    BATCH_LABELS = []

    for i in range(len(batch_splitter) - 1):

        cur_batch_e = batch_indexer[batch_splitter[i]: batch_splitter[i + 1]]
        cur_V_set = []
        for e in cur_batch_e:
            cur_V_set.extend(e_1_dict[e])
            cur_V_set.extend(e_2_dict[e])
        cur_Vs = list(set(cur_V_set))
        cur_Vs.sort()
        cur_V_dict = {v: t for t, v in enumerate(cur_Vs)}
        cur_B_V = [[], []]
        cur_B_E = [[], []]
        cur_B_label = []
        for i_, e in enumerate(cur_batch_e):
            cur_B_label.append(label[e])
            N_BV1, N_BV2 = 0, 0
            for v_1_ in e_1_dict[e]:
                cur_B_V[0].extend([cur_V_dict[v_1_]])
                N_BV1 += 1
            for v_2_ in e_2_dict[e]:
                cur_B_V[1].extend([cur_V_dict[v_2_]])
                N_BV2 += 1
            cur_B_E[0].extend([i_] * N_BV1)
            cur_B_E[1].extend([i_] * N_BV2)
        BATCH_V_MAPPER.append(cur_Vs)
        BATCH_V_SET.append(cur_B_V)
        BATCH_E_SET.append(cur_B_E)
        BATCH_LABELS.append(torch.tensor(cur_B_label).to(device))

    return BATCH_V_MAPPER, BATCH_V_SET, BATCH_E_SET, BATCH_LABELS


def ScalableTask1MLPTrainer(model, X, seed, dataloader,
                            lr=0.001, epoch=100, w_decay=1e-6, device='cuda:0'):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    criterion = torch.nn.BCELoss()

    train_v1, train_v2 = dataloader.train_V_map
    train_e1, train_e2 = dataloader.train_E_map
    train_labels = dataloader.train_labels

    valid_v1, valid_v2 = dataloader.valid_V_map
    valid_e1, valid_e2 = dataloader.valid_E_map
    valid_labels = dataloader.valid_labels

    test_v1, test_v2 = dataloader.test_V_map
    test_e1, test_e2 = dataloader.test_E_map
    test_labels = dataloader.test_labels

    val_ap = 0

    batch_v_dict, batch_v, batch_e, batch_label = create_edge_batch(train_v1, train_v2,
                                                                    train_e1, train_e2,
                                                                    train_labels, device)

    batch_shuffler = np.arange(len(batch_v_dict))

    for ep in tqdm(range(epoch)):

        batch_shuffler = np.random.permutation(batch_shuffler)

        for cur_b in batch_shuffler:

            cur_X = X[batch_v_dict[cur_b]].to(device)
            cur_V1, cur_V2 = batch_v[cur_b][0], batch_v[cur_b][1]
            cur_E1, cur_E2 = batch_e[cur_b][0], batch_e[cur_b][1]
            cur_label = batch_label[cur_b]

            model.train()
            optimizer.zero_grad()
            outs = model(cur_X, cur_V1, cur_V2, cur_E1, cur_E2)
            loss = criterion(outs, cur_label)
            loss.backward()
            optimizer.step()

            del cur_X

            if ep % 10 == 0:

                cur_val_ap, cur_val_au = Task1SSLBatchEvaluator(model, X, valid_v1, valid_v2,
                                                                valid_e1, valid_e2, valid_labels, device)

                if cur_val_ap > val_ap:
                    params = copy.deepcopy(model.state_dict())
                    val_ap = cur_val_ap

    model.load_state_dict(params)
    test_ap, test_auroc = Task1SSLBatchEvaluator(model, X, test_v1, test_v2,
                                                 test_e1, test_e2, test_labels, device)

    return model, test_ap, test_auroc


def ScalableTask2MLPTrainer(model, X,
                    dataloader,
                    device='cuda:0',
                    lr=0.001,
                    epoch=100,
                    w_decay=1e-6,
                    training_batch_size=50000,
                    seed=0,
                    valid_size=100000,
                    test_size=100000,
                    training_process=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=w_decay)
    criterion = torch.nn.BCELoss()
    ap_score = 0

    test_idx1, test_idx2, test_label = dataloader.partial_pair_loader(batch_size=test_size,
                                                                      device='cpu', mode='test', seed=seed)

    valid_idx1, valid_idx2, valid_label = dataloader.partial_pair_loader(batch_size=valid_size,
                                                                         device='cpu', mode='valid', seed=seed)

    valid_score = 0

    for ep in tqdm(range(epoch), disable=not training_process):

        batch_data = dataloader.train_batch_loader(batch_size=(training_batch_size),
                                                   device=device)
        re_batch_data, mapper = Tasl2MLPBatchReindexer(batch_data)

        model.train()

        for tup, cur_mapper in zip(re_batch_data, mapper):
            ind1, ind2, labels = tup
            optimizer.zero_grad()
            outs = model(X[cur_mapper].to(device), ind1, ind2)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

        if ep % 10 == 0:
            valid_ap, valid_auroc = Task2MLPSplitEvaluator(model, X, valid_idx1, valid_idx2, valid_label, device)

            if valid_ap > valid_score:
                valid_score = valid_ap
                params = copy.deepcopy(model.state_dict())

    model.load_state_dict(params)
    test_ap, test_auroc = Task2MLPSplitEvaluator(model, X, test_idx1, test_idx2, test_label, device)

    return model, test_ap, test_auroc

def Tasl2MLPBatchReindexer(batch) :
    batchwise_node_mapper = []
    for i in range(len(batch)) :
        entire_nodes = list(set(batch[i][0] + batch[i][1]))
        entire_nodes.sort()
        batchwise_node_mapper.append(entire_nodes)
        node_dict = {v:i for i, v in enumerate(entire_nodes)}
        for j in range(len(batch[i][0])) :
            batch[i][0][j] = node_dict[batch[i][0][j]]
            batch[i][1][j] = node_dict[batch[i][1][j]]

    return batch, batchwise_node_mapper

def Task2MLPSplitEvaluator(model, X, ind1, ind2, label, device) :

    if label.shape[0] > 5000000 :
        N_B = 10
    else :
        N_B = 5

    n_length = label.shape[0]
    n_add = int(n_length/N_B)
    S = 0
    split_batch = [0]

    for i in range(N_B) :
        S += n_add
        split_batch.append(S)
    if split_batch[-1] != n_length :
        split_batch.append(n_length)

    pred = []
    with torch.no_grad() :
        model.eval()
        for i in range(len(split_batch) - 1) :
            tmp_ind1 = ind1[split_batch[i] : split_batch[i + 1]]
            tmp_ind2 = ind2[split_batch[i]: split_batch[i + 1]]
            pure_nodes = list(set(tmp_ind1 + tmp_ind2))
            pure_nodes.sort()
            naming = {v : i for i, v in enumerate(pure_nodes)}
            for j in range(len(tmp_ind1)) :
                tmp_ind1[j] = naming[tmp_ind1[j]]
                tmp_ind2[j] = naming[tmp_ind2[j]]

            cur_X = X[pure_nodes].to(device)
            batch_pred = list(model(cur_X, tmp_ind1, tmp_ind2).squeeze(-1).to('cpu').detach().numpy())
            pred.extend(batch_pred)
            del cur_X
        label = label.to('cpu').numpy()
        return apscore(label, pred), auroc(label, pred)