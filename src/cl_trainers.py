import copy

import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score as apscore
from sklearn.metrics import roc_auc_score as auroc

import torch
from torch_scatter import scatter


class Task1CLClassifier(torch.nn.Module):

    def __init__(self, in_dim: int):
        super(Task1CLClassifier, self).__init__()

        self.in_dim = in_dim
        self.projection_head = torch.nn.Linear(self.in_dim, self.in_dim)
        self.linear_mapper = torch.nn.Linear(int(self.in_dim), 1)

    def forward(self, Z1, Z2):
        x1, x2 = torch.relu(self.projection_head(Z1)), torch.relu(self.projection_head(Z2))
        return torch.sigmoid(self.linear_mapper(x1 * x2)).squeeze(-1)

class Task2CLClassifier(torch.nn.Module):

    def __init__(self, in_dim: int):
        super(Task2CLClassifier, self).__init__()

        self.in_dim = in_dim
        self.projection_head = torch.nn.Linear(self.in_dim, self.in_dim)
        self.linear_mapper = torch.nn.Linear(int(self.in_dim), 1)

    def forward(self, Z, ind1, ind2):
        x1, x2 = torch.relu(self.projection_head(Z[ind1])), torch.relu(self.projection_head(Z[ind2]))
        return torch.sigmoid(self.linear_mapper(x1 * x2)).squeeze(-1)

def Task1CLEvaluator(model, E1, E2, label):

    with torch.no_grad():
        model.eval()
        pred = model(E1, E2).squeeze(-1)
        pred = pred.to('cpu').detach().numpy()
        label = label.to('cpu').numpy()
        return apscore(label, pred), auroc(label, pred)
    
def Task1CLEvaluator_batch(model, Z, v1, v2, e1, e2, label, device) :
    label = label.to('cpu').numpy()
    EDGE_IDX = np.max(e1)
    E1_indexer = []
    E2_indexer = []
    interval_ind = int(EDGE_IDX/50)
    begin = 0

    for i in range(50) :
        begin += interval_ind
        E1_indexer.append(begin)
        E2_indexer.append(begin)

    edge_indexing_bucket1 = [[], []]
    edge_indexing_bucket2 = [[], []]

    v1_indptr, v2_indptr = 0, 0

    for i in range(len(e1)) :
        if e1[i] == int(E1_indexer[0] + 1) :
            part_v1 = v1[v1_indptr:i]
            part_e1 = list(np.array(e1[v1_indptr:i]) - min(e1[v1_indptr:i]))
            edge_indexing_bucket1[0].append(part_v1)
            edge_indexing_bucket1[1].append(part_e1)
            v1_indptr = i
            E1_indexer.pop(0)
            if len(E1_indexer) == 0 : 
                break

    part_v1 = v1[v1_indptr:]
    part_e1 = list(np.array(e1[v1_indptr:]) - min(e1[v1_indptr:]))
    edge_indexing_bucket1[0].append(part_v1)
    edge_indexing_bucket1[1].append(part_e1)

    for i in range(len(e2)):
        if e2[i] == int(E2_indexer[0] + 1):
            part_v2 = v2[v2_indptr:i]
            part_e2 = list(np.array(e2[v2_indptr:i]) - min(e2[v2_indptr:i]))
            edge_indexing_bucket2[0].append(part_v2)
            edge_indexing_bucket2[1].append(part_e2)
            v2_indptr = i
            E2_indexer.pop(0)
            if len(E2_indexer) == 0 : 
                break

    part_v2 = v2[v2_indptr:]
    part_e2 = list(np.array(e2[v2_indptr:]) - min(e2[v2_indptr:]))
    edge_indexing_bucket2[0].append(part_v2)
    edge_indexing_bucket2[1].append(part_e2)

    total_loop1 = len(edge_indexing_bucket1[1])
    total_loop2 = len(edge_indexing_bucket2[1])

    if total_loop1 != total_loop2 :
        raise ValueError('BUcket Size Mismatch!')

    with torch.no_grad() :
        model.eval()
        total_prediction = []
        for i in range(total_loop1) :
            v_1, v_2 = edge_indexing_bucket1[0][i], edge_indexing_bucket2[0][i]
            e_1, e_2 = edge_indexing_bucket1[1][i], edge_indexing_bucket2[1][i]
            E1, E2 = bucket_loader(Z, v_1, v_2, e_1, e_2, device, 'sum')
            pred = model(E1, E2).squeeze(-1)
            pred = list(pred.to('cpu').detach().numpy())
            total_prediction.extend(pred)
            del E1, E2
        return apscore(label, total_prediction), auroc(label, total_prediction)

def Task2CLEvaluator(model, Z, ind1, ind2, label):

    with torch.no_grad():
        model.eval()
        pred = model(Z, ind1, ind2).squeeze(-1)
        pred = pred.to('cpu').detach().numpy()
        label = label.to('cpu').numpy()
        return apscore(label, pred), auroc(label, pred)

def bucket_loader(Z, v1, v2, e1, e2, device, pooling):
    Z1 = Z[v1]
    Z2 = Z[v2]

    Ie1 = torch.tensor(e1).to(device)
    Ie2 = torch.tensor(e2).to(device)

    if pooling in ['sum', 'mean']:

        Ze1 = scatter(Z1, Ie1, dim=0, reduce=pooling)
        Ze2 = scatter(Z2, Ie2, dim=0, reduce=pooling)

    elif pooling in ['maxmin']:
        Ze1 = scatter(Z1, Ie1, dim=0, reduce='max') - scatter(Z1, Ie1, dim=0, reduce='min')
        Ze2 = scatter(Z2, Ie2, dim=0, reduce='max') - scatter(Z2, Ie2, dim=0, reduce='min')

    else:
        raise TypeError('wrong pooling!')

    return Ze1, Ze2

def Task1CLTrainer(model, encoder, X, E,  emb_dim, seed,
                                            dataloader, device, pooling, infer_type = 0,
                                            full_partition = None, part_partition = None,
                                             lr=0.001, epoch=100, w_decay=1e-6):

    if infer_type == 0 :
        Zv = torch.zeros((X.shape[0], emb_dim)).to(device)
        cluster_node_mapper = dict()

        c_num = 0
        for c1, c2 in zip(full_partition, part_partition):

            cur_clus_nodes = list(c2['node_idx'].to('cpu').numpy())
            cur_full_clus_mapper = dict()
            cur_clus_index_mapper1 = []
            cur_clus_index_mapper2 = []
            idx = 0
            for idx, v in enumerate(c1['node_idx'].to('cpu').numpy()):
                cur_full_clus_mapper[v] = idx
                idx += 1

            for v in cur_clus_nodes:
                cur_clus_index_mapper1.extend([v])
                cur_clus_index_mapper2.extend([cur_full_clus_mapper[v]])

            cluster_node_mapper[c_num] = [cur_clus_index_mapper1, cur_clus_index_mapper2]
            c_num += 1

        with torch.no_grad():
            encoder.eval()
            for clus_n, cur_clus in enumerate(full_partition):
                n_node = int(torch.max(cur_clus['hyperedge_index'][0]) + 1)
                n_edge = int(torch.max(cur_clus['hyperedge_index'][1]) + 1)
                cur_X = X[cur_clus['node_idx']].to(device)
                cur_E = cur_clus['hyperedge_index'].to(device)
                zv, _ = encoder(cur_X, cur_E, n_node, n_edge)
                Zv[cluster_node_mapper[clus_n][0]] = zv.to('cpu').detach().to(device)[cluster_node_mapper[clus_n][1]]
                del cur_X, cur_E

    else :
        with torch.no_grad():
            encoder.eval()
            n_node = int(torch.max(E[0]) + 1)
            n_edge = int(torch.max(E[1]) + 1)
            Zv, _ = encoder(X, E, n_node, n_edge)

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

    train_E1, train_E2 = bucket_loader(Zv, train_v1, train_v2, train_e1, train_e2, device, pooling)
    valid_E1, valid_E2 = bucket_loader(Zv, valid_v1, valid_v2, valid_e1, valid_e2, device, pooling)
    if X.shape[0] < 1000000 : 
        test_E1, test_E2 = bucket_loader(Zv, test_v1, test_v2, test_e1, test_e2, device, pooling)
    else : 
        None ## Doing Partial Inference
    val_ap = 0

    for ep in tqdm(range(epoch)):

        model.train()
        optimizer.zero_grad()
        outs = model(train_E1, train_E2)
        loss = criterion(outs.squeeze(-1), train_labels)
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:

            cur_val_ap, cur_val_auroc = Task1CLEvaluator(model, valid_E1, valid_E2, valid_labels)
            if cur_val_ap > val_ap:
                params = copy.deepcopy(model.state_dict())
                val_ap = cur_val_ap

    model.load_state_dict(params)
    if X.shape[0] < 1000000 : 
        test_ap, test_auroc = Task1CLEvaluator(model, test_E1, test_E2, test_labels)
    else: 
        test_ap, test_auroc = Task1CLEvaluator_batch(model, Zv, test_v1, test_v2, test_e1, test_e2, test_labels, device)
        
    del train_E1, train_E2, valid_E1, valid_E2, Zv

    return model, test_ap, test_auroc

def Task2CLTrainer(model, encoder, dataloader,
                       X, E,
                   full_partition = None, part_partition = None,
                    infer_type = 0,
                   emb_dim = 128,
                      device = 'cuda:0',
                      lr = 0.001,
                      epoch = 100,
                      w_decay = 1e-6,
                      training_batch_size = 50000,
                      seed = 0,
                      valid_size = 100000,
                      test_size = 1000000,
                      training_process = True) :
    if infer_type == 0:
        Zv = torch.zeros((X.shape[0], emb_dim)).to(device)
        cluster_node_mapper = dict()

        c_num = 0
        for c1, c2 in zip(full_partition, part_partition):

            cur_clus_nodes = list(c2['node_idx'].to('cpu').numpy())
            cur_full_clus_mapper = dict()
            cur_clus_index_mapper1 = []
            cur_clus_index_mapper2 = []
            idx = 0
            for idx, v in enumerate(c1['node_idx'].to('cpu').numpy()):
                cur_full_clus_mapper[v] = idx
                idx += 1

            for v in cur_clus_nodes:
                cur_clus_index_mapper1.extend([v])
                cur_clus_index_mapper2.extend([cur_full_clus_mapper[v]])

            cluster_node_mapper[c_num] = [cur_clus_index_mapper1, cur_clus_index_mapper2]
            c_num += 1

        with torch.no_grad():
            encoder.eval()
            for clus_n, cur_clus in enumerate(full_partition):
                n_node = int(torch.max(cur_clus['hyperedge_index'][0]) + 1)
                n_edge = int(torch.max(cur_clus['hyperedge_index'][1]) + 1)
                cur_X = X[cur_clus['node_idx']].to(device)
                cur_E = cur_clus['hyperedge_index'].to(device)
                zv, _ = encoder(cur_X, cur_E, n_node, n_edge)
                Zv[cluster_node_mapper[clus_n][0]] = zv.to('cpu').detach().to(device)[cluster_node_mapper[clus_n][1]]

    else:
        with torch.no_grad():
            encoder.eval()
            n_node = int(torch.max(E[0]) + 1)
            n_edge = int(torch.max(E[1]) + 1)
            Zv, _ = encoder(X, E, n_node, n_edge)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr, weight_decay=w_decay)
    criterion = torch.nn.BCELoss()
    test_idx1, test_idx2, test_label = dataloader.partial_pair_loader(batch_size=test_size,
                                                                  device='cpu', mode='test', seed = seed)
    valid_idx1, valid_idx2, valid_label = dataloader.partial_pair_loader(batch_size=valid_size,
                                                                     device='cpu', mode='valid', seed = seed)

    valid_score = 0

    for ep in tqdm(range(epoch)) :

        batch_data = dataloader.train_batch_loader(batch_size = training_batch_size,
                                                   device = 'cpu')
        model.train()

        for ind1, ind2, labels in batch_data :
            optimizer.zero_grad()
            outs = model(Zv, ind1, ind2)
            loss = criterion(outs.squeeze(-1), labels.to(device))
            loss.backward()
            optimizer.step()

        if ep % 10 == 0 :
            valid_ap, valid_auroc = Task2CLEvaluator(model, Zv, valid_idx1, valid_idx2, valid_label)

            if valid_ap > valid_score :

                valid_score = valid_ap
                params = copy.deepcopy(model.state_dict())

    model.load_state_dict(params)
    test_ap, test_auroc = Task2CLEvaluator(model, Zv, test_idx1, test_idx2, test_label)

    return model, test_ap, test_auroc