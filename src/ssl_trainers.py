import copy
import random

from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import average_precision_score as apscore
from sklearn.metrics import roc_auc_score as auroc


def Task1FullSupervisedEvaluator(model, X, E, v1, v2, e1, e2, label):
    n_node = int(torch.max(E[0]) + 1)
    n_edge = int(torch.max(E[1]) + 1)
    with torch.no_grad():
        model.eval()
        pred = model(X, E, n_node, n_edge, v1, v2, e1, e2).squeeze(-1)
        pred = pred.to('cpu').detach().numpy()
        label = label.to('cpu').numpy()
        return apscore(label, pred), auroc(label, pred)


def Task1PartitionIntraSupervisedEvaluator(model, X, cluster_edges, partition_n_nodes, partition_n_edges,
                                           v1, v2, e1, e2, label, device):
    entire_pred = []
    entire_label = []
    with torch.no_grad():
        model.eval()
        for c in range(len(cluster_edges)):

            cur_C = cluster_edges[c]
            cur_X = X[cur_C['node_idx']].to(device)
            cur_E = cur_C['hyperedge_index'].to(device)
            n_node = partition_n_nodes[c]
            n_edge = partition_n_edges[c]

            cur_valid_v1, cur_valid_v2 = v1[c], v2[c]
            cur_valid_e1, cur_valid_e2 = e1[c], e2[c]
            cur_valid_label = label[c]

            #if len(cur_valid_label) > 2:
            #print(cur_valid_label.size(0))
            if cur_valid_label.size(0) > 2 : 
                pred = model(cur_X, cur_E, n_node, n_edge,
                             cur_valid_v1, cur_valid_v2, cur_valid_e1, cur_valid_e2).squeeze(-1)
                pred = list(pred.to('cpu').detach().numpy())
                cur_label = list(cur_valid_label.to('cpu').numpy())

                entire_pred += pred
                entire_label += cur_label
                
        #print(np.isnan(entire_pred))
        #print(np.isnan(entire_label))

        return apscore(entire_label, entire_pred), auroc(entire_label, entire_pred)

def Task1PartitionInterSupervisedEvaluator(model, X, full_partition, part_partition, v1, v2, e1, e2, label, device =
                                           'cuda:0'):
    with torch.no_grad():
        model.eval()
        '''
        pred = model.partition_test_inference(X, full_partition, part_partition,
                                              v1, v2, e1, e2).squeeze(-1)
        pred = list(pred.to('cpu').detach().numpy())
        label = list(label.to('cpu').numpy())
        return apscore(label, pred), auroc(label, pred)
        '''
        ap1, auroc1 = model.partition_test_inference(X, full_partition, part_partition, v1, v2, e1, e2, label)
        
        return ap1, auroc1
        

def Task2FullSupervisedEvaluator(model, X, hyperedge_index, label, ind1, ind2):
    label = label.to('cpu').numpy()
    N_node, N_edge = int(torch.max(hyperedge_index[0]) + 1), int(torch.max(hyperedge_index[1]) + 1)
    with torch.no_grad():
        model.eval()
        pred = model(X, hyperedge_index, N_node, N_edge,
                     ind1, ind2).to('cpu').detach().squeeze(-1).numpy()
        ap_score = apscore(label, pred)
        auroc_score = auroc(label, pred)

    return ap_score, auroc_score


def Task2PartitionIntraSupervisedEvaluator(model, X, cluster_edges, partition_n_nodes, partition_n_edges,
                                           valid_pairs, device):
    entire_pred = []
    entire_label = []
    with torch.no_grad():
        model.eval()
        for c in range(len(cluster_edges)):

            cur_C = cluster_edges[c]
            cur_X = X[cur_C['node_idx']].to(device)
            cur_E = cur_C['hyperedge_index'].to(device)
            n_node = partition_n_nodes[c]
            n_edge = partition_n_edges[c]

            ind1, ind2, label = valid_pairs[c]

            if len(ind1) > 1 :
                pred = model(cur_X, cur_E, n_node, n_edge,
                             ind1, ind2).squeeze(-1)
                pred = list(pred.to('cpu').detach().numpy())
                cur_label = list(label.to('cpu').numpy())

                entire_pred += pred
                entire_label += cur_label
            
            del cur_X, cur_E

        return apscore(entire_label, entire_pred), auroc(entire_label, entire_pred)


def Task2PartitionInterSupervisedEvaluator(model, X, full_partition, part_partition, ind1, ind2, label):
    with torch.no_grad():
        model.eval()
        pred = model.partition_test_inference(X, full_partition, part_partition,
                                              ind1, ind2).squeeze(-1)
        pred = list(pred.to('cpu').detach().numpy())
        label = list(label.to('cpu').numpy())

        return apscore(label, pred), auroc(label, pred)


def Task1FullSupervisedTrainer(model, X, E, n_node, n_edge, seed, dataloader,
                               lr=0.001, epoch=100, w_decay=1e-6):
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

    for ep in tqdm(range(epoch)):

        model.train()
        optimizer.zero_grad()
        outs = model(X, E, n_node, n_edge,
                     train_v1, train_v2, train_e1, train_e2)
        loss = criterion(outs, train_labels)
        loss.backward()
        optimizer.step()

        if ep % 10 == 0:

            cur_val_ap, cur_val_au = Task1FullSupervisedEvaluator(model, X, E, valid_v1, valid_v2,
                                                                  valid_e1, valid_e2, valid_labels)

            if cur_val_ap > val_ap:
                params = copy.deepcopy(model.state_dict())
                val_ap = cur_val_ap

    model.load_state_dict(params)
    test_ap, test_auroc = Task1FullSupervisedEvaluator(model, X, E, test_v1, test_v2,
                                                       test_e1, test_e2, test_labels)

    return model, test_ap, test_auroc


def Task1PartSupervisedTrainer(model, X, entire_edges, full_partition, part_partition, seed, dataloader, device,
                               infer_type=0, lr=0.001, epoch=100, w_decay=1e-6):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    train_v1, train_v2 = dataloader.train_V_map
    train_e1, train_e2 = dataloader.train_E_map
    train_labels = dataloader.train_labels

    valid_v1, valid_v2 = dataloader.valid_V_map
    valid_e1, valid_e2 = dataloader.valid_E_map
    valid_labels = dataloader.valid_labels

    test_v1, test_v2 = dataloader.test_V_map
    test_e1, test_e2 = dataloader.test_E_map
    test_labels = dataloader.test_labels

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=w_decay)
    criterion = torch.nn.BCELoss()
    partition_order = np.arange(len(full_partition))

    partition_n_nodes = {i: int(torch.max(cur_C['hyperedge_index'][0]) + 1) for i, cur_C in enumerate(full_partition)}
    partition_n_edges = {i: int(torch.max(cur_C['hyperedge_index'][1]) + 1) for i, cur_C in enumerate(full_partition)}
    valid_ap = 0

    for ep in tqdm(range(epoch)):

        partition_order = np.random.permutation(partition_order)
        model.train()

        for c in partition_order:

            optimizer.zero_grad()
            cur_C = full_partition[c]
            cur_X = X[cur_C['node_idx']].to(device)
            cur_E = cur_C['hyperedge_index'].to(device)
            n_node = partition_n_nodes[c]
            n_edge = partition_n_edges[c]

            cur_train_v1, cur_train_v2 = train_v1[c], train_v2[c]
            cur_train_e1, cur_train_e2 = train_e1[c], train_e2[c]
            cur_train_label = train_labels[c]

            if len(cur_train_v1) > 0:
                outs = model(cur_X, cur_E, n_node, n_edge, cur_train_v1, cur_train_v2,
                             cur_train_e1, cur_train_e2)

                loss = criterion(outs, cur_train_label)
                loss.backward()
                optimizer.step()

        if ep % 10 == 0:

            ap1, auroc1 = Task1PartitionIntraSupervisedEvaluator(model, X,
                                                                 full_partition, partition_n_nodes, partition_n_edges,
                                                                 valid_v1, valid_v2,
                                                                 valid_e1, valid_e2, 
                                                                 valid_labels,
                                                                 device=device)

            if ap1 > valid_ap:
                valid_ap = ap1
                params = copy.deepcopy(model.state_dict())

    model.load_state_dict(params)

    if infer_type == 0:
        test_ap, test_auroc = Task1PartitionInterSupervisedEvaluator(model, X, full_partition, part_partition,
                                                                     test_v1, test_v2, test_e1, test_e2, test_labels,
                                                                     device=device)
    else:
        n_node = int(torch.max(entire_edges[0]) + 1)
        n_edge = int(torch.max(entire_edges[1]) + 1)
        test_ap, test_auroc = Task1FullSupervisedEvaluator(model, X, entire_edges,
                                                           test_v1, test_v2, test_e1, test_e2, test_labels)

    return model, test_ap, test_auroc


def Task2FullSupervisedTrainer(model, X, hyperedge_index,
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

    N_node, N_edge = int(torch.max(hyperedge_index[0]) + 1), int(torch.max(hyperedge_index[1]) + 1)

    for ep in tqdm(range(epoch)):

        batch_data = dataloader.train_batch_loader(batch_size=training_batch_size,
                                                   device='cpu')
        model.train()

        for ind1, ind2, labels in batch_data:
            optimizer.zero_grad()
            outs = model(X, hyperedge_index, N_node, N_edge, ind1, ind2)
            loss = criterion(outs, labels.to(device))
            loss.backward()
            optimizer.step()

        if ep % 10 == 0:
            valid_ap, valid_auroc = Task2FullSupervisedEvaluator(model, X, hyperedge_index,
                                                                 valid_label, valid_idx1, valid_idx2)

            if valid_ap > valid_score:
                valid_score = valid_ap
                params = copy.deepcopy(model.state_dict())

    model.load_state_dict(params)
    test_ap, test_auroc = Task2FullSupervisedEvaluator(model, X, hyperedge_index,
                                                       test_label, test_idx1, test_idx2)

    return model, test_ap, test_auroc


def Task2PartitionSupervisedTrainer(model, X, hyperedge_index,
                                    dataloader,
                                    full_partition,
                                    part_partition,
                                    device='cuda:0',
                                    lr=0.001,
                                    epoch=100,
                                    w_decay=1e-6,
                                    seed=0,
                                    test_size=100000,
                                    training_process=True,
                                    infer_type=0):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=w_decay)
    criterion = torch.nn.BCELoss()
    ap_score = 0

    train_pairs = dataloader.train_pairs
    valid_pairs = dataloader.valid_pairs

    test_idx1, test_idx2, test_label = dataloader.partial_pair_loader(batch_size=test_size,
                                                                      device='cpu', seed=seed)

    partition_n_nodes = {i: int(torch.max(cur_C['hyperedge_index'][0]) + 1) for i, cur_C in enumerate(full_partition)}
    partition_n_edges = {i: int(torch.max(cur_C['hyperedge_index'][1]) + 1) for i, cur_C in enumerate(full_partition)}

    valid_score = 0

    N_node, N_edge = int(torch.max(hyperedge_index[0]) + 1), int(torch.max(hyperedge_index[1]) + 1)

    for ep in tqdm(range(epoch)):

        partition_order = np.random.permutation(np.arange(len(train_pairs)))
        model.train()

        for c in partition_order:
            optimizer.zero_grad()
            cur_C = full_partition[c]
            cur_X = X[cur_C['node_idx'], :].to(device)
            cur_E = cur_C['hyperedge_index'].to(device)
            n_node = partition_n_nodes[c]
            n_edge = partition_n_edges[c]
            ind1, ind2, cur_label = train_pairs[c]
            outs = model(cur_X, cur_E, n_node, n_edge, ind1, ind2)
            loss = criterion(outs, cur_label.to(device))
            loss.backward()
            optimizer.step()
            
            del cur_X, cur_E

        if ep % 10 == 0:

            valid_ap, valid_auroc = Task2PartitionIntraSupervisedEvaluator(model, X, full_partition, 
                                                                           partition_n_nodes, partition_n_edges,
                                                                           valid_pairs, device)

            if valid_ap > valid_score:
                valid_score = valid_ap
                params = copy.deepcopy(model.state_dict())

    model.load_state_dict(params)

    if infer_type == 0:  # Clusterwise Approximation Inference
        test_ap, test_auroc = Task2PartitionInterSupervisedEvaluator(model, X, full_partition, part_partition,
                                                                     test_idx1, test_idx2, test_label)

    else:  # Exact Inference
        test_ap, test_auroc = Task2FullSupervisedEvaluator(model, X, hyperedge_index,
                                                           test_label, test_idx1, test_idx2)

    return model, test_ap, test_auroc