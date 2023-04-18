import random

import numpy as np
import torch
from torch_scatter import scatter_add
from tqdm import tqdm


def load_split_data(EDGEs, init_seed):
    np.random.seed(init_seed)
    EDGEs = EDGEs.to('cpu')
    n_edge = int(torch.max(EDGEs[1]) + 1)
    threshold = 10
    min_degree = 5

    edge_degree = scatter_add(torch.ones(EDGEs.shape[1]),
                              EDGEs[1],
                              dim=0,
                              dim_size=n_edge)

    our_interest_edges = torch.where(edge_degree > threshold)[0]
    split_idx = our_interest_edges.clone().detach()
    split_indicator = {int(v): True for v in split_idx}
    print(len(split_indicator))

    prev_edge_idx = 0
    prev_indptr = 0
    edge_indexer = n_edge
    pos_idx = 0
    new_EDGEs = EDGEs.clone()
    pos_sample1 = [0] * len(our_interest_edges)
    pos_sample2 = [0] * len(our_interest_edges)

    for i, idx in tqdm(enumerate(EDGEs[1])):

        if prev_edge_idx != idx:

            try:
                cond = split_indicator[int(prev_edge_idx)]
            except:
                cond = False

            if cond:
                interest_idx = torch.arange(prev_indptr, i)
                interest_idx = interest_idx[torch.randperm(len(interest_idx))]
                # print(len(interest_idx) - 2 * min_degree)
                num_nodes = min_degree + int(random.randint(0, len(interest_idx) - 2 * min_degree))

                node_idx1 = interest_idx[:num_nodes]
                node_idx2 = interest_idx[num_nodes:]

                new_EDGEs[1][node_idx2] = edge_indexer
                pos_sample1[pos_idx] = int(prev_edge_idx)
                pos_sample2[pos_idx] = edge_indexer
                pos_idx += 1
                edge_indexer += 1

            prev_indptr = i
            prev_edge_idx = idx

    if prev_edge_idx in our_interest_edges:
        interest_idx = torch.arange(prev_indptr, i + 1)
        interest_idx = interest_idx[torch.randperm(len(interest_idx))]

        num_nodes = min_degree + int(random.randint(0, len(interest_idx) - 2 * min_degree))
        node_idx1 = interest_idx[:num_nodes]
        node_idx2 = interest_idx[num_nodes:]

        new_EDGEs[1][node_idx2] = edge_indexer
        pos_sample1[pos_idx] = int(prev_edge_idx)
        pos_sample2[pos_idx] = edge_indexer

    pos_sample1 = np.array(pos_sample1)
    pos_sample2 = np.array(pos_sample2)
    new_EDGEs = torch.tensor(new_EDGEs[:, np.argsort(new_EDGEs[1])])  # Re-Indexing according to hyperedge

    return new_EDGEs, split_idx, pos_sample1, pos_sample2


def ordering_full_hyperedge_index(hyperedge, device):
    order = torch.argsort(hyperedge[1])
    return hyperedge[:, order].to(device)


def ordering_partition_hyperedge_index(partition_dict, device):
    for i in range(len(partition_dict)):
        cur_E = partition_dict[i]['hyperedge_index']
        order = torch.argsort(cur_E[1])
        partition_dict[i]['hyperedge_index'] = cur_E[:, order].to(device)
    return partition_dict

class Task1FullDataLoader():

    def __init__(self, edge_dict, init_seed: int, device: str, split: float):

        self.device = device
        self.seed = init_seed
        self.split = split

        if isinstance(edge_dict, dict) : # Given as dictionary type
            self.pos_sample1 = edge_dict['idx_1']
            self.pos_sample2 = edge_dict['idx_2']
            self.edges = edge_dict['new_edge'].to('cpu').numpy()

        else : # Given as tuple
            self.pos_sample1 = edge_dict[2]
            self.pos_sample2 = edge_dict[3]
            self.edges = edge_dict[0].to('cpu').numpy()

        self.re_ordering = np.argsort(self.edges[1])
        self.edges = self.edges[:, self.re_ordering]

        self.N_edge = int(np.max(self.edges[1]) + 1)
        self.edge2node_dict = dict()
        self.edge_of_interest = []

        prev_indptr = 0
        prev_e = 0
        for i in range(self.edges[0].shape[0]):
            cur_e = int(self.edges[1][i])
            if cur_e != prev_e:
                cur_v = list(self.edges[0][prev_indptr: i])
                self.edge2node_dict[prev_e] = cur_v
                prev_indptr = i
                prev_e = cur_e
        self.edge2node_dict[cur_e] = self.edges[0][prev_indptr:]

        indic_index = 0
        position_dict = {}
        for e in self.edge2node_dict:

            if len(self.edge2node_dict[e]) >= 5:
                position_dict[e] = indic_index
                self.edge_of_interest.extend([e]) # These edges are edges of size greater than 5.

        self.pos_sample1 = np.array(self.pos_sample1)
        self.pos_sample2 = np.array(self.pos_sample2)
        self.N_total = len(self.edge_of_interest)

        ## Tell following edges are from identical pairs
        idx_indicator = 0
        self.are_they_brother = dict()
        for v1, v2 in zip(self.pos_sample1, self.pos_sample2) :
            self.are_they_brother[v1] = idx_indicator
            self.are_they_brother[v2] = idx_indicator
            idx_indicator += 1

    def split_train_valid_test(self, seed=None):

        self.edge_of_interest = {v: 0 for v in self.edge_of_interest}

        if seed != None:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        N_entire = np.arange(self.pos_sample1.shape[0])
        self.train_idx = np.random.choice(a=N_entire,
                                          size=int(N_entire.shape[0] * self.split),
                                          replace=False)

        self.test_idx = np.setdiff1d(N_entire, self.train_idx)

        train_edges = np.hstack([self.pos_sample1[self.train_idx] , self.pos_sample2[self.train_idx]])
        test_edges = np.hstack([self.pos_sample1[self.test_idx] , self.pos_sample2[self.test_idx]])

        for train_v in train_edges :
            self.edge_of_interest[train_v] = 1

        for test_v in test_edges :
            self.edge_of_interest[test_v] = 2

        self.train_pos1, self.train_pos2 = self.pos_sample1[self.train_idx], self.pos_sample2[self.train_idx]
        self.test_pos1, self.test_pos2 = self.pos_sample1[self.test_idx], self.pos_sample2[self.test_idx]

        ## Creating Train Negative Pairs
        E5_1 = np.array(list(self.edge_of_interest.keys()))
        E5_2 = np.array(list(self.edge_of_interest.keys()))
        E5_2 = np.random.permutation(E5_2)

        self.train_neg1, self.train_neg2 = np.ones(self.train_pos1.shape[0]), np.ones(self.train_pos1.shape[0])
        train_idx = 0
        shuffle_idx = 0
        while True :
            cur_cond = False
            cur_v1 = E5_1[shuffle_idx]
            cur_v2 = E5_2[shuffle_idx]
            shuffle_idx += 1
            if self.edge_of_interest[cur_v1] == self.edge_of_interest[cur_v2] == 1 : # Both are Train
                if self.are_they_brother[cur_v1] != self.are_they_brother[cur_v2] :
                    cur_cond = True
            elif (self.edge_of_interest[cur_v1] == 1) or (self.edge_of_interest[cur_v2] == 1) :
                    cur_cond = True

            if cur_cond :
                self.train_neg1[train_idx] = cur_v1
                self.train_neg2[train_idx] = cur_v2
                train_idx += 1
                if train_idx == self.train_neg1.shape[0] :
                    break

            if shuffle_idx == E5_1.shape[0] :
                shuffle_idx = 0
                E5_1 = np.random.permutation(E5_1)
                E5_2 = np.random.permutation(E5_2)

        self.test_neg1, self.test_neg2 = np.ones(self.test_pos1.shape[0]), np.ones(self.test_pos2.shape[0])

        E5_1 = np.array(list(self.edge_of_interest.keys()))
        E5_2 = np.array(list(self.edge_of_interest.keys()))
        E5_2 = np.random.permutation(E5_2)

        test_idx = 0
        shuffle_idx = 0
        while True :
            cur_cond = False
            cur_v1 = E5_1[shuffle_idx]
            cur_v2 = E5_2[shuffle_idx]
            shuffle_idx += 1

            if (self.edge_of_interest[cur_v1] == 1) or (self.edge_of_interest[cur_v2] == 1) : # One of them are Train / Skip
                cur_cond = False
            elif self.edge_of_interest[cur_v1] == self.edge_of_interest[cur_v2] == 2 : # Both are Test
                if self.are_they_brother[cur_v1] != self.are_they_brother[cur_v2] :
                    cur_cond = True
            elif (self.edge_of_interest[cur_v1] == 2) or (self.edge_of_interest[cur_v2] == 2) :
                    cur_cond = True

            if cur_cond :
                self.test_neg1[test_idx] = cur_v1
                self.test_neg2[test_idx] = cur_v2
                test_idx += 1
                if test_idx == self.test_neg1.shape[0] :
                    break

            if shuffle_idx == E5_1.shape[0] :
                shuffle_idx = 0
                E5_1 = np.random.permutation(E5_1)
                E5_2 = np.random.permutation(E5_2)

        true_train_idx = np.random.choice(np.arange(self.train_pos1.shape[0]),
                                          size = int(self.train_pos1.shape[0]/2),
                                          replace = False)

        true_valid_idx = np.delete(np.arange(self.train_pos1.shape[0]), true_train_idx)

        self.valid_pos1, self.valid_pos2 = self.train_pos1[true_valid_idx], self.train_pos2[true_valid_idx]
        self.valid_neg1, self.valid_neg2 = self.train_neg1[true_valid_idx], self.train_neg2[true_valid_idx]

        self.train_pos1, self.train_pos2 = self.train_pos1[true_train_idx], self.train_pos2[true_train_idx]
        self.train_neg1, self.train_neg2 = self.train_neg1[true_train_idx], self.train_neg2[true_train_idx]

    def load_dataset(self, data_type):

        if data_type == 'train':
            pos_ind1, pos_ind2 = self.train_pos1, self.train_pos2
            neg_ind1, neg_ind2 = self.train_neg1, self.train_neg2
        elif data_type == 'valid':
            pos_ind1, pos_ind2 = self.valid_pos1, self.valid_pos2
            neg_ind1, neg_ind2 = self.valid_neg1, self.valid_neg2
        else:  # Test
            pos_ind1, pos_ind2 = self.test_pos1, self.test_pos2
            neg_ind1, neg_ind2 = self.test_neg1, self.test_neg2

        ind1 = list(pos_ind1) + list(neg_ind1)
        ind2 = list(pos_ind2) + list(neg_ind2)

        labels = torch.tensor([1.0] * int(pos_ind1.shape[0]) + [0.0] * int(neg_ind1.shape[0])).to(self.device)

        return ind1, ind2, labels

    def give_edge_to_node_mapper(self, data_type):

        ## This function gives torch-scatter type of indexer
        ## By using this indexing function

        v_index1 = []
        edge_mapper1 = []
        v_index2 = []
        edge_mapper2 = []

        ind1, ind2, labels = self.load_dataset(data_type)
        e_idx = 0

        for e1, e2 in zip(ind1, ind2):
            v1 = self.edge2node_dict[e1]
            v2 = self.edge2node_dict[e2]

            v_index1.extend(v1);
            v_index2.extend(v2)
            edge_mapper1.extend([e_idx] * len(v1))
            edge_mapper2.extend([e_idx] * len(v2))
            e_idx += 1

        if data_type == 'train':
            self.train_V_map = (v_index1, v_index2)
            self.train_E_map = (edge_mapper1, edge_mapper2)
            self.train_labels = labels

        elif data_type == 'valid':
            self.valid_V_map = (v_index1, v_index2)
            self.valid_E_map = (edge_mapper1, edge_mapper2)
            self.valid_labels = labels

        else:
            self.test_V_map = (v_index1, v_index2)
            self.test_E_map = (edge_mapper1, edge_mapper2)
            self.test_labels = labels

    def generate_pairs(self):
        self.give_edge_to_node_mapper('train')
        self.give_edge_to_node_mapper('valid')
        self.give_edge_to_node_mapper('test')

    def create_data(self, seed):
        self.split_train_valid_test(seed)
        self.generate_pairs()


class Task1PartitionDataLoader():

    def __init__(self, edge_dict, full_partition, init_seed, device, split):

        self.device = device
        self.seed = init_seed
        self.split = split

        if isinstance(edge_dict, dict):  # Given as dictionary type
            self.pos_sample1 = edge_dict['idx_1']
            self.pos_sample2 = edge_dict['idx_2']
            self.edges = edge_dict['new_edge'].to('cpu').numpy()

        else:  # Given as tuple
            self.pos_sample1 = edge_dict[2]
            self.pos_sample2 = edge_dict[3]
            self.edges = edge_dict[0].to('cpu').numpy()

        self.re_ordering = np.argsort(self.edges[1])
        self.edges = self.edges[:, self.re_ordering]

        self.N_edge = int(np.max(self.edges[1]) + 1)
        self.edge2node_dict = dict()
        self.N_clus = len(full_partition)

        self.device = device
        self.seed = init_seed
        self.split = split

        if isinstance(edge_dict, dict):  # Given as dictionary type
            self.pos_sample1 = edge_dict['idx_1']
            self.pos_sample2 = edge_dict['idx_2']
            self.edges = edge_dict['new_edge'].to('cpu').numpy()

        else:  # Given as tuple
            self.pos_sample1 = edge_dict[2]
            self.pos_sample2 = edge_dict[3]
            self.edges = edge_dict[0].to('cpu').numpy()

        self.re_ordering = np.argsort(self.edges[1])
        self.edges = self.edges[:, self.re_ordering]

        self.N_edge = int(np.max(self.edges[1]) + 1)
        self.edge2node_dict = dict()
        self.edge_of_interest = []

        prev_indptr = 0
        prev_e = 0
        for i in range(self.edges[0].shape[0]):
            cur_e = int(self.edges[1][i])
            if cur_e != prev_e:
                cur_v = list(self.edges[0][prev_indptr: i])
                self.edge2node_dict[prev_e] = cur_v
                prev_indptr = i
                prev_e = cur_e
        self.edge2node_dict[cur_e] = self.edges[0][prev_indptr:]

        indic_index = 0
        position_dict = {}
        for e in self.edge2node_dict:

            if len(self.edge2node_dict[e]) >= 5:
                position_dict[e] = indic_index
                self.edge_of_interest.extend([e])  # These edges are edges of size greater than 5.

        self.pos_sample1 = np.array(self.pos_sample1)
        self.pos_sample2 = np.array(self.pos_sample2)
        self.N_total = len(self.edge_of_interest)

        ## Tell following edges are from identical pairs
        idx_indicator = 0
        self.are_they_brother = dict()
        for v1, v2 in zip(self.pos_sample1, self.pos_sample2):
            self.are_they_brother[v1] = idx_indicator
            self.are_they_brother[v2] = idx_indicator
            idx_indicator += 1

        for i in range(len(full_partition)):
            full_partition[i]['hyperedge_index'] = full_partition[i]['hyperedge_index'][:,
                                                   torch.argsort(full_partition[i]['hyperedge_index'][1])]

        N_pos = self.pos_sample1.shape[0]

        idx_indicator = 0
        self.are_they_brother = dict()
        for v1, v2 in zip(self.pos_sample1, self.pos_sample2) :
            self.are_they_brother[v1] = idx_indicator
            self.are_they_brother[v2] = idx_indicator
            idx_indicator += 1

        self.edge2cluster, self.edge_clus_ind = self.mapping_edge2cluster(
            full_partition)  ## Tells in which cluster a specific hyperedge is

    def split_train_valid_test(self, seed=None):

        self.edge_of_interest = {v: 0 for v in self.edge_of_interest}

        if seed != None:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)

        N_entire = np.arange(self.pos_sample1.shape[0])
        self.train_idx = np.random.choice(a=N_entire,
                                          size=int(N_entire.shape[0] * self.split),
                                          replace=False)

        self.test_idx = np.setdiff1d(N_entire, self.train_idx)

        train_edges = np.hstack([self.pos_sample1[self.train_idx], self.pos_sample2[self.train_idx]])
        test_edges = np.hstack([self.pos_sample1[self.test_idx], self.pos_sample2[self.test_idx]])

        for train_v in train_edges:
            self.edge_of_interest[train_v] = 1

        for test_v in test_edges:
            self.edge_of_interest[test_v] = 2

        self.train_pos1, self.train_pos2 = self.pos_sample1[self.train_idx], self.pos_sample2[self.train_idx]
        self.test_pos1, self.test_pos2 = self.pos_sample1[self.test_idx], self.pos_sample2[self.test_idx]

        self.test_neg1, self.test_neg2 = np.ones(self.test_pos1.shape[0]), np.ones(self.test_pos1.shape[0])

        E5_1 = np.array(list(self.edge_of_interest.keys()))
        E5_2 = np.array(list(self.edge_of_interest.keys()))
        E5_2 = np.random.permutation(E5_2)

        test_idx = 0
        shuffle_idx = 0
        while True:
            cur_cond = False
            cur_v1 = E5_1[shuffle_idx]
            cur_v2 = E5_2[shuffle_idx]
            shuffle_idx += 1

            if (self.edge_of_interest[cur_v1] == 1) or (
                    self.edge_of_interest[cur_v2] == 1):  # One of them are Train / Skip
                cur_cond = False
            elif self.edge_of_interest[cur_v1] == self.edge_of_interest[cur_v2] == 2:  # Both are Test
                if self.are_they_brother[cur_v1] != self.are_they_brother[cur_v2]:
                    cur_cond = True
            elif (self.edge_of_interest[cur_v1] == 2) or (self.edge_of_interest[cur_v2] == 2):
                cur_cond = True

            if cur_cond:
                self.test_neg1[test_idx] = cur_v1
                self.test_neg2[test_idx] = cur_v2
                test_idx += 1
                if test_idx == self.test_neg1.shape[0]:
                    break

            if shuffle_idx == E5_1.shape[0]:
                shuffle_idx = 0
                E5_1 = np.random.permutation(E5_1)
                E5_2 = np.random.permutation(E5_2)

        true_train_idx = np.random.choice(np.arange(self.train_pos1.shape[0]),
                                          size=int(self.train_pos1.shape[0] / 2),
                                          replace=False)

        true_valid_idx = np.delete(np.arange(self.train_pos1.shape[0]), true_train_idx)

        self.valid_pos1, self.valid_pos2 = self.train_pos1[true_valid_idx], self.train_pos2[true_valid_idx]
        self.train_pos1, self.train_pos2 = self.train_pos1[true_train_idx], self.train_pos2[true_train_idx]

    def mapping_edge2cluster(self, partition_data):

        edge_2_clus = {}
        edge_clus_index = {}  # Tells which index is a corresponding hyperedge in a specific cluster
        self.each_partition_edge = {i: dict() for i in range(len(partition_data))}
        self.each_partition_edge_size = {i: [] for i in range(len(partition_data))}

        for c_num, c in tqdm(enumerate(partition_data)):

            node_IDX = c['node_idx'].numpy()  ## Global Level Mapping
            cur_clus_edges = c['hyperedge_index'].to('cpu').numpy()
            order_idx = np.argsort(cur_clus_edges[1])
            cur_clus_edges = cur_clus_edges[:, order_idx]

            prev_indptr = 0
            prev_e = 0
            for i in range(cur_clus_edges[0].shape[0]):
                cur_e = int(cur_clus_edges[1][i])
                if cur_e != prev_e:
                    self.each_partition_edge[c_num][prev_e] = list(cur_clus_edges[0][prev_indptr: i])
                    self.each_partition_edge_size[c_num].append(cur_clus_edges[0][prev_indptr: i].shape[0])
                    try:
                        edge_2_clus[frozenset(node_IDX[cur_clus_edges[0][prev_indptr: i]])] = [c_num]
                        edge_clus_index[frozenset(node_IDX[cur_clus_edges[0][prev_indptr: i]])] = [prev_e]
                    except:
                        edge_2_clus[frozenset(node_IDX[cur_clus_edges[0][prev_indptr: i]])].extend([c_num])
                        edge_clus_index[frozenset(node_IDX[cur_clus_edges[0][prev_indptr: i]])].extend([prev_e])
                    prev_indptr = i
                    prev_e = cur_e
            self.each_partition_edge[c_num][prev_e] = list(cur_clus_edges[0][prev_indptr:])
            self.each_partition_edge_size[c_num].append(cur_clus_edges[0][prev_indptr:].shape[0])
            try:
                edge_2_clus[frozenset(node_IDX[cur_clus_edges[0][prev_indptr:]])].extend([c_num])
                edge_clus_index[frozenset(node_IDX[cur_clus_edges[0][prev_indptr:]])].extend([prev_e])
            except:
                edge_2_clus[frozenset(node_IDX[cur_clus_edges[0][prev_indptr:]])] = [c_num]
                edge_clus_index[frozenset(node_IDX[cur_clus_edges[0][prev_indptr:]])] = [prev_e]

        self.can_be_negative = dict()

        for i, c in enumerate(self.each_partition_edge_size):
            cur_edge_sizes = self.each_partition_edge_size[c]
            cur_satis = np.where(np.array(cur_edge_sizes) >= 5)[0]
            self.can_be_negative[c] = cur_satis

        return edge_2_clus, edge_clus_index

    def find_partitionwise_samples(self, data_type, n_clus=128, seed = None):
        
        if seed == None : 
            seed = self.seed

        edge2cluster = self.edge2cluster
        edge_clus_ind = self.edge_clus_ind
        
        np.random.seed(self.seed)

        if data_type == 'train':
            self.partitionwise_train_pairs = {i: [] for i in range(n_clus)}
        elif data_type == 'valid':
            self.partitionwise_valid_pairs = {i: [] for i in range(n_clus)}

        if data_type == 'train':
            train_samples1 = [frozenset(self.edge2node_dict[e]) for e in self.train_pos1]
            train_samples2 = [frozenset(self.edge2node_dict[e]) for e in self.train_pos2]

        elif data_type == 'valid':
            train_samples1 = [frozenset(self.edge2node_dict[e]) for e in self.valid_pos1]
            train_samples2 = [frozenset(self.edge2node_dict[e]) for e in self.valid_pos2]
        else:
            raise TypeError('Data Type Mis Match!')

        partition_pairs = {i: [[], []] for i in range(n_clus)}

        for e1, e2 in zip(train_samples1, train_samples2):  ## Mapping Each Positive Clusters

            try : # Is this edge preserved?
                K1 = edge2cluster[e1]
            except :
                continue

            try : # Is this edge preserved?
                K2 = edge2cluster[e2]
            except :
                continue

            two_pairs_union_HE = list(set(K1).intersection(set(K2)))
            if len(two_pairs_union_HE) > 0:
                target_clus = int(two_pairs_union_HE[0])
                edge_idx1 = int(np.where(np.array(edge2cluster[e1]) == target_clus)[0])
                edge_idx2 = int(np.where(np.array(edge2cluster[e2]) == target_clus)[0])
                partition_pairs[target_clus][0].append(edge_clus_ind[e1][edge_idx1])
                partition_pairs[target_clus][1].append(edge_clus_ind[e2][edge_idx2])

        self.partition_pairs = partition_pairs

        for i, c in enumerate(self.can_be_negative):  ## Creating Negative Pairs

            N_pos = len(partition_pairs[c][0])
            if N_pos > 0:
                indexer = {ind: v for v, ind in enumerate(self.can_be_negative[c])}
                indicator = np.arange(self.can_be_negative[c].shape[0])
                it_is_train = np.zeros(indicator.shape[0])

                for p_v1, p_v2 in zip(partition_pairs[c][0], partition_pairs[c][1]):
                    it_is_train[indexer[p_v1]] = 1
                    it_is_train[indexer[p_v2]] = 1
                    indicator[indexer[p_v1]] = indicator[indexer[p_v2]]

                cur_neg_1 = self.can_be_negative[c].copy()
                cur_neg_2 = np.random.permutation(self.can_be_negative[c].copy())

                neg_sample1 = np.zeros(N_pos)
                neg_sample2 = np.zeros(N_pos)

                neg_idx = 0
                shuffle_idx = -1

                while True :
                    shuffle_idx += 1
                    e1, e2 = cur_neg_1[shuffle_idx], cur_neg_2[shuffle_idx]

                    if it_is_train[indexer[e1]] == it_is_train[indexer[e2]] == 1 :

                        if indicator[indexer[e1]] != indicator[indexer[e2]] :
                            neg_sample1[neg_idx] = e1
                            neg_sample2[neg_idx] = e2
                            neg_idx += 1

                    elif (it_is_train[indexer[e1]] == 1)  or (it_is_train[indexer[e2]] == 1) :
                        neg_sample1[neg_idx] = e1
                        neg_sample2[neg_idx] = e2
                        neg_idx += 1

                    if neg_idx == N_pos:
                        break

                    if shuffle_idx == cur_neg_1.shape[0] :
                        shuffle_idx = -1
                        cur_neg_1 = np.random.permutation(cur_neg_1)
                        cur_neg_2 = np.random.permutation(cur_neg_2)

                if data_type == 'train':
                    self.partitionwise_train_pairs[c] = [partition_pairs[c][0], partition_pairs[c][1],
                                                         neg_sample1, neg_sample2]
                elif data_type == 'valid':
                    self.partitionwise_valid_pairs[c] = [partition_pairs[c][0], partition_pairs[c][1],
                                                         neg_sample1, neg_sample2]
            else:
                if data_type == 'train':
                    self.partitionwise_train_pairs[c] = [[], [], [], []]
                elif data_type == 'valid':
                    self.partitionwise_valid_pairs[c] = [[], [], [], []]

    def clusterwise_node_mapper(self, data_type, device):

        if data_type == 'train':
            interest_pairs = self.partitionwise_train_pairs
        elif data_type == 'valid':
            interest_pairs = self.partitionwise_valid_pairs
        else :
            raise TypeError('One of train or valid')

        interest_V1 = {c: [] for c in interest_pairs}
        interest_V2 = {c: [] for c in interest_pairs}
        interest_E1 = {c: [] for c in interest_pairs}
        interest_E2 = {c: [] for c in interest_pairs}
        interest_labels = dict()

        for c in interest_pairs:
            cur_pos_pair1, cur_pos_pair2 = interest_pairs[c][0], interest_pairs[c][1]
            cur_neg_pair1, cur_neg_pair2 = interest_pairs[c][2], interest_pairs[c][3]
            cur_edges = self.each_partition_edge[c]

            edge_idx = 0
            for e1, e2 in zip(cur_pos_pair1, cur_pos_pair2):
                V1, V2 = list(cur_edges[e1]), list(cur_edges[e2])
                E1, E2 = [edge_idx] * len(V1), [edge_idx] * len(V2)
                interest_V1[c].extend(V1);
                interest_V2[c].extend(V2)
                interest_E1[c].extend(E1);
                interest_E2[c].extend(E2)
                edge_idx += 1
            N_pos = edge_idx
            for e1, e2 in zip(cur_neg_pair1, cur_neg_pair2):
                V1, V2 = list(cur_edges[e1]), list(cur_edges[e2])
                E1, E2 = [edge_idx] * len(V1), [edge_idx] * len(V2)
                interest_V1[c].extend(V1);
                interest_V2[c].extend(V2)
                interest_E1[c].extend(E1);
                interest_E2[c].extend(E2)
                edge_idx += 1
            N_neg = edge_idx - N_pos

            interest_labels[c] = torch.tensor([1.0] * N_pos + [0.0] * N_neg).to(device)

        if data_type == 'train':
            self.train_V_map = (interest_V1, interest_V2)
            self.train_E_map = (interest_E1, interest_E2)
            self.train_labels = interest_labels

        if data_type == 'valid':
            self.valid_V_map = (interest_V1, interest_V2)
            self.valid_E_map = (interest_E1, interest_E2)
            self.valid_labels = interest_labels

    def load_test_dataset(self):

        pos_ind1, pos_ind2 = self.test_pos1, self.test_pos2
        neg_ind1, neg_ind2 = self.test_neg1, self.test_neg2

        ind1 = list(pos_ind1) + list(neg_ind1)
        ind2 = list(pos_ind2) + list(neg_ind2)

        labels = torch.tensor([1.0] * int(pos_ind1.shape[0]) + [0.0] * int(neg_ind1.shape[0])).to(self.device)

        return ind1, ind2, labels

    def give_final_test_data(self):

        ## This function gives torch-scatter type of indexer
        ## By using this indexing function

        v_index1 = []
        edge_mapper1 = []
        v_index2 = []
        edge_mapper2 = []

        ind1, ind2, labels = self.load_test_dataset()
        e_idx = 0

        for e1, e2 in zip(ind1, ind2):
            v1 = self.edge2node_dict[e1]
            v2 = self.edge2node_dict[e2]

            v_index1.extend(v1);
            v_index2.extend(v2)
            edge_mapper1.extend([e_idx] * len(v1))
            edge_mapper2.extend([e_idx] * len(v2))
            e_idx += 1

        self.test_V_map = (v_index1, v_index2)
        self.test_E_map = (edge_mapper1, edge_mapper2)
        self.test_labels = labels

    def create_data(self, seed=None) :
        
        if seed == None : 
            seed = self.seed

        self.split_train_valid_test(seed)
        self.find_partitionwise_samples(data_type='train', n_clus=self.N_clus, seed = seed)
        self.find_partitionwise_samples(data_type='valid', n_clus=self.N_clus, seed = seed)
        self.clusterwise_node_mapper(data_type='train', device=self.device)
        self.clusterwise_node_mapper(data_type='valid', device=self.device)
        self.give_final_test_data()