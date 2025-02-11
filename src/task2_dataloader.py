import time
from itertools import combinations

from tqdm import tqdm
import numpy as np
import torch

class Task2FullGlobalDataLoader():

    def __init__(self, labels, edge_index, init_seed, device, sampling=True, split_ratio=10,
                 n_train_per_each=10000,
                 n_valid_per_each=10000,
                 n_test_per_each=10000):
        np.random.seed(init_seed)  ## Fixing seed
        torch.manual_seed(init_seed)  ## Fixing seed

        self.N_NODE = labels.shape[0]
        self.device = device
        self.LABEL, self.LABEL_to_NODE = np.unique(labels.to('cpu').numpy(),
                                                   return_inverse=True)
        self.N_LABEL = self.LABEL.shape[0]
        
        self.truncated_label = np.setdiff1d(self.LABEL, np.array([-1]))
        np.random.seed(init_seed)
        self.train_venue = list(np.random.choice(a=self.truncated_label,
                                                 size=int(self.N_LABEL * (split_ratio / 2)),
                                                 replace=False))

        valid_candidate = np.setdiff1d(self.truncated_label, self.train_venue)
        self.valid_venue = list(np.random.choice(a=valid_candidate,
                                                 size=int(self.N_LABEL * (split_ratio / 2)),
                                                 replace=False))

        self.test_venue = list(np.setdiff1d(valid_candidate, self.valid_venue))

        self.train_venue.sort()
        self.valid_venue.sort()
        self.test_venue.sort()

        self.train_labelwise_taxonomy = {i: [] for i in self.train_venue}
        self.valid_labelwise_taxonomy = {i: [] for i in self.valid_venue}
        self.test_labelwise_taxonomy = {i: [] for i in self.test_venue}

        ## Index 0: Train / Index 1: Validation / Index 2: Test
        self.TrainValidTest_Indexer = {v : 0 for v in self.LABEL}

        for v in (self.LABEL):

            if np.isin(v, self.train_venue):
                continue
            elif v == -1 : 
                self.TrainValidTest_Indexer[v] = -1
            elif np.isin(v, self.valid_venue):
                self.TrainValidTest_Indexer[v] = 1
            else:
                self.TrainValidTest_Indexer[v] = 2

        for i in range(labels.shape[0]):

            cur_v_label = self.LABEL[self.LABEL_to_NODE[i]]

            if self.TrainValidTest_Indexer[cur_v_label] == -1 : 
                continue
            elif self.TrainValidTest_Indexer[cur_v_label] == 0:
                self.train_labelwise_taxonomy[cur_v_label].append(i)
            elif self.TrainValidTest_Indexer[cur_v_label] == 1:
                self.valid_labelwise_taxonomy[cur_v_label].append(i)
            else:
                self.test_labelwise_taxonomy[cur_v_label].append(i)

        train_positive_pair1, train_positive_pair2 = [], []
        valid_positive_pair1, valid_positive_pair2 = [], []
        test_positive_pair1, test_positive_pair2 = [], []

        self.entire_train_nodes = []
        self.entire_valid_nodes = []
        self.entire_test_nodes = []

        np.random.seed(init_seed)
        for i in tqdm(self.train_venue):
            cur_interest = self.train_labelwise_taxonomy[i]
            self.entire_train_nodes.extend(cur_interest)
            if (len(cur_interest)**2)/2 < n_train_per_each : 
                result = list(combinations(cur_interest, 2))
                for v1, v2 in result : 
                    train_positive_pair1.extend([v1])
                    train_positive_pair2.extend([v2])
            else : 
                IDX1 = list(np.random.choice(a=cur_interest, size=n_train_per_each))
                IDX2 = list(np.random.choice(a=cur_interest, size=n_train_per_each))

                IDX_candid1 = list(np.random.choice(a=cur_interest, size=10*len(IDX1)))
                IDX_candid2 = list(np.random.choice(a=cur_interest, size=10*len(IDX2)))

                candid_idx = 0

                for i in range(len(IDX1)):
                    while True:
                        if IDX1[i] != IDX2[i]:
                            break
                        else:
                            IDX1[i] = IDX_candid1[candid_idx]
                            IDX2[i] = IDX_candid2[candid_idx]
                            candid_idx += 1
                train_positive_pair1.extend(IDX1)
                train_positive_pair2.extend(IDX2)

        for i in tqdm(self.valid_venue):
            cur_interest = self.valid_labelwise_taxonomy[i]
            self.entire_valid_nodes.extend(cur_interest)
            
            if (len(cur_interest)**2)/2 < n_valid_per_each : 
                result = list(combinations(cur_interest, 2))
                for v1, v2 in result : 
                    valid_positive_pair1.extend([v1])
                    valid_positive_pair2.extend([v2])
            else : 
                IDX1 = list(np.random.choice(a=cur_interest, size=n_valid_per_each))
                IDX2 = list(np.random.choice(a=cur_interest, size=n_valid_per_each))

                IDX_candid1 = list(np.random.choice(a=cur_interest, size=10*len(IDX1)))
                IDX_candid2 = list(np.random.choice(a=cur_interest, size=10*len(IDX2)))

                candid_idx = 0

                for i in range(len(IDX1)):
                    while True:
                        if IDX1[i] != IDX2[i]:
                            break
                        else:
                            IDX1[i] = IDX_candid1[candid_idx]
                            IDX2[i] = IDX_candid2[candid_idx]
                            candid_idx += 1
                valid_positive_pair1.extend(IDX1)
                valid_positive_pair2.extend(IDX2)

        train_negative_pair1, train_negative_pair2 = train_positive_pair1[:], train_positive_pair2[:]
        valid_negative_pair1, valid_negative_pair2 = valid_positive_pair1[:], valid_positive_pair2[:]

        train_negative_pair2 = np.random.permutation(train_negative_pair2)
        valid_negative_pair2 = np.random.permutation(valid_negative_pair2)

        if self.LABEL.shape[0] < 10:
            train_neg_candid = np.random.choice(self.entire_train_nodes, int(2000 * len(train_positive_pair1)))
            valid_neg_candid = np.random.choice(self.entire_valid_nodes, int(2000 * len(valid_positive_pair1)))
        elif self.LABEL.shape[0] < 200 : 
            train_neg_candid = np.random.choice(self.entire_train_nodes, int(100 * len(train_positive_pair1)))
            valid_neg_candid = np.random.choice(self.entire_valid_nodes, int(100 * len(valid_positive_pair1)))
        else:
            train_neg_candid = np.random.choice(self.entire_train_nodes, int(10 * len(train_positive_pair1)))
            valid_neg_candid = np.random.choice(self.entire_valid_nodes, int(10 * len(valid_positive_pair1)))

        neg_idx = -1
        pair_idx = -1
        for v1, v2 in tqdm(zip(train_negative_pair1, train_negative_pair2)):
            pair_idx += 1
            if self.LABEL_to_NODE[v1] == self.LABEL_to_NODE[v2]:
                while True:
                    neg_idx += 1
                    if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[train_neg_candid[neg_idx]]:
                        train_negative_pair2[pair_idx] = train_neg_candid[neg_idx]
                        break

        neg_idx = -1
        pair_idx = -1
        for v1, v2 in tqdm(zip(valid_negative_pair1, valid_negative_pair2)):
            pair_idx += 1
            if self.LABEL_to_NODE[v1] == self.LABEL_to_NODE[v2]:
                while True:
                    neg_idx += 1
                    if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[valid_neg_candid[neg_idx]]:
                        valid_negative_pair2[pair_idx] = valid_neg_candid[neg_idx]
                        break

        np.random.seed(init_seed)
        for i in tqdm(self.test_venue):
            cur_interest = self.test_labelwise_taxonomy[i]
            self.entire_test_nodes.extend(cur_interest)
            
            if (len(cur_interest)**2)/2 < n_test_per_each : 
                result = list(combinations(cur_interest, 2))
                for v1, v2 in result : 
                    test_positive_pair1.extend([v1])
                    test_positive_pair2.extend([v2])
                
            else : 
                
                IDX1 = list(np.random.choice(a=cur_interest, size=n_test_per_each))
                IDX2 = list(np.random.choice(a=cur_interest, size=n_test_per_each))

                IDX_candid1 = list(np.random.choice(a=cur_interest, size=len(IDX1)))
                IDX_candid2 = list(np.random.choice(a=cur_interest, size=len(IDX2)))

                candid_idx = 0

                for i in range(len(IDX1)):
                    while True:
                        if IDX1[i] != IDX2[i]:
                            break
                        else:
                            IDX1[i] = IDX_candid1[candid_idx]
                            IDX2[i] = IDX_candid2[candid_idx]
                            candid_idx += 1
                test_positive_pair1.extend(IDX1)
                test_positive_pair2.extend(IDX2)

        test_negative_pair1, test_negative_pair2 = test_positive_pair1[:], test_positive_pair2[:]
        test_negative_pair2 = np.random.permutation(test_negative_pair2)
        test_neg_candid = np.random.choice(self.entire_test_nodes, int(len(test_positive_pair1) * 2))

        neg_idx = -1
        pair_idx = -1
        for v1, v2 in tqdm(zip(test_negative_pair1, test_negative_pair2)):
            pair_idx += 1
            if self.LABEL_to_NODE[v1] == self.LABEL_to_NODE[v2]:
                while True:
                    neg_idx += 1
                    if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[test_neg_candid[neg_idx]]:
                        test_negative_pair2[pair_idx] = test_neg_candid[neg_idx]
                        break

        self.train_positive_pair = np.array([train_positive_pair1, train_positive_pair2])
        self.train_negative_pair = np.array([train_negative_pair1, train_negative_pair2])
        self.valid_positive_pair = np.array([valid_positive_pair1, valid_positive_pair2])
        self.valid_negative_pair = np.array([valid_negative_pair1, valid_negative_pair2])
        self.test_positive_pair = np.array([test_positive_pair1, test_positive_pair2])
        self.test_negative_pair = np.array([test_negative_pair1, test_negative_pair2])

    def partial_pair_loader(self, batch_size, seed, device='cpu', mode='train'):

        if mode == 'train':
            interest_pos_set = self.train_positive_pair
            interest_neg_set = self.train_negative_pair
        elif mode == 'valid':
            interest_pos_set = self.valid_positive_pair
            interest_neg_set = self.valid_negative_pair
        else:
            interest_pos_set = self.test_positive_pair
            interest_neg_set = self.test_negative_pair

        np.random.seed(seed)
        if int(batch_size / 2) < min(interest_pos_set.shape[1], interest_neg_set.shape[1]):

            pos_idxs = np.random.choice(a=np.arange(interest_pos_set.shape[1]),
                                        size=int(batch_size / 2),
                                        replace=False)
            neg_idxs = np.random.choice(a=np.arange(interest_neg_set.shape[1]),
                                        size=int(batch_size / 2),
                                        replace=False)

            IND1 = list(interest_pos_set[0, pos_idxs]) + list(interest_neg_set[0, neg_idxs])
            IND2 = list(interest_pos_set[1, pos_idxs]) + list(interest_neg_set[1, neg_idxs])
            N_pos, N_neg = int(batch_size / 2), int(batch_size / 2)

        else:
            IND1 = list(interest_pos_set[0]) + list(interest_neg_set[0])
            IND2 = list(interest_pos_set[1]) + list(interest_neg_set[1])
            N_pos, N_neg = int(interest_pos_set[0].shape[0]), int(interest_neg_set[0].shape[0])

        label = torch.tensor([1.0] * int(N_pos) + [0.0] * int(N_neg)).to(device)

        return IND1, IND2, label

    def train_batch_loader(self, batch_size, device='cpu'):

        n_pos = self.train_positive_pair.shape[1]
        n_neg = self.train_negative_pair.shape[1]

        if min(n_pos, n_neg) > batch_size:

            n_tmp_batch = min(int(n_pos // batch_size), int(n_neg // batch_size))

            n_pos_arg = np.arange(n_pos)
            n_neg_arg = np.arange(n_neg)

            np.random.shuffle(n_pos_arg)
            np.random.shuffle(n_neg_arg)

            remain_V1 = np.setdiff1d(n_pos_arg, n_pos_arg[:int(n_tmp_batch * batch_size)])
            remain_V2 = np.setdiff1d(n_neg_arg, n_neg_arg[:int(n_tmp_batch * batch_size)])

            n_pos_arg = n_pos_arg[:int(n_tmp_batch * batch_size)]
            n_neg_arg = n_neg_arg[:int(n_tmp_batch * batch_size)]

            batches = []

            pos_IDXS = np.split(n_pos_arg, n_tmp_batch)
            neg_IDXS = np.split(n_neg_arg, n_tmp_batch)

            interest_pos_set = self.train_positive_pair
            interest_neg_set = self.train_negative_pair

            for i in range(n_tmp_batch):
                pos_idxs = pos_IDXS[i]
                neg_idxs = neg_IDXS[i]

                IND1 = list(interest_pos_set[0, pos_idxs]) + list(interest_neg_set[0, neg_idxs])
                IND2 = list(interest_pos_set[1, pos_idxs]) + list(interest_neg_set[1, neg_idxs])

                label = torch.tensor([1.0] * int(pos_idxs.shape[0]) + [0.0] * int(neg_idxs.shape[0])).to(device)
                batches.append((IND1, IND2, label))

            if min(remain_V1.shape[0], remain_V2.shape[0]) > 0:
                pos_idxs = remain_V1
                neg_idxs = remain_V2

                IND1 = list(interest_pos_set[0, pos_idxs]) + list(interest_neg_set[0, neg_idxs])
                IND2 = list(interest_pos_set[1, pos_idxs]) + list(interest_neg_set[1, neg_idxs])

                label = torch.tensor([1.0] * int(pos_idxs.shape[0]) + [0.0] * int(neg_idxs.shape[0])).to(device)
                batches.append((IND1, IND2, label))

        else:
            interest_pos_set = self.train_positive_pair
            interest_neg_set = self.train_negative_pair

            IND1 = list(interest_pos_set[0]) + list(interest_neg_set[0])
            IND2 = list(interest_pos_set[1]) + list(interest_neg_set[1])

            label = torch.tensor([1.0] * int(interest_pos_set.shape[1]) + [0.0] * int(interest_neg_set.shape[1])).to(
                device)

            batches = [(IND1, IND2, label)]

        return batches


class Task2PartitionGlobalDataLoader():

    def __init__(self, labels, init_seed, device, partitioned_result, full_partition, split_ratio=10,
                 n_train_per_each=10000, n_valid_per_each=10000, n_test_per_each=10000):

        np.random.seed(init_seed)  ## Fixing seed
        torch.manual_seed(init_seed)  ## Fixing seed

        self.N_NODE = labels.shape[0]
        self.device = device
        self.LABEL, self.LABEL_to_NODE = np.unique(labels.to('cpu').numpy(),
                                                   return_inverse=True)
        self.N_LABEL = self.LABEL.shape[0]

        np.random.seed(init_seed)
        self.train_venue = list(np.random.choice(a=self.LABEL,
                                                 size=int(self.N_LABEL * (split_ratio / 2)),
                                                 replace=False))

        valid_candidate = np.setdiff1d(self.LABEL, self.train_venue)
        self.valid_venue = list(np.random.choice(a=valid_candidate,
                                                 size=int(self.N_LABEL * (split_ratio / 2)),
                                                 replace=False))

        self.test_venue = list(np.setdiff1d(valid_candidate, self.valid_venue))

        self.train_venue.sort();
        self.valid_venue.sort();
        self.test_venue.sort()

        self.train_labelwise_taxonomy = {i: [] for i in self.train_venue}
        self.valid_labelwise_taxonomy = {i: [] for i in self.valid_venue}
        self.test_labelwise_taxonomy = {i: [] for i in self.test_venue}

        partition_node_index_mapper = {i: dict() for i in range(len(full_partition))}

        for i, c in enumerate(full_partition):
            cur_idx = 0
            for v in c['node_idx'].to('cpu').numpy():
                partition_node_index_mapper[i][v] = cur_idx
                cur_idx += 1

        ## Index 0: Train / Index 1: Validation / Index 2: Test
        self.TrainValidTest_Indexer = np.zeros(int(np.max(self.LABEL) + 1))

        for v in (self.LABEL):

            if np.isin(v, self.train_venue):
                continue
            elif np.isin(v, self.valid_venue):
                self.TrainValidTest_Indexer[v] = 1
            else:
                self.TrainValidTest_Indexer[v] = 2

        for i in range(labels.shape[0]):

            cur_v_label = self.LABEL[self.LABEL_to_NODE[i]]

            if self.TrainValidTest_Indexer[cur_v_label] == 0:
                self.train_labelwise_taxonomy[cur_v_label].append(i)

            elif self.TrainValidTest_Indexer[cur_v_label] == 1:
                self.valid_labelwise_taxonomy[cur_v_label].append(i)

            else:
                self.test_labelwise_taxonomy[cur_v_label].append(i)

        test_positive_pair1, test_positive_pair2 = [], []

        self.entire_train_nodes, self.entire_valid_nodes, self.entire_test_nodes = [], [], []

        train_partition_positive_dictionary = {i: {v: [] for v in self.train_venue} for i in
                                               range(len(partitioned_result))}
        train_partition_negative_dictionary = dict()

        valid_partition_positive_dictionary = {i: {v: [] for v in self.valid_venue} for i in
                                               range(len(partitioned_result))}
        valid_partition_negative_dictionary = dict()

        train_partition_positive_pairs, train_partition_negative_pairs = dict(), dict()
        valid_partition_positive_pairs, valid_partition_negative_pairs = dict(), dict()

        for i in range(len(partitioned_result)):
            train_partition_positive_pairs[i] = [[], []]
            train_partition_negative_pairs[i] = [[], []]
            valid_partition_positive_pairs[i] = [[], []]
            valid_partition_negative_pairs[i] = [[], []]

        for i in tqdm(self.train_venue):
            cur_interest = self.train_labelwise_taxonomy[i]
            self.entire_train_nodes.extend(cur_interest)

        for i in tqdm(self.valid_venue):
            cur_interest = self.valid_labelwise_taxonomy[i]
            self.entire_valid_nodes.extend(cur_interest)

        self.TRAIN_V = self.entire_train_nodes[:]
        self.VALID_V = self.entire_valid_nodes[:]

        self.entire_train_nodes = set(self.entire_train_nodes)
        self.entire_valid_nodes = set(self.entire_valid_nodes)

        for cur_p in range(len(partitioned_result)):

            cur_p_V = set(partitioned_result[cur_p]['node_idx'].to('cpu').numpy().copy())

            ## Training
            cur_train_nodes = cur_p_V.intersection(self.entire_train_nodes)
            self.entire_train_nodes = self.entire_train_nodes - cur_train_nodes
            cur_train_nodes = list(cur_train_nodes)
            this_parts_labels = self.LABEL_to_NODE[list(cur_train_nodes)]

            for i, lab in enumerate(this_parts_labels):  ## I am here
                train_partition_positive_dictionary[cur_p][lab].append(cur_train_nodes[i])
            train_partition_negative_dictionary[cur_p] = cur_train_nodes

            ## Validation
            cur_valid_nodes = cur_p_V.intersection(self.entire_valid_nodes)
            self.entire_valid_nodes = self.entire_valid_nodes - cur_valid_nodes
            cur_valid_nodes = list(cur_valid_nodes)
            this_parts_labels = self.LABEL_to_NODE[list(cur_valid_nodes)]

            for i, lab in enumerate(this_parts_labels):
                valid_partition_positive_dictionary[cur_p][lab].append(cur_valid_nodes[i])
            valid_partition_negative_dictionary[cur_p] = cur_valid_nodes

        for cur_p in range(len(partitioned_result)):

            for e in train_partition_positive_dictionary[cur_p]:
                cur_V = train_partition_positive_dictionary[cur_p][e]
                if (len(cur_V) >= 2) & (np.unique(self.LABEL_to_NODE[cur_V]).shape[0] > 1):
                    cur_v1, cur_v2 = [], []
                    N_pos_here = 0
                    if (len(cur_V) * (len(cur_V) - 1)) / 2 < n_train_per_each:
                        all_pos_pairs = list(combinations(cur_V, 2))
                        for v1, v2 in all_pos_pairs:
                            cur_v1.append(partition_node_index_mapper[cur_p][v1]);
                            cur_v2.append(partition_node_index_mapper[cur_p][v2])
                            N_pos_here += 1
                    else:
                        N_pos_here = n_train_per_each
                        all_pos_pairs1 = np.random.choice(cur_V, N_pos_here)
                        all_pos_pairs2 = np.random.choice(cur_V, N_pos_here)
                        for v1, v2 in zip(all_pos_pairs1, all_pos_pairs2):
                            cur_v1.append(partition_node_index_mapper[cur_p][v1]);
                            cur_v2.append(partition_node_index_mapper[cur_p][v2])
                            N_pos_here += 1

                    train_partition_positive_pairs[cur_p][0].extend(cur_v1)
                    train_partition_positive_pairs[cur_p][1].extend(cur_v2)

                    neg_ind1 = np.random.choice(train_partition_negative_dictionary[cur_p], int(5 * N_pos_here))
                    neg_ind2 = np.random.permutation(neg_ind1)

                    N_neg_here = 0
                    for v1, v2 in zip(neg_ind1, neg_ind2):
                        if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[v2]:
                            train_partition_negative_pairs[cur_p][0].extend([partition_node_index_mapper[cur_p][v1]])
                            train_partition_negative_pairs[cur_p][1].extend([partition_node_index_mapper[cur_p][v2]])
                            N_neg_here += 1
                            if N_neg_here == N_pos_here:
                                break

            ## Same process to validation pairs here.
            for e in valid_partition_positive_dictionary[cur_p]:
                cur_V = valid_partition_positive_dictionary[cur_p][e]
                N_pos_here = 0
                if len(cur_V) >= 2:
                    cur_v1, cur_v2 = [], []
                    N_pos_here = 0
                    if (len(cur_V) * (len(cur_V) - 1)) / 2 < n_train_per_each:
                        all_pos_pairs = list(combinations(cur_V, 2))
                        for v1, v2 in all_pos_pairs:
                            cur_v1.append(partition_node_index_mapper[cur_p][v1]);
                            cur_v2.append(partition_node_index_mapper[cur_p][v2])
                            N_pos_here += 1
                    else:
                        N_pos_here = n_train_per_each
                        all_pos_pairs1 = np.random.choice(cur_V, N_pos_here)
                        all_pos_pairs2 = np.random.choice(cur_V, N_pos_here)
                        for v1, v2 in zip(all_pos_pairs1, all_pos_pairs2):
                            cur_v1.append(partition_node_index_mapper[cur_p][v1]);
                            cur_v2.append(partition_node_index_mapper[cur_p][v2])
                            N_pos_here += 1

                    valid_partition_positive_pairs[cur_p][0].extend(cur_v1)
                    valid_partition_positive_pairs[cur_p][1].extend(cur_v2)

                neg_ind1 = np.random.choice(valid_partition_negative_dictionary[cur_p], int(5 * N_pos_here))
                neg_ind2 = np.random.permutation(neg_ind1)

                N_neg_here = 0
                for v1, v2 in zip(neg_ind1, neg_ind2):
                    if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[v2]:
                        valid_partition_negative_pairs[cur_p][0].extend([partition_node_index_mapper[cur_p][v1]])
                        valid_partition_negative_pairs[cur_p][1].extend([partition_node_index_mapper[cur_p][v2]])
                        N_neg_here += 1
                        if N_neg_here == N_pos_here:
                            break

        np.random.seed(init_seed)
        for i in tqdm(self.test_venue):
            cur_interest = self.test_labelwise_taxonomy[i]
            self.entire_test_nodes.extend(cur_interest)
            
            if (len(cur_interest)**2)/2 < n_test_per_each : 
                
                result = list(combinations(cur_interest, 2))
                for v1, v2 in result : 
                    test_positive_pair1.extend([v1])
                    test_positive_pair2.extend([v2])
                
            else : 
                
                IDX1 = list(np.random.choice(a=cur_interest, size=n_test_per_each))
                IDX2 = list(np.random.choice(a=cur_interest, size=n_test_per_each))

                IDX_candid1 = list(np.random.choice(a=cur_interest, size=len(IDX1)))
                IDX_candid2 = list(np.random.choice(a=cur_interest, size=len(IDX2)))

                candid_idx = 0

                for i in range(len(IDX1)):
                    while True:
                        if IDX1[i] != IDX2[i]:
                            break
                        else:
                            IDX1[i] = IDX_candid1[candid_idx]
                            IDX2[i] = IDX_candid2[candid_idx]
                            candid_idx += 1
                test_positive_pair1.extend(IDX1)
                test_positive_pair2.extend(IDX2)

        ## Generating Positivie Pair
        test_negative_pair1, test_negative_pair2 = test_positive_pair1[:], test_positive_pair2[:]
        test_negative_pair2 = np.random.permutation(test_negative_pair2)
        test_neg_candid = np.random.choice(self.entire_test_nodes, int(len(test_positive_pair1) * 2))

        neg_idx = -1
        pair_idx = -1
        for v1, v2 in tqdm(zip(test_negative_pair1, test_negative_pair2)):
            pair_idx += 1
            if self.LABEL_to_NODE[v1] == self.LABEL_to_NODE[v2]:
                while True:
                    neg_idx += 1
                    if self.LABEL_to_NODE[v1] != self.LABEL_to_NODE[test_neg_candid[neg_idx]]:
                        test_negative_pair2[pair_idx] = test_neg_candid[neg_idx]
                        break

        self.train_pairs = {
            i: self.give_partitionwise_pairs(train_partition_positive_pairs, train_partition_negative_pairs, i) for i in
            range(len(train_partition_positive_pairs))}
        self.valid_pairs = {
            i: self.give_partitionwise_pairs(valid_partition_positive_pairs, valid_partition_negative_pairs, i) for i in
            range(len(valid_partition_positive_pairs))}

        self.test_positive_pair = np.array([test_positive_pair1, test_positive_pair2])
        self.test_negative_pair = np.array([test_negative_pair1, test_negative_pair2])

    def give_partitionwise_pairs(self, pos_pairs, neg_pairs, partition_idx):

        N_pos = len(pos_pairs[partition_idx][0])
        N_neg = len(neg_pairs[partition_idx][0])

        ind1 = pos_pairs[partition_idx][0] + neg_pairs[partition_idx][0]
        ind2 = pos_pairs[partition_idx][1] + neg_pairs[partition_idx][1]

        labels = torch.tensor([1.0] * N_pos + [0.0] * N_neg).to(self.device)

        return ind1, ind2, labels

    def partial_pair_loader(self, batch_size, seed, device='cpu'):

        interest_pos_set = self.test_positive_pair
        interest_neg_set = self.test_negative_pair

        np.random.seed(seed)
        if int(batch_size / 2) < min(interest_pos_set.shape[1], interest_neg_set.shape[1]):

            pos_idxs = np.random.choice(a=np.arange(interest_pos_set.shape[1]),
                                        size=int(batch_size / 2),
                                        replace=False)
            neg_idxs = np.random.choice(a=np.arange(interest_neg_set.shape[1]),
                                        size=int(batch_size / 2),
                                        replace=False)

            IND1 = list(interest_pos_set[0, pos_idxs]) + list(interest_neg_set[0, neg_idxs])
            IND2 = list(interest_pos_set[1, pos_idxs]) + list(interest_neg_set[1, neg_idxs])
            N_pos, N_neg = int(batch_size / 2), int(batch_size / 2)

        else:
            N_pos, N_neg = interest_pos_set[0].shape[0], interest_neg_set[0].shape[0]
            IND1 = list(interest_pos_set[0]) + list(interest_neg_set[0])
            IND2 = list(interest_pos_set[1]) + list(interest_neg_set[1])

        label = torch.tensor([1.0] * N_pos + [0.0] * N_neg).to(device)

        return IND1, IND2, label