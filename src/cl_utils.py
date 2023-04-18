from itertools import permutations
import random

import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index

    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p

    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
                                hyperedge_index.new_ones((hyperedge_index.shape[1],)),
                                (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index


def clique_expansion(hyperedge_index: Tensor):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)

    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)

class PartitionSampler():

    def __init__(self, partition_batch, init_seed, device, mix_up = False):

        self.device = device
        self.N_partition = len(partition_batch)
        self.partition = partition_batch
        self.mix_up = mix_up

        for batch in self.partition:
            for part_ in ['node_idx', 'hyperedge_index']:
                batch[part_] = batch[part_].to(device)

    def batch_cluster_loader(self, init_seed=0): ## Loading Cluster one by one
        
        np.random.seed(init_seed)
        partition_order = np.arange(self.N_partition)
        partition_order = np.random.permutation(partition_order)

        if self.mix_up : 
            partition_new = [(int(2*i), int(2*i + 1)) for i in range(int(self.N_partition/2))]
            entire_batch = []            
            for i1, i2 in partition_new : 
                entire_batch.append(merge_partition(self.partition, i1, i2, self.device))

        else : 
            np.random.seed(init_seed)
            partition_order = np.arange(self.N_partition)
            partition_order = np.random.permutation(partition_order)
            entire_batch = []
            for i in partition_order:
                cur_partition = self.partition[i]
                entire_batch.append(cur_partition)

        return entire_batch
    
def merge_partition(hypE, e1, e2, device) : 
    
    Ev1 = list(hypE[e1]['node_idx'].to('cpu').numpy())
    Ev2 = list(hypE[e2]['node_idx'].to('cpu').numpy())
    
    global_Ev = list(set(Ev1 + Ev2))
    global_Ev.sort()
    
    global_dict = {v : i for i, v in enumerate(global_Ev)}
    inv_global_dict = {i : v for i, v in enumerate(global_Ev)}
    inv_e1_dict = {i : v for i, v in enumerate(Ev1)}
    inv_e2_dict = {i : v for i, v in enumerate(Ev2)}
    
    HE1 = hypE[e1]['hyperedge_index'][:, torch.argsort(hypE[e1]['hyperedge_index'][1])].to('cpu').numpy()
    HE2 = hypE[e2]['hyperedge_index'][:, torch.argsort(hypE[e2]['hyperedge_index'][1])].to('cpu').numpy()
    
    EDGE = set()
    
    e_idx = 0
    prev_indptr = 0
    for idx1 in range(HE1.shape[1]) : 
        cur_e = HE1[1, idx1]
        if cur_e != e_idx : 
            cur_V = HE1[0, prev_indptr : idx1]
            for i_ in range(int(idx1 - prev_indptr)) : 
                cur_V[i_] = global_dict[inv_e1_dict[cur_V[i_]]]
            EDGE.add(frozenset(cur_V))
            e_idx = cur_e
            prev_indptr = idx1
    
    cur_V = HE1[0, prev_indptr : ]
    for i_ in range(int(idx1 - prev_indptr)) : 
        cur_V[i_] = global_dict[inv_e1_dict[cur_V[i_]]]
    EDGE.add(frozenset(cur_V))
            
    e_idx = 0
    prev_indptr = 0
    for idx1 in range(HE2.shape[1]) : 
        cur_e = HE2[1, idx1]
        if cur_e != e_idx : 
            cur_V = HE2[0, prev_indptr : idx1]
            for i_ in range(int(idx1 - prev_indptr)) : 
                cur_V[i_] = global_dict[inv_e2_dict[cur_V[i_]]]
            EDGE.add(frozenset(cur_V))
            e_idx = cur_e
            prev_indptr = idx1
            
    cur_V = HE2[0, prev_indptr : ]
    for i_ in range(int(idx1 - prev_indptr)) : 
        cur_V[i_] = global_dict[inv_e2_dict[cur_V[i_]]]
    EDGE.add(frozenset(cur_V))
    
    final_EDGE = [[], []]
    
    for i, e in enumerate(list(EDGE)) : 
        cur_V = list(e)
        cur_e = [i] * len(cur_V)
        
        final_EDGE[0].extend(cur_V)
        final_EDGE[1].extend(cur_e)
    
    return {'node_idx' : torch.tensor(global_Ev).to(device), 
           'hyperedge_index' : torch.tensor(final_EDGE).to(device)}
    
