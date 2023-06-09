# Dataset Description

## Overview

We provide 5 hypergraph datasets: dblp, trivago, ogbn_mag, aminer, mag

For dblp, trivago, and ogbn_mag datasets, we offer
- data_X_vec.pt
- data_Y.pt
- data_E.pt
- data_split_E.pickle
- data_orig_partition_C.pt
- data_orig_partition_C_PIOS.pt
- data_split_partition_C.pt
- data_split_partition_C_PIOS.pt

For aminer and mag datasets, we offer
- data_X_vec.pt
- data_Y.pt
- data_E.pt
- data_split_E.pickle
- data_orig_partition_C.pt
- data_split_partition_C.pt

## Detailed description

** All indexing method is based on Python. **

1. data_X_vec.pt
This data is a (torch float32 tensor-based) node feature matrix of a dataset.
Each row corresponds to the node, and each column corresponds to one feature.
E.g., 0th-row vector indicates a feature vector of a 0-index node.

2. data_Y.pt
This data is a (torch longtensor-based) node label vector of a dataset.
Each entry implies the label (class) of the corresponding position's node.
E.g., [0, 2, 4, ...] implies 0-index node belongs to class 0, 1-index node belongs to class 2, 
and 2-index node belongs to class 4.

3. data_E.pt
This data is a (torch longtensor based) hyperedge index matrix of a dataset.
The first row indicates the node index, and the second row indicates the edge index.
E.g., torch.tensor([[0, 1, 2, 0, 3], [0, 0, 0, 1, 1]]) implies:
0-index hyperedge consists of 0-index node, 1-index node, and 2-index node. 
1-index hyperedge consists of 0-index node and 3-index node. 
(i.e., $e_0 = \{v_0, v_1, v_2\} , e_1 = \{v_0, v_3\}$)

4. data_split_E.pt
This data is a Python pickle dataset that consists of the four following files:
4.1) First element is a hyperedge index matrix of the split hypergraph whose hyperedges are split from the original hypergraph for Task 1.
     Its format is identical to that of the data_E.pt.
4.2) Second element is a list that implies the index of split hyperedges.
4.3) Third and Fourth elements are lists that imply the ground-truth hyperedge pairs.
We provide examples of the second, third, and fourth elements:
second element: [2, 3, ...]
third element: [1, 5, ...]
fourth element: [7, 10, ...]
- 1-index hyperedge and 7-index hyperedge are ground-truth (split) hyperedges pair, 
and their union is identical to the 2-index hyperedge of the original hypergraph.
- 5-index hyperedge and 10-index hyperedge are ground-truth (split) hyperedges pair, 
and their union is identical to the 3-index hyperedge of the original hypergraph.

5. data_orig_partition_C.pt
This data is about hypergraph partitions: a tuple of a dictionary of hypergraph partitions.
Each element (dictionary) of a tuple is a hypergraph partition.
Each dictionary consists of the following {key:value}:
5.1) (Key) 'node_idx': (Value) Tensor of node index (in the entire hypergraph) of the nodes that belong to the current partition.
5.2) (Key) 'hyperedge_index': (Value) Hyperedge index matrix of the corresponding partition. Its format is identical to that of the data_E.pt.

6. data_split_partition_C.pt
This data has the same format as data_orig_partition_C.pt,
but they are partitions of the split hypergraph (data_split_E.pt), while the above partitions are from the original hypergraph (data_E.pt).

7. data_orig_partition_C_PIOS.pt
This data is similar to the data_orig_partition_C.pt, but P-IOS technique is applied.
Refer to the main paper for more details regarding P-IOS.

8. data_split_partition_C_PIOS.pt
This data is similar to the data_orig_partition_C.pt, but P-IOS technique is applied.
Refer to the main paper for more details regarding P-IOS.