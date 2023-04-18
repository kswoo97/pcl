from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from sklearn.metrics import average_precision_score as apscore
from sklearn.metrics import roc_auc_score as auroc

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class MeanPoolingConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0,
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False,
                 row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm

        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot')
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot')

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)

            if self.row_norm:
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]

            else:
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                De_inv_sqrt = De.pow(-0.5)
                De_inv_sqrt[De_inv_sqrt == float('inf')] = 0

                norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_n2e = norm
                norm_e2n = norm

            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        x = self.lin_n2e(x)
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e,
                           size=(num_nodes, num_edges))  # Node to edge

        if self.bias_n2e is not None:
            e = e + self.bias_n2e
        e = self.act(e)
        e = F.dropout(e, p=self.dropout, training=self.training)

        x = self.lin_e2n(e)
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n,
                           size=(num_edges, num_nodes))  # Edge to node

        if self.bias_e2n is not None:
            n = n + self.bias_e2n

        return n, e  # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j


class Task1SupervisedModel(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, device, pooling_method, num_layers=2,
                 aggregation_type='inner_dot', act: Callable = nn.PReLU()):
        super(Task1SupervisedModel, self).__init__()

        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act
        self.dropouts = torch.nn.Dropout(p=0.5)
        self.convs = nn.ModuleList()
        self.device = device
        self.pooling = pooling_method

        self.agg_type = aggregation_type  # How to produce last target scalar
        if self.agg_type == 'concat':
            self.last_linear = torch.nn.Linear(int(2 * node_dim), 1)
        elif self.agg_type in ['hadamard', 'abs_sub']:
            self.last_linear = torch.nn.Linear(int(node_dim), 1)
        elif self.agg_type == 'inner_dot':
            self.last_linear = None
        else:
            raise TypeError('Given Wrong Aggregation Type')

        if num_layers == 1:
            self.convs.append(MeanPoolingConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        else:
            self.convs.append(MeanPoolingConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(MeanPoolingConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            self.convs.append(MeanPoolingConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

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
    
    def Task1CLEvaluator_batch(self, Z, v1, v2, e1, e2, label, device) :
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
            total_prediction = []
            for i in range(total_loop1) :
                v_1, v_2 = edge_indexing_bucket1[0][i], edge_indexing_bucket2[0][i]
                e_1, e_2 = edge_indexing_bucket1[1][i], edge_indexing_bucket2[1][i]
                E1, E2 = self.aggregate_Z_for_E(Z, v_1, v_2, e_1, e_2)
                pred = torch.sigmoid(torch.sum(E1 * E2, 1))  ## Inner Product
                pred = list(pred.to('cpu').detach().numpy())
                total_prediction.extend(pred)
                del E1, E2
            return apscore(label, total_prediction), auroc(label, total_prediction)

    def forward(self, x, hyperedge_index, num_nodes, num_edges,
                V_map1, V_map2, E_map1, E_map2):
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
            x = self.dropouts(x)

        Ze1, Ze2 = self.aggregate_Z_for_E(x, V_map1, V_map2, E_map1, E_map2)
        if self.agg_type == 'inner_dot':
            pred = torch.sigmoid(torch.sum(Ze1 * Ze2, 1))  ## Inner Product
        elif self.agg_type == 'concat':
            pred = torch.sigmoid(self.last_linear(torch.hstack([Ze1, Ze2]))).squeeze(-1)  ## Inner Product.
        elif self.agg_type == 'abs_sub':
            pred = torch.sigmoid(self.last_linear(torch.abs(Ze1 - Ze2))).squeeze(-1)  ## Inner Product.
        else:
            pred = torch.sigmoid(self.last_linear(Ze1 * Ze2)).squeeze(-1)  ## Inner Product.

        return pred

    def return_node_embeddings(self, X, full_partition, part_partition):

        partition_mapper = {i: [[], []] for i in range(len(part_partition))}

        i = 0
        for c1, c2 in zip(full_partition, part_partition):
            full_partition_indexer = dict()
            cur_part_nodes = c1['node_idx'].to('cpu').numpy()
            this_partition_nodes = c2['node_idx'].to('cpu').numpy()
            cur_idx = 0
            for v in cur_part_nodes:
                full_partition_indexer[v] = cur_idx
                cur_idx += 1

            for v in this_partition_nodes:
                partition_mapper[i][0].append(v)
                partition_mapper[i][1].append(full_partition_indexer[v])

            i += 1

        Zv = torch.zeros((X.shape[0], self.node_dim)).to(self.device)

        for c_num, e in enumerate(full_partition):

            part_hypE = e['hyperedge_index'].to(self.device)
            part_v_idx = e['node_idx']
            cur_x = X[part_v_idx].to(self.device)
            n_node = int(torch.max(part_hypE[0]) + 1)
            n_edge = int(torch.max(part_hypE[1]) + 1)

            for i in range(self.num_layers):
                if i == 0:
                    x, e = self.convs[i](cur_x, part_hypE, n_node, n_edge)
                    x = self.act(x)
                else:
                    x, e = self.convs[i](x, part_hypE, n_node, n_edge)
                    x = self.act(x)

            Zv[partition_mapper[c_num][0], :] = x[partition_mapper[c_num][1], :]
            
            del x, part_hypE, cur_x

        return Zv.to('cpu').detach().to(self.device)

    def partition_test_inference(self, x, full_partition, part_partition,
                                 V_map1, V_map2, E_map1, E_map2, label):

        Zv = self.return_node_embeddings(x, full_partition, part_partition)
        ap1, ap2 = self.Task1CLEvaluator_batch(Zv, V_map1, V_map2, E_map1, E_map2, label, self.device)
        return ap1, ap2


class Task2SupervisedModel(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, device = 'cuda:0', 
                 aggregation_type='inner_dot', act: Callable = nn.PReLU()):
        super(Task2SupervisedModel, self).__init__()

        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act
        self.dropouts = torch.nn.Dropout(p=0.5)
        self.device = device
        self.convs = nn.ModuleList()

        self.agg_type = aggregation_type  # How to produce last target scalar
        
        self.agg_type = aggregation_type  # How to produce last target scalar
        if self.agg_type == 'concat' : 
            self.last_linear = torch.nn.Linear(int(2*node_dim), 1)
        elif self.agg_type in ['hadamard', 'abs_sub'] : 
            self.last_linear = torch.nn.Linear(int(node_dim), 1)
        elif self.agg_type == 'inner_dot' : 
            self.last_linear = None
        else : 
            raise TypeError('Given Wrong Aggregation Type')

        if num_layers == 1:
            self.convs.append(MeanPoolingConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        else:
            self.convs.append(MeanPoolingConv(self.in_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(MeanPoolingConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
            self.convs.append(MeanPoolingConv(self.node_dim, self.edge_dim, self.node_dim, cached=False, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int,
                ind1: list, ind2: list):
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
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

    def return_node_embeddings(self, X, full_partition, part_partition):

        partition_mapper = {i: [[], []] for i in range(len(part_partition))}

        i = 0
        for c1, c2 in zip(full_partition, part_partition):
            full_partition_indexer = dict()
            cur_part_nodes = c1['node_idx'].to('cpu').numpy()
            this_partition_nodes = c2['node_idx'].to('cpu').numpy()
            cur_idx = 0
            for v in cur_part_nodes:
                full_partition_indexer[v] = cur_idx
                cur_idx += 1

            for v in this_partition_nodes:
                partition_mapper[i][0].append(v)
                partition_mapper[i][1].append(full_partition_indexer[v])

            i += 1

        Zv = torch.zeros((X.shape[0], self.node_dim)).to(self.device)

        for c_num, e in enumerate(full_partition):

            part_hypE = e['hyperedge_index'].to(self.device)
            part_v_idx = e['node_idx']
            cur_x = X[part_v_idx].to(self.device)
            n_node = int(torch.max(part_hypE[0]) + 1)
            n_edge = int(torch.max(part_hypE[1]) + 1)

            for i in range(self.num_layers):
                if i == 0 :
                    x, e = self.convs[i](cur_x, part_hypE, n_node, n_edge)
                else : 
                    x, e = self.convs[i](x, part_hypE, n_node, n_edge)
                x = self.act(x)

            Zv[partition_mapper[c_num][0], :] = x[partition_mapper[c_num][1], :]
            
            del x, part_hypE, cur_x

        return Zv.to('cpu').detach().to(self.device)

    def partition_test_inference(self, x, full_partition, part_partition,
                                 ind1, ind2):

        Zv = self.return_node_embeddings(x, full_partition, part_partition)
        if self.agg_type == 'abs_sub':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear(torch.abs(Zv[ind1] - Zv[ind2]))).squeeze(-1)
        elif self.agg_type == 'hadamard':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear((Zv[ind1] * Zv[ind2]))).squeeze(-1)
        elif self.agg_type == 'concat':  ## Absolute difference
            pred = torch.sigmoid(self.last_linear(torch.hstack([Zv[ind1] , Zv[ind2]]))).squeeze(-1)
        else:
            pred = torch.sigmoid(torch.sum(Zv[ind1] * Zv[ind2], 1))  ## Inner Product
            
        return pred