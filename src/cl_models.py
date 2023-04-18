from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

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

class HyperEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, num_layers=2, act: Callable = nn.PReLU()):
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = act

        self.convs = nn.ModuleList()
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

    def forward(self, x: Tensor, hyperedge_index: Tensor, num_nodes: int, num_edges: int):
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, num_nodes, num_edges)
            x = self.act(x)
        return x, e  # act, act

class TriCon(nn.Module):
    def __init__(self, encoder: HyperEncoder, proj_dim: int):
        super(TriCon, self).__init__()
        self.encoder = encoder

        self.node_dim = self.encoder.node_dim
        self.edge_dim = self.encoder.edge_dim

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)

        self.disc = nn.Bilinear(self.node_dim, self.edge_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.disc.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None, only_node=False):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
        n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + num_nodes)
        if only_node:
            return n
        else:
            return n, e[:num_edges]

    def without_selfloop(self, x: Tensor, hyperedge_index: Tensor, node_mask: Optional[Tensor] = None,
                         num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1

        if node_mask is not None:
            node_idx = torch.where(~node_mask)[0]
            edge_idx = torch.arange(num_edges, num_edges + len(node_idx), device=x.device)
            self_loop = torch.stack([node_idx, edge_idx])
            self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)
            n, e = self.encoder(x, self_loop_hyperedge_index, num_nodes, num_edges + len(node_idx))
            return n, e[:num_edges]
        else:
            return self.encoder(x, hyperedge_index, num_nodes, num_edges)

    def f(self, x, tau):
        return torch.exp(x / tau)

    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))

    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))

    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def pairwise_cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.sum((z1 * z2), dim=1)

    def inter_cluster_loss(self, z1: Tensor, z2: Tensor, tau: float):
        return torch.mean(torch.log(torch.sum(self.f(self.cosine_similarity(z1, z2), tau), 1)))  # Cosine similarity
    
    def avg_inter_cluster_loss(self, z1: Tensor, z2: Tensor):
        avg_z1 = torch.mean(z1, 0)
        avg_z2 = torch.mean(z2, 0)
        #return torch.log(torch.sum(self.f(self.cosine_similarity(avg_z1, avg_z2), tau), 1))  # Cosine similarity
        return torch.mean(torch.abs(torch.mean(avg_z1, 0) - torch.mean(avg_z2, 0)))

    def semi_loss(self, z1: Tensor, z2: Tensor, tau: float, num_negs: float):
        pos_sim = self.f(F.cosine_similarity(z1, z2), tau)
        negs = []
        for _ in range(num_negs):
            negs.append(z2[torch.randperm(z2.size(0))])
        negs = torch.stack(negs, dim=-1)
        neg_sim = self.f(F.cosine_similarity(z1.unsqueeze(-1).tile(num_negs), negs), tau)
        return torch.mean(-torch.log(pos_sim / (pos_sim + neg_sim.sum(1))))

    def total_sampled_loss(self, z1: Tensor, z2: Tensor, tau: float, num_negs: int):

        L1 = self.semi_loss(z1, z2, tau, num_negs)
        L2 = self.semi_loss(z2, z1, tau, num_negs)
        return L1 + L2

    def total_cluster_negative_loss(self, z1, z2, z_1, z_2, tau):
        L1 = 0.5 * (self.inter_cluster_loss(z1, z_1, tau) + self.inter_cluster_loss(z1, z_2, tau))
        L2 = 0.5 * (self.inter_cluster_loss(z2, z_1, tau) + self.inter_cluster_loss(z2, z_2, tau))

        return 0.5 * (L1 + L2)
    
    def avg_cluster_negative_loss(self, z1, z2, z_1, z_2) : 
        L1 = 0.5 * (self.avg_inter_cluster_loss(z1, z_1) + self.avg_inter_cluster_loss(z1, z_2))
        L2 = 0.5 * (self.avg_inter_cluster_loss(z2, z_1) + self.avg_inter_cluster_loss(z2, z_2))
        
        return 0.5 * (L1 + L2)

    def disc_similarity(self, z1: Tensor, z2: Tensor):
        return torch.sigmoid(self.disc(z1, z2)).squeeze()