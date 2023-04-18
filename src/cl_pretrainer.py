import torch
from tqdm import tqdm

from cl_utils import *
from time import time

class CLFullTrainer(object):

    def __init__(self, X, EDGEs, model, is_split, data_name, optimizer):  ## Data is given as torch scatter data class type
        self.X = X
        self.hyperedges = EDGEs
        self.model = model
        self.optimizer = optimizer
        self.lr = self.optimizer.param_groups[0]['lr']
        self.data_name = data_name
        if is_split : 
            self.split = 'split'
        else : 
            self.split = 'orig'

    def do_forward_pass(self, num_negs,
                        drop_incidence_rate=0.2, drop_feature_rate=0.2,
                        tau_n=0.5):

        torch.autograd.set_detect_anomaly(True)
        model, optimizer = self.model, self.optimizer

        num_nodes, num_edges = self.X.shape[0], int(torch.max(self.hyperedges[1, :]) + 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Augmentation
        hyperedge_index1 = drop_incidence(self.hyperedges, drop_incidence_rate)
        hyperedge_index2 = drop_incidence(self.hyperedges, drop_incidence_rate)
        X1 = drop_features(self.X, drop_feature_rate)
        X2 = drop_features(self.X, drop_feature_rate)

        # Encoder
        n1, _ = model(X1, hyperedge_index1, num_nodes, num_edges, only_node=False)
        n2, _ = model(X2, hyperedge_index2, num_nodes, num_edges, only_node=False)

        n1, n2 = model.node_projection(n1), model.node_projection(n2)

        loss_n = model.total_sampled_loss(z1=n1, z2=n2,
                                          tau=tau_n, num_negs=num_negs)

        loss = loss_n
        loss.backward()
        optimizer.step()

        return loss.item()

    def fit(self, epoch = 300, saving_interval = 100, num_negs = 1,
            drop_incidence_rate=0.3, drop_feature_rate=0.3,
            tau_n=0.5,  save_model = True) :

        self.epochwise_loss = []
        self.model.train()

        for ep in tqdm(range(epoch)):

            cur_ep_loss = self.do_forward_pass(num_negs,
                                               drop_incidence_rate, drop_feature_rate,
                                               tau_n)
            self.epochwise_loss.append(cur_ep_loss)

            if (save_model & (int(ep + 1) == saving_interval)):
                pretrained_encoder = self.model.encoder.state_dict()
                print('Pre-determined CL Training Epoch has been reached. Finishing CL Training...')
                break
        return pretrained_encoder

class CLPartTrainer(object):

    def __init__(self, X, dataloader, model, optimizer, device) :

        self.dataloader = dataloader  # Partitioned Hypergraph DataLoader
        self.model = model  # Contrastive Learning Model
        self.optimizer = optimizer
        self.X = X
        self.device = device

    def do_forward_pass(self, X, hyperedges, num_negs,
                        drop_incidence_rate=0.2, drop_feature_rate=0.2,
                        tau_n=0.5):

        torch.autograd.set_detect_anomaly(True)
        model, optimizer = self.model, self.optimizer

        num_nodes, num_edges = int(torch.max(hyperedges[0]) + 1), int(torch.max(hyperedges[1]) + 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Augmentation
        hyperedge_index1 = drop_incidence(hyperedges, drop_incidence_rate)
        hyperedge_index2 = drop_incidence(hyperedges, drop_incidence_rate)
        X1 = drop_features(X, drop_feature_rate)
        X2 = drop_features(X, drop_feature_rate)

        # Encoder
        n1, _ = model(X1, hyperedge_index1, num_nodes, num_edges, only_node=False)
        n2, _ = model(X2, hyperedge_index2, num_nodes, num_edges, only_node=False)

        n1, n2 = model.node_projection(n1), model.node_projection(n2)

        loss_n = model.total_sampled_loss(z1=n1, z2=n2,
                                              tau=tau_n, num_negs=num_negs)

        loss = loss_n
        loss.backward()
        optimizer.step()

        return loss.item()

    def fit(self, epoch, num_negs,
            saving_interval=20, drop_incidence_rate=0.2, drop_feature_rate=0.2,
            tau_n=0.5, save_model=False) :

        self.epochwise_loss = []
        self.model.train()

        for ep in tqdm(range(epoch)):

            batch_set = self.dataloader.batch_cluster_loader(init_seed=0)  ## Loading Cluster one by one
            ep_loss_mean = 0

            for batch in batch_set:  ## Batch is a list regarding train/valid/test node indicator together with overall hyperedge tensor data

                cur_hyperedges = batch['hyperedge_index'].to(self.device)
                cur_X = self.X[batch['node_idx'], :].to(self.device)

                cur_ep_loss = self.do_forward_pass(cur_X, cur_hyperedges, num_negs,
                                                   drop_incidence_rate, drop_feature_rate, tau_n)
                ep_loss_mean += cur_ep_loss

            cur_ep_loss /= len(batch_set)
            self.epochwise_loss.append(cur_ep_loss)

            if (save_model & (int(ep + 1) == saving_interval)):
                pretrained_encoder = self.model.encoder.state_dict()
                print('Pre-determined CL Training Epoch has been reached. Finishing CL Training...')
                break

        return pretrained_encoder

class CLInterPartitionTrainer(object):

    def __init__(self, X, dataloader, model, optimizer, device):  ## Data is given as torch scatter data class type

        self.dataloader = dataloader  # Partitioned Hypergraph DataLoader
        self.model = model  # Contrastive Learning Model
        self.optimizer = optimizer
        self.X = X
        self.device = device

    def do_forward_pass(self, X, hyperedges, num_negs, idx,
                        z_1=None, z_2=None, drop_incidence_rate=0.2, drop_feature_rate=0.2,
                        tau_n=0.5, w_c  = 0.1) :

        torch.autograd.set_detect_anomaly(True)
        model, optimizer = self.model, self.optimizer

        num_nodes, num_edges = int(torch.max(hyperedges[0]) + 1), int(torch.max(hyperedges[1]) + 1)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Augmentation
        hyperedge_index1 = drop_incidence(hyperedges, drop_incidence_rate)
        hyperedge_index2 = drop_incidence(hyperedges, drop_incidence_rate)
        X1 = drop_features(X, drop_feature_rate)
        X2 = drop_features(X, drop_feature_rate)

        node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
        node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
        node_mask = node_mask1 & node_mask2
        edge_mask = edge_mask1 & edge_mask2

        # Encoder
        n1, _ = model(X1, hyperedge_index1, num_nodes, num_edges, only_node=False)
        n2, _ = model(X2, hyperedge_index2, num_nodes, num_edges, only_node=False)

        n1, n2 = model.node_projection(n1), model.node_projection(n2)

        loss_n, loss_g, loss_m = 0, 0, 0
        loss_prev_clus = 0
        
        if idx > 0:

            loss_prev_clus = model.total_cluster_negative_loss(z1=n1, z2=n2,
                                                               z_1=z_1, z_2=z_2,
                                                               tau=tau_n)

        loss_n = model.total_sampled_loss(z1=n1, z2=n2,
                                                  tau=tau_n, num_negs=num_negs)

        loss = loss_n + w_c * loss_prev_clus
        loss.backward()
        optimizer.step()
        neg_sams = list(np.random.choice(np.arange(n1.shape[0]), 1))

        with torch.no_grad():

            n1, e1 = model(X1, hyperedge_index1, num_nodes, num_edges, only_node=False)
            n2, e2 = model(X2, hyperedge_index2, num_nodes, num_edges, only_node=False)

            z_1, z_2 = model.node_projection(n1)[neg_sams, :], model.node_projection(n2)[neg_sams, :]

        return loss.item(), z_1, z_2

    def fit(self, epoch, num_negs,
            saving_interval=20, drop_incidence_rate=0.2, drop_feature_rate=0.2,
            tau_n=0.5, w_c=0.1, save_model=False) :

        self.epochwise_loss = []
        self.model.train()

        for ep in tqdm(range(epoch)):

            batch_set = self.dataloader.batch_cluster_loader(init_seed=0)  ## Loading Cluster one by one
            ep_loss_mean = 0
            idx = 0

            for batch in batch_set:  ## Batch is a list regarding train/valid/test node indicator together with overall hyperedge tensor data

                cur_hyperedges = batch['hyperedge_index'].to(self.device)
                cur_X = self.X[batch['node_idx'], :].to(self.device)

                if idx > 0:
                    cur_ep_loss, z_1, z_2 = self.do_forward_pass(cur_X, cur_hyperedges, num_negs,
                                                                 idx, z_1, z_2,
                                                                 drop_incidence_rate, drop_feature_rate,
                                                                 tau_n, w_c)

                else:
                    cur_ep_loss, z_1, z_2 = self.do_forward_pass(cur_X, cur_hyperedges, num_negs,
                                                                 idx, None, None,
                                                                 drop_incidence_rate, drop_feature_rate,
                                                                 tau_n, w_c)
                ep_loss_mean += cur_ep_loss
                idx += 1

            cur_ep_loss /= len(batch_set)
            self.epochwise_loss.append(cur_ep_loss)

            if (save_model & (int(ep + 1) == saving_interval)):
                pretrained_encoder = self.model.encoder.state_dict()
                print('Pre-determined CL Training Epoch has been reached. Finishing CL Training...')
                break
        return pretrained_encoder