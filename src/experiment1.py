import argparse
import pickle

from task1_dataloader import *

from mlp_codes import *

from ssl_models import *
from ssl_trainers import *

from cl_models import *
from cl_pretrainer import *
from cl_trainers import *
from cl_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Task1 Experiment Code")
    parser.add_argument("-data", "--dataset", type=str, help="Name of Dataset")
    parser.add_argument("-model", "--model_type", type=str, help="Name of Method")
    parser.add_argument("-device", "--device", type=str, help="Select device for this code")
    parser.add_argument("-lr", "--learning_rate", type=float, default = 0.001, help="Learning Rate of a Model")
    parser.add_argument("-n_neg", "--n_neg", type=int, default = 1, help="Number of Negative Samples")
    parser.add_argument("-d_rate", "--drop_rate", type=float, default = 0.3, help="Drop Rate of Feature and Incidence Matrix")
    parser.add_argument("-ep", "--cl_epoch", type=int, default = 25, help="Checkpoint Epoch of CL Encoder")
    parser.add_argument("-seed", "--data_seed", type=int, default=0, help="Fixing Random Seed")

    args = parser.parse_args()
    data_name = args.dataset
    model_type = args.model_type

    assert data_name in ["dblp", "trivago", "ogbn_mag", "aminer", "mag"]
    assert model_type in ["mlp", "full_ssl", "full_cl", "part_ssl", "HCL", "HCL_PINS", "HCL_PIOS"]

    if (data_name in ["aminer", "mag"]) and (model_type in ["HCL_PIOS"]) :
        raise TypeError("aminer or mag dataset is not supported by HCL_PIOS due to memory issue.")

    if data_name == "dblp" : N_C = 4
    elif data_name == "trivago": N_C = 32
    elif data_name == "ogbn_mag": N_C = 128
    elif data_name == "aminer": N_C = 256
    else : N_C = 256

    device = args.device
    lr = args.learning_rate
    n_neg = args.n_neg
    d_rate = args.drop_rate
    ep = args.cl_epoch
    init_seed = args.data_seed

    X = torch.load("data/{0}/{0}_X_vec.pt".format(data_name))
    with open("data/{0}/{0}_split_E.pickle".format(data_name), "rb") as fp :
        E_dict = pickle.load(fp)
    E = E_dict[0]

    if model_type in ["part_ssl", "HCL", "HCL_PINS"] :
        PART = torch.load("data/{0}/{0}_split_partition_{1}.pt".format(data_name, N_C))
    elif model_type in ["HCL_PIOS"] :
        PART = torch.load("data/{0}/{0}_split_partition_{1}_PIOS.pt".format(data_name, N_C))

    if model_type == "mlp" :
        loader = Task1FullDataLoader(edge_dict=E_dict,
                                     init_seed=init_seed,
                                     device=device,
                                     split=0.1)
        loader.create_data(init_seed)
        torch.manual_seed(init_seed)
        model = Task1MLP(in_dim=X.shape[1],
                         hidden_dim=128,
                         num_layers=2,
                         device=device).to(device)
        if data_name in ["dblp", "trivago", "ogbn_mag"] :
            model, t1, t2 = Task1MLPTrainer(model=model, X=X.to(device),
                            seed=init_seed, dataloader=loader, lr=lr)
        else :
            model, t1, t2 = ScalableTask1MLPTrainer(model, X,
                                                    init_seed, loader,
                                                    lr=0.001, epoch=100, w_decay=1e-6, device=device)

    elif model_type == "full_ssl" :
        loader = Task1FullDataLoader(edge_dict=E_dict,
                                     init_seed=init_seed,
                                     device=device,
                                     split=0.1)
        loader.create_data(init_seed)
        torch.manual_seed(init_seed)

        model = Task1SupervisedModel(in_dim=X.shape[1],
                                     edge_dim=128,
                                     node_dim=128,
                                     device=device,
                                     pooling_method='sum',
                                     aggregation_type='inner_dot',
                                     num_layers=2).to(device)

        model, t1, t2 = Task1FullSupervisedTrainer(model=model, X=X.to(device), E=E.to(device),
                                                           n_node=int(max(E[0]) + 1), n_edge=int(max(E[1]) + 1),
                                                           seed=init_seed, dataloader=loader, lr=lr)

    elif model_type == "full_cl" :
        loader = Task1FullDataLoader(edge_dict=E_dict,
                                     init_seed=init_seed,
                                     device=device,
                                     split=0.1)
        loader.create_data(init_seed)
        torch.manual_seed(init_seed)
        encoder = HyperEncoder(in_dim=X.shape[1],
                               edge_dim=128,
                               node_dim=128,
                               num_layers=2).to(device)
        contrastive_model = TriCon(encoder=encoder,
                                   proj_dim=128).to(device)
        optimizer = torch.optim.Adam(contrastive_model.parameters(),
                                     lr=lr,
                                     weight_decay=1e-6)
        CL_trainer = CLFullTrainer(X = X.to(device),
                      EDGEs = E.to(device),
                      model = contrastive_model,
                      is_split = True,
                      data_name = data_name,
                      optimizer = optimizer)
        encoder_weight = CL_trainer.fit(epoch=300, saving_interval=ep, num_negs=n_neg,
                        drop_incidence_rate=d_rate, drop_feature_rate=d_rate,
                        tau_n=0.5, save_model=True)
        encoder.load_state_dict(encoder_weight)
        model = Task1CLClassifier(in_dim=128).to(device)
        model, t1, t2 = Task1CLTrainer(model=model,
                                       encoder=encoder,
                                       X=X.to(device), E=E.to(device), emb_dim=128, seed=init_seed,
                                       dataloader=loader, device=device,
                                       pooling='sum', infer_type=1, lr=0.001, epoch=100, w_decay=1e-6)

    elif model_type == "part_ssl" :
        loader = Task1PartitionDataLoader(edge_dict=E_dict,
                                           full_partition=PART,
                                           init_seed=init_seed,
                                           device=device,
                                           split=0.1)
        loader.create_data(init_seed)
        torch.manual_seed(init_seed)
        model = Task1SupervisedModel(in_dim=X.shape[1],
                                     edge_dim=128,
                                     node_dim=128,
                                     device=device,
                                     pooling_method='sum',
                                     aggregation_type='inner_dot',
                                     num_layers=2).to(device)

        model, t1, t2 = Task1PartSupervisedTrainer(model=model,
                                                   X=X, entire_edges=E,
                                                   full_partition=PART, part_partition=PART,
                                                   device=device, seed=init_seed,
                                                   dataloader=loader,
                                                   infer_type=0, lr=lr, epoch=100, w_decay=1e-6)

    elif model_type in ["HCL", "HCL_PINS", "HCL_PIOS"] :
        loader = Task1FullDataLoader(edge_dict=E_dict,
                                     init_seed=init_seed,
                                     device=device,
                                     split=0.1)
        loader.create_data(init_seed)
        torch.manual_seed(init_seed)
        partition_dataloader = PartitionSampler(partition_batch=PART, init_seed=init_seed, device=device)
        encoder = HyperEncoder(in_dim=X.shape[1],
                               edge_dim=128,
                               node_dim=128,
                               num_layers=2).to(device)
        contrastive_model = TriCon(encoder=encoder,
                                   proj_dim=128).to(device)
        optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=lr, weight_decay=1e-6)

        if model_type in ["HCL", "HCL_PIOS"] :
            partition_contrastive_trainer = CLPartTrainer(X=X, dataloader=partition_dataloader,
                                                      model=contrastive_model, optimizer=optimizer,
                                                      device=device)
        else :
            partition_contrastive_trainer = CLInterPartitionTrainer(X=X,
                                                                    dataloader=partition_dataloader,
                                                                    model=contrastive_model,
                                                                    optimizer=optimizer,
                                                                    device=device)

        encoder_weight = partition_contrastive_trainer.fit(epoch=50,
                                                    save_model=True, num_negs=n_neg, saving_interval=ep,
                                                    drop_incidence_rate=d_rate, drop_feature_rate=d_rate,
                                                    tau_n=0.5)
        encoder.load_state_dict(encoder_weight)

        model = Task1CLClassifier(in_dim=128).to(device)
        model, t1, t2 = trainer = Task1CLTrainer(model=model, encoder=encoder, X=X, E=E,
                                                 emb_dim=128, seed=init_seed,
                                                 dataloader=loader, device=device,
                                                 pooling='sum', infer_type=0,
                                                 full_partition=PART,
                                                 part_partition=PART,
                                                 lr=0.001, epoch=100, w_decay=1e-6)

    print("Data: {0} / Method: {1} / AP: {2} / AUROC: {3}".format(data_name, model_type, t1, t2))
