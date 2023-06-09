# Source Code and Datasets for "Datasets, Tasks, and Training Methods for Benchmarking Large-scale Hypergraph Learning." 

## Dataset Description

We provide hypergraph datasets at **https://www.dropbox.com/sh/1908u1h5w6zxzjs/AABP04dM4VU_0fadrpe_5pb-a?dl=0**. 

We provide 
- Each dataset's feature, node label, original hyperedge information, split hyperedge information (for task 1)
- Each dataset's partitioned hypergraph (# of Partition $|P|$ / DBLP:4, Trivago:32, OGBN_MAG:128, AMINER and MAG: 256)
- For DBLP, Trivago, OGBN_MAG, we also provide partitioned hypergraph **P-IOS** partition.  

Refer to README_DATA.txt file for more details regarding datasets.

## Code Description 

### Overview
In this repository, we provide source codes for
- Obtaining performance on proposed task 1 (hyperedge disambiguation)
- Obtaining performance on proposed task 2 (local clustering)

### Dataset
For dataset one aims to run the code, files of corresponding dataset in **https://www.dropbox.com/sh/qhz6rqol5mue4wc/AAByciJuLNba8uGv8MdMpbf-a?dl=0** should be located in scr/data folder. 
For example
```
src
  |_ data
      |_ aminer
            |_ aminer_X_vec.pt
            |_ aminer_y.pt
            |_ aminer_E.pt
            |_ ...
```

### Hyperparameters

We provide hyperparameter combination for reproducibility of experimental results.  
Refer to best_hyperparameter directory, where we saved each dataset-model combination's best hyperparameter as .json files.

### How to implement codes

One can simply run code with codes in src folder as follows;  
For task 1 (hyperedge disambiguation)
```
python experiment1 -data "data" -model "learning-method" -device "GPU-device" -lr 0.001 -seed 0 
```  
For task 2 (local clustering)
```
python experiment2 -data "data" -model "learning-method" -device "GPU-device" -lr 0.001 -seed 0 
```
For contrastive learning, additional arguments can be given as   
```
python experiment1 -data "data" -model "learning-method" -device "GPU-device" -lr 0.001 -seed 0 -n_neg 1 -d_rate 0.3 -ep 25
```
Arguments are
- -data : Dataset one wants to perform experiments. (Possible Options: "dblp", "trivago", "ogbn_mag", "aminer", "mag")
- -model : Training Method one wants to perform experiments. (Possible Options: "mlp", "full_ssl", "full_cl", "part_ssl", "HCL", "HCL_PINS", "HCL_PIOS")
- -device : GPU machine (e.g. "cuda:0")
- -lr : Learning rate for method. (float / e.g. 0.001)
- -n_neg : Applicable to CL methods. Number of negative samples used for CL training. (int / e.g. 2)
- -d_rate : Applicable to CL methods. Feature & Incidence matrix dropping probability. (float / e.g. 0.3)
- -ep : Applicable to CL methods. Contrastive encoder's checkpoint for contrastive training. (int / e.g. 25)
- -seed : Dataset & model initialization seed. (int / e.g. 0)
