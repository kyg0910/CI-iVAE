This folder provides code to reproduce results in our paper. The code for simulation is written in python with dependency in Keras, mainly extended from the implementation code for pi-VAE [1]. The code for simulation is written in python with dependency in pytorch. The code for image datasets iss mainly extended from the implementation code for GIN [2].

## 1. Commands to reproduce experimental results
#### 1.1. Simulation study
To run code for simulation study, please execute the following commands at ./simulation.
```bash
# CI-iVAE
python conti_simulation.py --latent_type {sine, quadratic, circle} --seed_num_dataset_list {0, 1, ..., 19}
# iVAE
python conti_simulation.py --latent_type {sine, quadratic, circle} --seed_num_dataset_list {0, 1, ..., 19} --fix_alpha 0.0
```
The seed numbers to reproduce results in the manucript are 0, 1, ..., 19. For example, you can run ``python conti_simulation.py --latent_type sine --seed_num_dataset_list 0`` to get results for CI-iVAE on sine latent structure with seed number 0. Results are saved at ./simulation/results.

#### 1.2. EMNIST and Fashion-MNIST
To run code for EMNIST and Fashion-MNIST datasets, please execute the following commands at ./EMNIST_and_FashionMNIST.
```bash
# CI-iVAE
python main.py --dataset {EMNIST, FashionMNIST} --seed {0, 1, ..., 19} --method CI-iVAE --nf 32 --intermediate_nodes 128 --dim_z 64 --beta 0.001
# GIN
python main.py --dataset {EMNIST, FashionMNIST} --seed {0, 1, ..., 19} --method GIN
# iVAE
python main.py --dataset {EMNIST, FashionMNIST} --seed {0, 1, ..., 19} --method iVAE --nf 32 --intermediate_nodes 128 --dim_z 64 --beta 0.001
# IDVAE
python main.py --dataset {EMNIST, FashionMNIST} --seed {0, 1, ..., 19} --method IDVAE --nf 32 --intermediate_nodes 128 --dim_z 64 --beta 0.001
```
For example, you can run ``python main.py --dataset EMNIST --seed 0 --method CI-iVAE --nf 32 --intermediate_nodes 128 --dim_z 64 --beta 0.001`` to get results for CI-iVAE on the EMNIST dataset with seed number 0. Results are saved at ./EMNIST_and_FashionMNIST/results. To apply KL annealing and aggressive_post, respectively, please add ``--kl_annealing`` and ``--aggressive_post``. The seed numbers to reproduce results in the manucript are 0, 1, ..., 19. The seed numbers to reproduce results in the manucript are 0, 1, ..., 19.

#### 1.3. ABCD
To run code for the ABCD dataset, please first download ABCD dataset at https://abcdstudy.org, and go to ./ABCD and run ``python main.py --seed_num_opt 0 --beta_kl_post_prior 0.01 --beta_kl_encoded_prior 0.01 --gen_nodes 4096 --dim_z 128``. Please add ``--fix_alpha 0.0`` for iVAE. The seed number to reproduce results in the manucript is 0.

## 2. Package dependencies
### 2.1. Simulation Study
- argparse=1.1
- keras=2.3.1
- matplotlib=3.1.2
- numpy=1.16.0
- pandas=1.1.5
- python=3.6.13
- scipy=1.5.4
- sklearn=0.23.2
- tensorflow=1.13.1

The code worked well with RTX 2080 Ti and Cuda compilation tools, release 10.2, V10.2.89.

### 2.2. Experiments on EMNIST and Fashion-MNIST
- argparse=1.1
- matplotlib=3.5.2
- numpy=1.21.4
- pandas=1.4.1
- python=3.6.10
- scipy=1.10.0
- seaborn=0.11.1
- sklearn=1.0.2
- torch=1.10.1+cu111
- torchvision=0.11.2+cu111

Please install FrEIA package by following guidelines at http://github.com/vislearn/FrEIA. The code worked well with RTX 3090 and Cuda compilation tools, release 11.6, V11.6.124.

### 2.3. Experiments on ABCD
- argparse=1.1
- matplotlib=3.5.2
- numpy=1.21.4
- pandas=1.4.1
- sklearn=1.0.2
- statsmodels=0.14.0.dev0+350.g408eae829
- tensorboardX=2.4
- torch=1.10.1+cu111
- tqdm=4.40.0
- yaml=5.3.1

The code worked well with RTX 3090 and Cuda compilation tools, release 11.6, V11.6.124.

## 3. References
[1] Zhou, Ding, and Xue-Xin Wei. "Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE." Advances in Neural Information Processing Systems 33 (2020): 7234-7247. Github: https://github.com/zhd96/pi-vae

[2] Sorrenson, Peter, Carsten Rother, and Ullrich Köthe. "Disentanglement by Nonlinear ICA with General Incompressible-flow Networks (GIN)." International Conference on Learning Representations. Github: https://github.com/vislearn/GIN
