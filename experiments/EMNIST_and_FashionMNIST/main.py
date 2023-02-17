import argparse
import os
import random
import torch
from trainer import GIN, VAE

parser = argparse.ArgumentParser(description='Experiments on EMNIST with GIN (training script)')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='Number of training epochs (default 100)')
parser.add_argument('--epochs_per_line', type=int, default=1,
                    help='Print a new line after this many epochs (default 1)')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='Learn rate (default 3e-4)')
parser.add_argument('--lr_schedule', nargs='+', type=int, default=[50],
                    help='Learn rate schedule (decrease lr by factor of 10 at these epochs, default [50]). \
                            Usage example: --lr_schedule 20 40')
parser.add_argument('--batch_size', type=int, default=240,
                    help='Batch size (default 240)')
parser.add_argument('--save_frequency', type=int, default=10,
                    help='Save a new checkpoint and make plots after this many epochs (default 10)')
parser.add_argument('--data_root_dir', type=str, default='./',
                    help='Directory in which \'EMNIST\' directory storing data is located (defaults to current directory). If the data is not found here you will be prompted to download it')
parser.add_argument('--incompressible_flow', type=int, default=1,
                    help='Use an incompressible flow (GIN) (1, default) or compressible flow (GLOW) (0)')
parser.add_argument('--empirical_vars', type=int, default=1,
                    help='Estimate empirical variables (means and stds) for each batch (1, default) or learn them along \
                            with model weights (0)')
parser.add_argument('--dim_z', type=int, default=32, help='dimension of representation for VAE-based methods')
parser.add_argument('--nf', type=int, default=64, help='number of filter for VAE-based methods')
parser.add_argument('--intermediate_nodes', type=int, default=256, help='number of intermediate nodes for VAE-based methods')
parser.add_argument('--beta', type=float, default=1.0, help='the coefficient of kl divergence terms')
parser.add_argument('--dataset', required=True, dest='dataset', choices=('EMNIST', 'FashionMNIST'), help="Method to train identifiable models")
parser.add_argument('--method', required=True, dest='method', choices=('GIN', 'iVAE', 'IDVAE', 'CI-iVAE'), help="Method to train identifiable models")
parser.add_argument('--seed', type=int, default=0, help='seed number for reproducibility')
parser.add_argument('--kl_annealing', action='store_true', help='applying KL annealing')
parser.add_argument('--aggressive_post', action='store_true', help='applying aggresive posterior training')
parser.add_argument('--knn_evaluation', action='store_true', help='calculating knn classifier error rate')
args = parser.parse_args()

print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

assert args.incompressible_flow in [0,1], 'Argument should be 0 or 1'
assert args.empirical_vars in [0,1], 'Argument should be 0 or 1'

if args.method == "GIN":
    model = GIN(dataset=args.dataset, 
                n_epochs=args.n_epochs, 
                epochs_per_line=args.epochs_per_line, 
                lr=args.lr, 
                lr_schedule=args.lr_schedule, 
                batch_size=args.batch_size,
                data_root_dir=args.data_root_dir, 
                incompressible_flow=args.incompressible_flow, 
                empirical_vars=args.empirical_vars,
                seed=args.seed)
    if args.knn_evaluation:
        model.calculate_knn_accuracy()
    else:
        model.train_model()
elif args.method in ['iVAE', 'IDVAE', 'CI-iVAE']:
    model = VAE(dataset=args.dataset, 
                n_epochs=args.n_epochs, 
                epochs_per_line=args.epochs_per_line, 
                lr=args.lr, 
                lr_schedule=args.lr_schedule, 
                batch_size=args.batch_size,
                data_root_dir=args.data_root_dir, 
                dim_z=args.dim_z,
                nf=args.nf,
                intermediate_nodes=args.intermediate_nodes,
                beta=args.beta,
                method=args.method,
                seed=args.seed,
                kl_annealing=args.kl_annealing,
                aggressive_post=args.aggressive_post)
    if args.knn_evaluation:
        model.calculate_knn_accuracy(test=False)
    else:
        model.train_model()