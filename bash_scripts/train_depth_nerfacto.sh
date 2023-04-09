#!/usr/bin/bash

#SBATCH -J ns-dn-mask16
#SBATCH -p high
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -c 8
#SBATCH --chdir=/homedtic/xpavon/nerfstudio_root
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G

#SBATCH -o ./run_outputs/%x.%J.out # STDOUT
#SBATCH -e ./run_outputs/%x.%J.err # STDERR

singularity exec --nv ./nerfstudio_0.1.19.sif /homedtic/xpavon/.local/bin/ns-train depth-nerfacto --data \
    ./data/mask/outputs/mask_16 --output-dir ./data/mask/outputs/mask_16
