#!/usr/bin/bash

#SBATCH -J pullns
#SBATCH -p high
#SBATCH -N 2
#SBATCH --chdir=/homedtic/xpavon/nerfstudio
### #SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G

#SBATCH -o ./run_outputs/%x.%J.out # STDOUT
#SBATCH -e ./run_outputs/%x.%J.err # STDERR

singularity pull docker://dromni/nerfstudio:0.1.19



