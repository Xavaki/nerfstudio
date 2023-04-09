#!/usr/bin/bash

#SBATCH -J ns-phott
#SBATCH -p high
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -c 8
#SBATCH --chdir=/homedtic/xpavon/nerfstudio_root/nerfstudio
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G

#SBATCH -o ./run_outputs/%x.%J.out # STDOUT
#SBATCH -e ./run_outputs/%x.%J.err # STDERR

base_data_root="/homedtic/xpavon/nerfstudio_root/data/"
scenes=("phototourism/brandenburg-gate")
n_scenes=${#scenes[@]}
#SBATCH --array=0-$(($n_scenes-1)):1


scene=${scenes[${SLURM_ARRAY_TASK_ID}]}
scene_path="$base_data_root${scene}"
/usr/bin/bash /homedtic/xpavon/nerfstudio_root/nerfstudio/bash_scripts/phototourism.sh ${scene_path}