#!/usr/bin/bash

#SBATCH -J ns-process
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
inputs_dirs=("mask/inputs/mask_32" "mask/inputs/mask_64")
outputs_dirs=("mask/outputs/mask_32" "mask/outputs/mask_64")

n_scenes=${#inputs_dirs[@]}
#SBATCH --array=0-$(($n_scenes-1)):1

input=${inputs_dirs[${SLURM_ARRAY_TASK_ID}]}
input_path="$base_data_root${input}"
output=${outputs_dirs[${SLURM_ARRAY_TASK_ID}]}
output_path="$base_data_root${output}"
/usr/bin/bash /homedtic/xpavon/nerfstudio_root/nerfstudio/bash_scripts/process_data.sh ${input_path} ${output_path}