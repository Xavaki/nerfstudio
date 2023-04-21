#!/usr/bin/bash
#SBATCH -J test_script
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 8
#SBATCH --chdir=/homedtic/xpavon/nerfstudio_root/nerfstudio
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./run_outputs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}/%x-%A-%a.out
#SBATCH --error=./run_outputs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}/%x-%A-%a.out

#SBATCH --array=0-1:1

mkdir -p /homedtic/xpavon/nerfstudio_root/nerfstudio/run_outputs/${SLURM_JOB_ID}_${SLURM_JOB_NAME}

commands_array=("singularity exec --nv /homedtic/xpavon/nerfstudio_root/nerfstudio_0.1.19.sif /homedtic/xpavon/nerfstudio_root/ train.py hanerfacto-phototourism --viewer.start-train False--data blablabla" "singularity exec --nv /homedtic/xpavon/nerfstudio_root/nerfstudio_0.1.19.sif /homedtic/xpavon/nerfstudio_root/ train.py hanerfacto-phototourism --viewer.start-train False--data lelele")

command=${commands_array[${SLURM_ARRAY_TASK_ID}]}

# Execute the command for the current task
eval $command
    