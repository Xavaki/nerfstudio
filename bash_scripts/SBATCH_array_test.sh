#!/bin/bash
#SBATCH --chdir=/homedtic/xpavon/nerfstudio_root/nerfstudio
#SBATCH --output=./run_outputs/%x-%A-%a.out

#SBATCH --array=0-1:1
commands_array=("echo u" "echo v")
command=${commands_array[${SLURM_ARRAY_TASK_ID}]}

# Execute the command for the current task
echo "Executing task $SLURM_ARRAY_TASK_ID with command: $command"
eval $command