#!/usr/bin/bash
#SBATCH -J 2-maskha-1682947572
#SBATCH -p high
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --chdir=/homedtic/xpavon/nerfstudio_root
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./nerfstudio/bash_scripts/maskha-1682947572/run_outputs/%x-%J.out
#SBATCH --error=./nerfstudio/bash_scripts/maskha-1682947572/run_outputs/%x-%J.out

echo "command executed:"
echo "singularity exec --nv /homedtic/xpavon/nerfstudio_root/nerfstudio_0.1.19.sif python3 /homedtic/xpavon/nerfstudio_root/nerfstudio/scripts/train.py hanerfacto --viewer.quit-on-train-completion True --experiment-name maskha-1682947572 --data data/mask/outputs/mask_32 --pipeline.model.hanerf-loss-mask-size-delta 0.6 --pipeline.datamanager.train-num-images-to-sample-from 1"
echo -------------------
echo -------------------
# Execute the command for the current task
singularity exec --nv /homedtic/xpavon/nerfstudio_root/nerfstudio_0.1.19.sif python3 /homedtic/xpavon/nerfstudio_root/nerfstudio/scripts/train.py hanerfacto --viewer.quit-on-train-completion True --experiment-name maskha-1682947572 --data data/mask/outputs/mask_32 --pipeline.model.hanerf-loss-mask-size-delta 0.6 --pipeline.datamanager.train-num-images-to-sample-from 1
    