#!/usr/bin/bash

#SBATCH -J ns-render
#SBATCH -p high
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -c 8
#SBATCH --chdir=/homedtic/xpavon/nerfstudio_root/nerfstudio
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G

#SBATCH -o ./run_outputs/%x.%J.out # STDOUT
#SBATCH -e ./run_outputs/%x.%J.err # STDERR

# ns-render --load-config outputs/brandenburg-gate/phototourism/2023-04-02_085503/config.yml --traj filename --camera-path-filename data/phototourism/brandenburg-gate/camera_paths/2023-04-02_085503.json --output-path renders/brandenburg-gate/2023-04-02_085503.mp4

base_data_root="/homedtic/xpavon/nerfstudio_root/data/"
camera_path_filename="$base_data_root/phototourism/brandenburg-gate/camera_paths/2023-04-02_130257.json"
inputs=("phototourism/brandenburg-gate/brandenburg-gate/phototourism/2023-04-02_121827/config.yml")
outputs=("phototourism/brandenburg-gate/brandenburg-gate/phototourism/2023-04-02_121827/nerfacto_transient.mp4")

n_scenes=${#inputs[@]}
#SBATCH --array=0-$(($n_scenes-1)):1


input=${inputs[${SLURM_ARRAY_TASK_ID}]}
input_path="$base_data_root$input"
output=${outputs[${SLURM_ARRAY_TASK_ID}]}
output_path="$base_data_root$output"
/usr/bin/bash /homedtic/xpavon/nerfstudio_root/nerfstudio/bash_scripts/render.sh ${input_path} ${camera_path_filename} ${output_path}
