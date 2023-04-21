import argparse
import os
import subprocess
from pathlib import Path

"""
Helper script to handle sbatch script calling. 
All input logic is defineed and processed here:
    - This prevents having to define multiple sbatch scripts per job, which is time consuming.
    - This prevents having to modify a sbatch script multiple times each time we want to specify a different set of arguments, which is error-prone.
    - Python is way more accessible. 

SBATCH_array_prototype is called only once 
Configurable arguments list:
    job name
    gpu?
    commands/jobs to call
"""

def main(mode):
    nerfstudio_root = Path('/homedtic/xpavon/nerfstudio_root')

    # prefixes and templates (trailing whitespaces)
    singularity_exec_prefix = "singularity exec --nv /homedtic/xpavon/nerfstudio_root/nerfstudio_0.1.19.sif "
    nerfstudio_root_prefix = singularity_exec_prefix + "/homedtic/xpavon/nerfstudio_root/ "
    nerfstudio_train_script_prefix = nerfstudio_root_prefix + "train.py "
    hanerfacto_phototourism_train_script_prefix = nerfstudio_train_script_prefix + "hanerfacto-phototourism --viewer.start-train False"

    commands = [
        hanerfacto_phototourism_train_script_prefix + "--data blablabla",
        hanerfacto_phototourism_train_script_prefix + "--data lelele",
    ]
    commands = [f'"{t}"' for t in commands]

    job_name = "test_script"
    job_name = "_".join(job_name.split(" "))

    num_gpus=0

    sbatch_file_content = f"""#!/usr/bin/bash
#SBATCH -J {job_name}
#SBATCH -p high
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -c 8
#SBATCH --chdir=/homedtic/xpavon/nerfstudio_root/nerfstudio
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --mem-per-cpu=8G
#SBATCH --output=./run_outputs/${{SLURM_JOB_ID}}_${{SLURM_JOB_NAME}}/%x-%A-%a.out
#SBATCH --error=./run_outputs/${{SLURM_JOB_ID}}_${{SLURM_JOB_NAME}}/%x-%A-%a.out

#SBATCH --array=0-{len(commands)-1}:1

mkdir -p /homedtic/xpavon/nerfstudio_root/nerfstudio/run_outputs/${{SLURM_JOB_ID}}_${{SLURM_JOB_NAME}}

commands_array={f'({(" ").join(commands)})'}

command=${{commands_array[${{SLURM_ARRAY_TASK_ID}}]}}

# Execute the command for the current task
eval $command
    """
    sbatch_filename= nerfstudio_root / 'nerfstudio/bash_scripts' / 'SBATCH_array_prototype.sh'

    if mode=="exec":
        with open(sbatch_filename, 'w') as file:
            file.write(sbatch_file_content)
        subprocess.call(['sbatch', sbatch_filename])
    elif mode=="write":
        with open(sbatch_filename, 'w') as file:
            file.write(sbatch_file_content)
    else:
        print(sbatch_file_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='whether to print or execute sbatch file', default='print')
    kwargs = vars(parser.parse_args())
    main(**kwargs)
