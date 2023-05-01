import argparse
import os
import subprocess
from pathlib import Path

import pprint

import calendar
import time

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

def generate_sbatch_script_for_command(job_session_id, command, command_n, c=8, num_gpus=1, m=8):
    sbatch_file_content = f"""#!/usr/bin/bash
#SBATCH -J {command_n}-{job_session_id}
#SBATCH -p high
#SBATCH -N 1
#SBATCH -c {c}
#SBATCH --chdir=/homedtic/xpavon/nerfstudio_root
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --mem-per-cpu={m}G
#SBATCH --output=./nerfstudio/bash_scripts/{job_session_id}/run_outputs/%x-%J.out
#SBATCH --error=./nerfstudio/bash_scripts/{job_session_id}/run_outputs/%x-%J.out

echo "command executed:"
echo "{command}"
echo -------------------
echo -------------------
# Execute the command for the current task
{command}
    """
    return sbatch_file_content

timestamp = calendar.timegm(time.localtime())
JOB_SESSION_NAME = "maskha"
JOB_SESSION_ID = JOB_SESSION_NAME + "-" + str(timestamp)
NERFSTUDIO_ROOT = Path('/homedtic/xpavon/nerfstudio_root')
JOB_SESSION_DIR = NERFSTUDIO_ROOT / "nerfstudio/bash_scripts" / JOB_SESSION_ID

# prefixes and templates 
singularity_exec_prefix = "singularity exec --nv /homedtic/xpavon/nerfstudio_root/nerfstudio_0.1.19.sif "
nerfstudio_exec_script_prefix = singularity_exec_prefix + "python3 /homedtic/xpavon/nerfstudio_root/nerfstudio/scripts/"
nerfstudio_train_script_prefix = nerfstudio_exec_script_prefix + "train.py "
hanerfacto_phototourism_train_script_prefix = nerfstudio_train_script_prefix + f"hanerfacto-phototourism --viewer.quit-on-train-completion True --experiment-name {JOB_SESSION_ID} "
hanerf_phototourism_train_script_prefix = nerfstudio_train_script_prefix + f"hanerfacto --viewer.quit-on-train-completion True --experiment-name {JOB_SESSION_ID} "

def main(mode):

    # hanerf_loss_mask_size_delta: float = 0.006
    # hanerf_loss_mask_digit_delta: float = 0.001
    # occlusions --use-synthetic-occlusions False
    # --pipeline.datamanager.train-num-images-to-sample-from
    # --pipeline.datamanager.num-times-to-repeat-images

    commands = [
        hanerf_phototourism_train_script_prefix + "--data data/mask/outputs/mask_32 --pipeline.model.enable-hanerf-loss False --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 1 occlusions --use-synthetic-occlusions False",
        hanerf_phototourism_train_script_prefix + "--data data/mask/outputs/mask_32 --pipeline.model.hanerf-loss-mask-size-delta 0.06 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 1",
        hanerf_phototourism_train_script_prefix + "--data data/mask/outputs/mask_32 --pipeline.model.hanerf-loss-mask-size-delta 0.6 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 1",
    ]

    assert len(commands) != 0, "no commands to run!"

    if mode != "print":
        JOB_SESSION_DIR.mkdir()
        RUN_OUTPUTS_DIR = JOB_SESSION_DIR / "run_outputs"
        RUN_OUTPUTS_DIR.mkdir()

    sbatch_filenames = []
    sbatch_filecontents = []
    for i, command in enumerate(commands):
        current_sbatch_filename = (JOB_SESSION_DIR / f"{i}-{JOB_SESSION_NAME}").with_suffix('.bash')
        sbatch_filenames.append(current_sbatch_filename)
        if i != len(commands) - 1:
            next_sbatch_filename = (JOB_SESSION_DIR / f"{i+1}-{JOB_SESSION_NAME}").with_suffix('.bash')
            command += f" ; sbatch {next_sbatch_filename}"
        
        filecontent = generate_sbatch_script_for_command(JOB_SESSION_ID, command, i)
        sbatch_filecontents.append(filecontent)


    if mode == "print":
        pprint.pprint(sbatch_filenames)
        pprint.pprint(sbatch_filecontents)
        return
    
    for fname, fcontent in zip(sbatch_filenames, sbatch_filecontents):
        with open(fname, 'w') as file:
            file.write(fcontent)

    if mode=="exec":
        subprocess.call(['sbatch', sbatch_filenames[0]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='whether to print or execute sbatch file', default='print')
    kwargs = vars(parser.parse_args())
    main(**kwargs)
