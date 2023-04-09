#!/usr/bin/bash
singularity exec --nv /homedtic/xpavon/nerfstudio_root/nerfstudio_0.1.19.sif /homedtic/xpavon/.local/bin/ns-train phototourism --data $1 --output-dir $1 --viewer.quit-on-train-completion True
