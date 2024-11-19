#!/bin/bash

# sbatch params
c=10
time=12:00:00
mem=10GB
gres=gpu:rtx6000:1

# Environment setup
source /h/cdoumont/miniconda3/etc/profile.d/conda.sh
conda activate aabo
script_dir="/h/cdoumont/aabo/scripts"
cd $script_dir || exit

# Execute the command
cmd="python3 run_bo.py debug=False"
sbatch --wrap="$cmd" --cpus-per-task=$c --mem=$mem --time=$time \
        --gres=$gres --output="output.log"