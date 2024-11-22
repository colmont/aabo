#!/bin/bash

# sbatch params
c=4
time=8:00:00
mem=8GB
qos=m2

# Environment setup
source /h/cdoumont/miniconda3/etc/profile.d/conda.sh
conda activate aabo
script_dir="/h/cdoumont/aabo/scripts"
cd $script_dir || exit

# Task and seed setup
SEEDS="1 2 3 4 5 6 7 8 9 10"
# SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
TASKS="dna"
# TASKS="hartmann6 dna lunar rover fexo"

# Loop over tasks and seeds
for task in $TASKS; do
    for seed in $SEEDS; do
        cmd="python3 run_bo.py benchmark=$task seed=$seed debug=False"
        sbatch --wrap="$cmd" --cpus-per-task=$c --mem=$mem --time=$time \
               --output="output.log" --qos=$qos --partition=rtx6000 --gres=gpu:1
    done
done