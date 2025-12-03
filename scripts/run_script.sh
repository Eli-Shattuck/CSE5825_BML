#!/usr/bin/env bash

# SBATCH --partition=priority-gpu
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=16
# SBATCH --output=.cache/analysis_out_%j.log
# SBATCH --error=.cache/analysis_err_%j.log
# SBATCH --time=0

SCRIPT=$1

python "$SCRIPT"
