#!/usr/bin/env bash

# SBATCH --partition=priority
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=8
# SBATCH --output=.cache/analysis_out_%j.log
# SBATCH --error=.cache/analysis_err_%j.log
# SBATCH --time=0

SCRIPT=$1

python "$SCRIPT"
