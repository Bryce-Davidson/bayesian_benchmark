#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --output=main_output.log
#SBATCH --error=main_error.err
#SBATCH --job-name=main_job

# Load necessary modules (if any)
# module load python/3.8

# Run the Python script
python main.py
