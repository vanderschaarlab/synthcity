#!/bin/bash
# add job requirements
#SBATCH -J UIAYN
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=100:00:00 
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=j.l.achterberg@lumc.nl

# add necessary modules
module purge 
module add system/python/3.10.2
module add library/cuda/11.8.0/gcc.8.5.0
module add library/cudnn/11.8/cudnn

source .venv/bin/activate

# run script
python3 UIAYN_experiments/benchmark.py

deactivate
