#!/bin/bash

# SLURM Job Configuration
#SBATCH --account=p70533
#SBATCH --job-name=seg_likelihood
#SBATCH --output=slurm_output/seg_likelihood.out
#SBATCH --error=slurm_output/seg_likelihood.err
#SBATCH --time=00-02:00:00

# Partition and Resources Configuration
#SBATCH --partition=zen2_0256_a40x2
#SBATCH --qos=zen2_0256_a40x2
#SBATCH --gres=gpu:1


# Email Configuration
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=muhammedaabdelaal@gmail.com


# Load Conda Module and Activate Environment
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate /eodc/private/tuwgeo/users/mabdelaa/anaconda3/envs/cnn

# Verify Environment Activation
echo "Activated Conda Environment: $(conda info --envs | grep 'cnn')"
echo "Using Python: $(which python)"

# Check GPU Status
echo "Checking GPU status with nvidia-smi:"
nvidia-smi

# Run Python Script
python_script="/eodc/private/tuwgeo/users/mabdelaa/repos/seg_likelihood/src/train.py"
echo "Executing Python script: $python_script"
/eodc/private/tuwgeo/users/mabdelaa/anaconda3/envs/cnn/bin/python  $python_script
