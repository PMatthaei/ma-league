#!/usr/bin/env bash
#
#SBATCH --job-name ma-league
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1

# debug info
hostname
which python3
nvidia-smi

env

# venv
python3 -m venv ./venv/ma-league
source ./venv/ma-league/bin/activate
pip install -U pip setuptools wheel
# For CUDA 11, we need to explicitly request the correct version
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
# test cuda
python3 -c "import torch; print(torch.cuda.device_count())"

# download example script for CNN training
SRC=src/${SLURM_ARRAY_JOB_ID}
mkdir -p ${SRC}
wget https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py -O ${SRC}/torch-test.py
cd ${SRC}

# train
python3 ./torch-test.py
