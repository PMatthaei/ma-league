
#!/usr/bin/env bash
#
#SBATCH --job-name ma-league
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
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

# Fetch requirements.txt from main folder
cp ./requirements.txt .
pip install -r requirements.txt
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# For CUDA 11, we need to explicitly request the correct version.
pip install git+https://github.com/PMatthaei/ma-env.git

# test cuda
python3 -c "import torch; print(torch.cuda.device_count())"

echo "Execute command as Slurm job..."

# "Starting Experiment Assistent"
chmod 755 run.py

./run.py

