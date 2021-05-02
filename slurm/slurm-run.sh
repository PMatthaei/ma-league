
#!/usr/bin/env bash

cmd="$@"
echo "Forcing python3."
cmd=$(echo "$cmd" | sed "s/python/python3/")
echo "Slurm script received command: '$cmd'"

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

echo "Execute command as Slurm job..."
# train
eval $cmd

