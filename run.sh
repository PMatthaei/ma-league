#!/bin/bash
Help() {
  # Display Help
  echo
  echo "Run script for the ma-league. Build your experiment command and choose infrastructure to run the command with."
  echo "For a detailed usage guide visit the wiki at: https://github.com/PMatthaei/ma-league/wiki"
  echo
  echo "Usage:   run.sh [-h]"
  echo "Options:"
  echo "  -h           Print this Help."
  echo
}

if [ -z "$(ls -A ./venv/ma-league)" ]; then
   echo "Build venv..."
  python3 -m venv ./venv/ma-league
  echo "Activate venv..."
  source ./venv/ma-league/bin/activate
  echo "Install requirements..."
  pip install -U pip setuptools wheel
  # Fetch requirements.txt from main folder
  cp ./requirements.txt .
  pip install -r requirements.txt
  # For CUDA 11, we need to explicitly request the correct version.
  pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
  # Install env from github
  pip install git+https://github.com/PMatthaei/ma-env.git
else
   echo "venv already created. Skipping..."
fi

echo "Starting Experiment Assistent"

chmod 755 run.py

./run.py
