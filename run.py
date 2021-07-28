#!/usr/bin/env python3
import re
import subprocess
from typing import Dict

from pip._vendor.distlib.compat import raw_input

args = []


def options_string(options: Dict):
    return ", ".join([f"{num}) {option}" for num, option in options.items()])


def select(title, options, arg, only_arg=False):
    response = None
    options.append("Quit")
    num_options = {str(num + 1): option for num, option, in enumerate(options)}
    while response not in options:
        response = raw_input(f"----- Choose {title}:\n {options_string(num_options)} \n Enter: ")
        response = num_options[response]
        if response == "Quit":
            print("Quitting Assistent")
            exit(1)
        else:
            yes = True if response == "True" else False
            print(f"\n Choice: {response} \n")

            if yes and only_arg:
                args.append(f"{arg}")
                return yes
            else:
                response = response if yes else response.lower()
                args.append(f"{arg}={response}")
                return yes


def enter(title, arg):
    response = raw_input(f"----- Enter desired {title}: \n ")
    print(f"\n Choice: {response} \n")
    args.append(f"{arg}={response}")


def choice(title, arg, only_arg=False) -> bool:
    yes = select(title=f"Activate {title} ?", options=["True", "False"], arg=arg, only_arg=only_arg)
    return yes


if __name__ == '__main__':
    print("Starting Experiment Assistant...")
    python_cmd_base = "python src/central_worker_main.py"

    select(title="MARL algorithm Config", options=["QMIX", "VDN", "IQL"], arg="--config")
    select(title="Environment Config", options=["ma"], arg="--env-config")
    select(title="League Config", options=["Ensemble", "Matchmaking", "Rolebased"], arg="--league-config")
    select(title="Experiment Config", options=["Ensemble", "Matchmaking", "Rolebased"], arg="--experiment")
    select(title="Matchmaking Config", options=["Adversaries", "FSP", "PFSP"], arg="--matchmaking")

    enter("League size", arg="--league_size")
    enter("Team size", arg="--team_size")

    choice("CUDA", arg="--use_cuda")
    choice("CUDA work balance", arg="--balance-cuda-workload", only_arg=True)

    yes = choice("model saving", arg="--save_model")
    save_model_interval = enter("model saving interval", arg="--save_model_interval") if yes else ""

    choice("Tensorboard", arg="--use_tensorboard")

    python_cmd = f"{python_cmd_base} {' '.join(args)} force-unit --unique --role=HEALER --attack=RANGED"

    python_cmd = re.sub(' +', ' ', python_cmd)  # Clean multiple whitespaces
    print(f"\n\n  Command: {python_cmd} \n\n")

    subprocess.check_call(python_cmd.split(" "))

    exit(1)
