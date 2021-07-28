#!/usr/bin/env python3
import re
import subprocess
import uuid

from pip._vendor.distlib.compat import raw_input


def select(title, options, arg):
    response = None
    options.append("Quit")
    num_options = {str(num + 1): option for num, option, in enumerate(options)}
    while response not in options:
        response = raw_input(f"Choose {title}: {num_options} ")
        response = num_options[response]
        if response == "Quit":
            print("Quitting Assistent")
            exit(1)
        else:
            print(f"Choice: {response}")
            response = response.lower()
            return f"{arg}={response}"


def enter(title, arg):
    response = raw_input(f"Enter desired {title}: ")
    print(f"Choice: {response}")
    return f"{arg}={response}"


def choice(title, arg):
    print(f"Activate {title} ?")
    response = select(title="", options=["Yes", "No"], arg=arg)
    yes = "yes" in response
    final_response = response.replace("yes", "True").replace("no", "False")
    return final_response, yes


if __name__ == '__main__':
    print("Experiment Run Assistent")
    name = f"ma-league-{uuid.uuid4().hex[0:5]}"
    python_cmd_base = "python src/central_worker_main.py"

    alg = select(title="MARL algorithm Config", options=["QMIX", "VDN", "IQL"], arg="--config")
    env = select(title="Environment Config", options=["ma"], arg="--env-config")
    league = select(title="League Config", options=["Ensemble", "Matchmaking", "Rolebased"], arg="--league-config")
    experiment = select(title="Experiment Config", options=["Ensemble", "Matchmaking", "Rolebased"], arg="--experiment")
    matchmaking = select(title="Matchmaking Config", options=["Adversaries", "FSP", "PFSP"], arg="--matchmaking")

    league_size = enter("League size", arg="--league_size")
    team_size = enter("Team size", arg="--team_size")

    use_cuda, _ = choice("CUDA", arg="--use_cuda")

    save_model, yes = choice("model saving", arg="--save_model")
    save_model_interval = enter("model saving interval", arg="--save_model_interval") if yes else ""

    python_cmd = f"{python_cmd_base} {experiment} {matchmaking} {alg} {env} {league} {league_size} {team_size} {use_cuda} {save_model} {save_model_interval} force-unit --unique --role=HEALER --attack=RANGED"

    python_cmd = re.sub(' +', ' ', python_cmd)  # Clean multiple whitespaces
    print(f"Python Command: {python_cmd}")

    subprocess.check_call(python_cmd.split(" "))

    exit(1)
