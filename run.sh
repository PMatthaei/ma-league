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

while getopts ":h" option; do
  case $option in
  h) # display Help
    Help
    exit
    ;;
  *)
    break
    ;;
  esac
done

base_command="python src/central_worker_main.py "

#
#
# ALGORITHM SELECT
#
#
title="Choose multi-agent reinforcement learning algorithm:"
prompt="Pick:"
base_alg="--config="
algs_options=("QMIX" "VDN" "IQL" "Quit")
echo "$title"
PS3="$prompt "
select alg in "${algs_options[@]}"; do
  case "$REPLY" in

  1)
    alg=$base_alg"qmix "
    break
    ;;

  2)
    alg=$base_alg"vdn "
    break
    ;;

  3)
    alg=$base_alg"iql "
    break
    ;;

  4)
    echo "User forced quit."
    exit
    ;;

  *)
    echo "Invalid option. Try another one. Or Quit"
    continue
    ;;

  esac
done

#
#
# EXPERIMENT SELECT
#
#
title="Choose experiment:"
prompt="Pick:"
base_league="--league-config="
exp_options=("Ensemble Self-Play" "Matchmaking League")
echo "$title"
PS3="$prompt "
select exp in "${exp_options[@]}"; do
  case "$REPLY" in

  1)
    exp=$base_league"ensemble "
    break
    ;;

  2)
    exp=$base_league"matchmaking "
    break
    ;;

  3)
    echo "User forced quit."
    exit
    ;;

  *)
    echo "Invalid option. Try another one. Or Quit"
    continue
    ;;

  esac
done

#
# Sizes
#
echo "Enter desired league size: "
read -r league_size
league_size=" --league_size=${league_size} "
echo "Enter desired team size: "
read -r team_size
team_size=" --team_size=${team_size} "

#
# FIXED ENVIRONMENT
#
env_config="--env-config=ma "


run=$base_command$alg$env_config$exp$league_size$team_size"--save_model=True --save_interval=250000 --headless_controls=False --use_tensorboard=True force-unit --unique --role=HEALER --attack=RANGED"

# Split run command string to array of strings
read -ra run -d '' <<<"$run"

#
#
# INFRASTRUCTURE SELECT
#
#
title="On which infrastructure should this experiment run:"
prompt="Pick:"
instances=""
infra_options=("Docker" "Slurm" "Quit")
echo "$title"
PS3="$prompt "
select infra in "${infra_options[@]}"; do
  case "$REPLY" in

  1)
    #
    #
    # HARDWARE SELECT
    #
    #
    title="Select hardware usage for your containered experiment mode:"
    prompt="Pick:"
    hardware_options=("GPUs (all)" "CPUs (all)" "Quit")
    echo "$title"
    all_cpus="$(grep -c ^processor /proc/cpuinfo).0"
    PS3="$prompt "
    select hardware in "${hardware_options[@]}"; do
      case "$REPLY" in

      1)
        hardware=("--gpus" "all")
        break
        ;;

      2)
        hardware=("--cpus" "${all_cpus}")
        break
        ;;

      3)
        echo "User forced quit."
        exit
        ;;

      *)
        echo "Invalid option. Try another one. Or Quit"
        continue
        ;;

      esac
    done
    HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
    name=${USER}_ma_league_${HASH}
    echo "Launching docker container named '${name}' on '${hardware[*]}'"
    echo "Command: '${run[*]}'"
    docker run \
      "${hardware[@]}" \
      --name "$name" \
      --user "$(id -u)":"$(id -g)" \
      -v "$(pwd)":/ma-league \
      -t ma-league:1.0 \
      "${run[@]}"

    # !! For development: Changes within code will be mounted into the docker container !!
    break
    ;;

  2)
    echo "Launching in slurm cluster"
    echo "Command: '${run[*]}'"
    sh ./slurm/slurm-run.sh ${run[*]}
    break
    ;;

  3)
    echo "User forced quit."
    exit
    ;;

  *)
    echo "Invalid option. Try another one. Or Quit"
    continue
    ;;

  esac
done
