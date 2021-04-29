#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
name=${USER}_ma_league_${HASH}

title="Select hardware usage for your containered experiment mode:"
prompt="Pick:"
hardware_options=("GPUs (all)" "CPUs (all)" "Quit")
#
#
# HARDWARE SELECT
#
#
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

base_command="python src/main.py "
#
#
# EXPERIMENT SELECT
#
#
title="Select experiment mode:"
prompt="Pick:"
mode_options=("Normal Play" "Self Play" "League Play" "JPC Evaluation" "Quit")
echo "$title"
PS3="$prompt "
select mode in "${mode_options[@]}"; do
  case "$REPLY" in

  1)
    mode="play_mode=normal "
    break
    ;;

  2)
    mode="play_mode=self "
    break
    ;;

  3)
    base_command="python src/league_main.py "
    mode="play_mode=league "
    break
    ;;

  4)
    mode="play_mode=self eval=jpc "
    break
    ;;

  5)
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
# ALGORITHM SELECT
#
#
title="Choose multi-agent reinforcement learning algorithm:"
prompt="Pick:"
base_alg="--config="
algs_options=("QMIX" "Quit")
echo "$title"
PS3="$prompt "
select alg in "${algs_options[@]}"; do
  case "$REPLY" in

  1)
    alg=$base_alg"qmix "
    break
    ;;

  2)
    echo "User forced quit."
    exit
    ;;

  *)
    echo "Invalid option. Try another one. Or Quit"
    continue
    ;;

  esac
done

env_config="--env-config=ma with "

#
#
# SAVE MODEL SELECT
#
#
title="Should training models be saved?:"
prompt="Pick:"
save_options=("Yes" "No" "Quit")
echo "$title"
PS3="$prompt "
select save_model in "${save_options[@]}"; do
  case "$REPLY" in

  1)
    save_model="save_model=True "
    #
    #
    # SAVE INTERVAL INPUT
    #
    #
    echo "Enter desired save interval(in env steps) or press enter for default: "
    read -r save_model_interval
    save_model_interval="save_model_interval=${save_model_interval} "
    break
    ;;

  2)
    save_model="save_model=False "
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
#
# PARALLELISM SELECT
#
#
title="Use parallelism in environment stepping:"
prompt="Pick:"
instances=""
parallel_options=("Yes" "No" "Quit")
echo "$title"
PS3="$prompt "
select parallel in "${parallel_options[@]}"; do
  case "$REPLY" in

  1)
    parallel="runner=parallel "
    #
    #
    # INSTANCES INPUT
    #
    #
    echo "Enter desired number of parallel env instances: "
    read -r instances
    instances="batch_size=${instances} batch_size_run=${instances} "
    break
    ;;

  2)
    parallel="runner=episode "
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

run=$base_command$alg$env_config$mode$save_model$save_model_interval$parallel$instances" headless_controls=False"

# Split run command string to array of strings
read -ra run -d '' <<< "$run"

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