#!/bin/bash
HASH=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
name=${USER}_ma_league_${HASH}

title="Select hardware usage for your containered experiment run:"
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

#
#
# EXPERIMENT SELECT
#
#
title="Select experiment run:"
prompt="Pick:"
run_options=("Normal Play" "Self Play" "League Play" "JPC Evaluation" "Quit")
echo "$title"
PS3="$prompt "
select run in "${run_options[@]}"; do
  case "$REPLY" in

  1)
    run="python src/main.py --config=qmix --env-config=ma with play_mode=normal save_model=True headless_controls=False"
    break
    ;;

  2)
    run="python src/main.py --config=qmix --env-config=ma with play_mode=self save_model=True headless_controls=False"
    break
    ;;

  3)
    run="python src/league_main.py --config=qmix --env-config=ma with play_mode=league save_model=True --league-config=default headless_controls=False"
    break
    ;;

  4)
    run="python src/main.py --config=qmix --env-config=ma with play_mode=self save_model=True eval=jpc headless_controls=False runner=parallel"
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

# Split run command to array
read -ra run -d '' <<< "$run"

echo "Launching container named '${name}' on '${hardware[*]}' with command '${run[*]}'"
docker run \
  "${hardware[@]}" \
  --name "$name" \
  --user "$(id -u)":"$(id -g)" \
  -v "$(pwd)":/ma-league \
  -t ma-league:1.0 \
   "${run[@]}"
