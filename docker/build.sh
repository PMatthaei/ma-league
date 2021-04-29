#!/bin/bash
Help() {
  # Display Help
  echo
  echo "Build script for the ma-league image."
  echo
  echo "Usage:   build.sh [-h] ARGUMENTS"
  echo "Options:"
  echo "  -h           Print this Help."
  echo
  echo "ARGUMENTS    Additional arguments to directly feed into 'docker build' command."
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
echo 'Building Image "ma-league:1.0" from Dockerfile.'
# Copy the requirements file to reproduce environment in docker image
cp ../requirements.txt .
docker build "$@" -t ma-league:1.0 .
