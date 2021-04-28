#!/bin/bash

echo 'Building Image "ma-league:1.0" from Dockerfile.'
# Copy the requirements file to reproduce environment in docker image
cp ../requirements.txt .
docker build -t ma-league:1.0 .
