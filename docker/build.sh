#!/bin/bash

echo 'Building Dockerfile with image name pymarl:1.0'
cp ../requirements.txt .
docker build -t ma-league:1.0 .
