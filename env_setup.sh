#!/bin/bash

conda env create -f environment.yml

source ~/Programs/anaconda3/etc/profile.d/conda.sh
conda activate drl

mkdir -p tmp && cd tmp && git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym && pip install -e .
