#! /bin/bash

source ~/.bashrc
cd ~/thesis/USM4AD
export DISPLAY=:0

data=('random' 'expert' 'mixture' 'online' 'medium')

for d_type in ${data[@]};
do
  python experiment.py --dataset_type $d_type
  cd ~/thesis/USM4AD/wandb
  wandb sync latest-run
  cd ~/thesis/USM4AD
done