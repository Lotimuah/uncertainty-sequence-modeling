import os
import sys
import gym
import yaml
import json
import wandb
import joblib
import argparse
import datetime
import numpy as np

import d3rlpy
from rl_baselines3_zoo.utils.utils import create_test_env
from stable_baselines3.common.log_to_wandb import CollectLog

with open('env_hyperparams.yaml', 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)

env = create_test_env(
    'CarRacing-v0',
    hyperparams=hyperparams,
)

sac = d3rlpy.algos.SAC(batch_size=32, scaler='pixel', use_gpu=True)

buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

date = datetime.datetime.now()

wandb.init(
    name='gym-CarRacing-v0-sac-{}.{}.{}'.format(date.year, date.month, date.day),
    group='gym-CarRacing-v0-sac',
    project='OnlineRL',
    # config=variant,
    # mode='offline',
    allow_val_change=True,
)

sac.fit_online(env, buffer, n_steps=10000000, n_steps_per_epoch=10000, eval_env=env)
