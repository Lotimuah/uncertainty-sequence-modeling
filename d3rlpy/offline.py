import os
import sys
import yaml
import json
import wandb
import joblib
import argparse
import datetime
import numpy as np

import d3rlpy
from d3rlpy.dataset import Episode, MDPDataset, Transition
from rl_baselines3_zoo.utils.utils import create_test_env
from stable_baselines3.common.log_to_wandb import CollectLog
from d3rlpy.models.encoders import PixelEncoderFactory
from sklearn.model_selection import train_test_split

def trans_data(x):

    bucket = dict(
        observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = [],
    )

    for epi in x:
        for k in ['observations', 'actions', 'rewards', 'terminals', 'timeouts']:
            bucket[k].append(epi[k])

    for k in bucket:
        bucket[k] = np.concatenate(bucket[k])

        if k == 'actions' or k == 'rewards':
            bucket[k] = bucket[k].squeeze()

        if k == 'observations':
            bucket[k] = bucket[k].squeeze()
            bucket[k] = np.transpose(bucket[k], (0, 3, 1, 2))

    mdp_dataset = MDPDataset(
        observations=np.array(bucket['observations'], dtype=np.uint8),   # (B, H, W) -> (B, 4, H, W)
        actions=np.array(bucket['actions'], dtype=np.float32),
        rewards=np.array(bucket['rewards'], dtype=np.float32),
        terminals=np.array(bucket['terminals'], dtype=np.float32),
        episode_terminals=np.array(np.logical_or(bucket['terminals'], bucket['timeouts'])),
    )

    return mdp_dataset


def experiment(
    exp_prefix,
    variant,
):
    data_type = variant['dataset_type']
    data_path = '../data/{}_dataset.pkl'.format(data_type)
    dataset = joblib.load(data_path)

    dataset = trans_data(dataset)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.3)

    total_steps = variant['total_steps']
    n_steps_per_epoch = variant['n_steps_per_epoch']

    env_id = variant['env']
    algo = variant['algo']
    dataset_type = variant['dataset_type']

    # factory = PixelEncoderFactory(filters=[(16, 8, 4), (32, 5, 2)], feature_size=1024)

    # prepare algorithm
    if algo == 'cql':
        _algo = d3rlpy.algos.CQL(batch_size=32, use_gpu=True)
    elif algo == 'awac':
        _algo = d3rlpy.algos.AWAC(batch_size=32, actor_encoder_factory='pixel', critic_encoder_factory='pixel', scaler='pixel', action_scaler='min_max', reward_scaler='min_max', use_gpu=True)

    date = datetime.datetime.now()
    group_name = '{}-{}-{}'.format(exp_prefix, env_id, algo)
    exp_prefix = '{}-{}-{}.{}.{}'.format(group_name, dataset_type, date.year, date.month, date.day)

    if variant['log_to_wandb']:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='OfflineRL',
            config=variant,
            # mode='offline',
            allow_val_change=True,
        )

    with open('env_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)

    env = create_test_env(
        env_id,
        hyperparams=hyperparams,
    )

    # train offline
    _algo.fit(train_episodes,
              eval_episodes=test_episodes,
              n_epochs=1000,
              # n_steps=total_steps,
              # n_steps_per_epoch=n_steps_per_epoch,
              scorers={
                  'environment': d3rlpy.metrics.evaluate_on_environment(env),
                  'td_error': d3rlpy.metrics.td_error_scorer
              })

    with open('{}/params.json'.format(CollectLog.path)) as f:
        config = json.load(f)

    wandb.config.update(config, allow_val_change=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CarRacing-v0')
    parser.add_argument('--algo', type=str, default='awac')
    parser.add_argument('--total_steps', type=int, default=300000)
    parser.add_argument('--n_steps_per_epoch', type=int, default=300)
    parser.add_argument('--dataset_type', type=str, default='mixture')
    parser.add_argument('--log_to_wandb', type=bool, default=True)

    args = parser.parse_args()

    experiment('gym', variant=vars(args))
