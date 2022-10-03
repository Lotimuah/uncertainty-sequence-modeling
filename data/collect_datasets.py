import os
import sys
import yaml
import joblib
import argparse
import numpy as np

from stable_baselines3.common.utils import set_random_seed
from rl_baselines3_zoo.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams


def get_data_bucket():
    data = dict(
        observations = [],
        next_observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = [],
    )
    return data


def rollout_data():

    for type in data_type:
        dataset = []
        trajectory = get_data_bucket()

        epi_len = 0
        epi_returns = 0
        checkpoint = 1
        mix_iter = 0
        done = False
        obs = env.reset()

        if type == 'random':
            pass
        elif type == 'online':
            model_path = os.path.join(log_path, 'rl_model_{}_steps'.format(checkpoint))
            model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)
        else:
            model_path = os.path.join(log_path, '{}_model'.format(type))
            model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

        while len(dataset) < num_data:

            if type == 'random':
                action = env.action_space.sample()
                action = np.reshape(action, (1, -1))
            elif type == 'mixture':
                if mix_iter % 2 == 0:
                    action = env.action_space.sample()
                    action = np.reshape(action, (1, -1))
                else:
                    action, _ = model.predict(obs, state=None, deterministic=True)
            else:
                action, _ = model.predict(obs, state=None, deterministic=True)

            next_obs, reward, done, info = env.step(action)

            epi_len += 1
            epi_returns += reward
            timeout = False
            terminal = False

            if epi_len == max_path:
                timeout = True
            elif done:
                terminal = True

            trajectory['observations'].append(obs)
            trajectory['next_observations'].append(next_obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['terminals'].append(terminal)
            trajectory['timeouts'].append(timeout)

            obs = next_obs

            if terminal or timeout:
                print("========================= Finished trajectory =========================")
                print("Type : {} | Len : {} | Returns : {} | Progress : {}/{}".format(type, epi_len, epi_returns, len(dataset) + 1, num_data))
                print("=" * 69)
                obs = env.reset()
                epi_len = 0
                epi_returns = 0
                checkpoint += 1
                mix_iter += 1

                for k in trajectory:
                    if k == 'terminals' or k == 'timeouts':
                        trajectory[k] = np.array(trajectory[k]).astype(np.bool_)
                    else:
                        trajectory[k] = np.array(trajectory[k]).astype(np.float32)

                dataset.append(trajectory)
                trajectory = get_data_bucket()

                if type == 'online' and checkpoint <= num_data:
                    model_path = os.path.join(log_path, 'rl_model_{}_steps'.format(checkpoint))
                    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, **kwargs)

        path = os.path.join(storage_path, '{}_dataset.pkl'.format(type))
        joblib.dump(dataset, path)
        env.close()

    complete = 'All Processes are Complete.'
    return complete



if __name__ == '__main__':

    env_name = 'CarRacing-v0'
    max_path = 1000
    num_data = 100
    folder = os.path.join(os.pardir, 'rl_baselines3_zoo/logs')
    algo = 'ppo'
    seed = 0
    storage_path = os.getcwd()

    log_path = os.path.join(folder, algo, '{}_{}'.format(env_name, 0))
    stats_path = os.path.join(log_path, env_name)
    # hyperparams, _ = get_saved_hyperparams(stats_path, test_mode=True)

    with open('env_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)

    set_random_seed(seed)

    env = create_test_env(
        env_name,
        hyperparams=hyperparams,
    )

    kwargs = dict(seed=seed)

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_scheduler": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    data_type = ['random', 'expert', 'mixture', 'online', 'medium']

    print(rollout_data())
