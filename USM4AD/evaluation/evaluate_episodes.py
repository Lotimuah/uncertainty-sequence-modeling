import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from log import Log


def enable_dropout_in_eval(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def evaluate_episodes(
        env,
        algo,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=10,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
):

    model.eval()
    model.to(device=device)

    if algo == 'mcdt':
        enable_dropout_in_eval(model=model)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    states = torch.from_numpy(state).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, *act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0

    # traj_box_0 = []
    # traj_box_1 = []
    # traj_box_2 = []
    for t in range(max_ep_len):

        actions = torch.cat([actions, torch.zeros((1, *act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        if algo == 'dt':
            action = model.get_action(
                (states - state_mean) / (state_std + 1e-5),
                actions,
                rewards,
                target_return,
                timesteps,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

        elif algo == 'mcdt':
            step_box = []
            # step_box_0 = []
            # step_box_1 = []
            # step_box_2 = []
            for _ in range(50):
                action = model.get_action(
                    (states - state_mean) / (state_std + 1e-5),
                    actions,
                    rewards,
                    target_return,
                    timesteps,
                )
                actions[-1] = action
                action = action.detach().cpu().numpy()
                step_box.append(action)
            #     step_box_0.append(action[0])
            #     step_box_1.append(action[1])
            #     step_box_2.append(action[2])
            # step_box_0 = np.array(step_box_0, dtype=object)
            # step_box_1 = np.array(step_box_1, dtype=object)
            # step_box_2 = np.array(step_box_2, dtype=object)
            # traj_box_0.append(step_box_0)
            # traj_box_1.append(step_box_1)
            # traj_box_2.append(step_box_2)

            action = np.mean(step_box, axis=0)

        action = action.reshape(1, -1)
        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, *state_dim)
        states = torch.cat([states, cur_state], dim=0)
        reward = reward.item()
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward/scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    # if algo == 'mcdt':
    #     traj_box_0 = np.array(traj_box_0, dtype=object)
    #     traj_box_1 = np.array(traj_box_1, dtype=object)
    #     traj_box_2 = np.array(traj_box_2, dtype=object)
    #     Log.eval_box_0.append(traj_box_0)
    #     Log.eval_box_1.append(traj_box_1)
    #     Log.eval_box_2.append(traj_box_2)

    return episode_return, episode_length

