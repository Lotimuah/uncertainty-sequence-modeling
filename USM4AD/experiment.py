import os
import sys
import yaml
import torch
import wandb
import joblib
import random
import argparse
import datetime
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from evaluation.evaluate_episodes import evaluate_episodes
from models.dt_model import DecisionTransformer
from training.dt_trainer import Trainer
from rl_baselines3_zoo.utils.utils import create_test_env


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    env_name, dataset = variant['env'], variant['dataset_type']
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb')
    algo = variant['algo']

    with open('env_hyperparams.yaml', 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)

    env = create_test_env(
        env_name,
        hyperparams=hyperparams,
    )

    max_ep_len = 1000
    env_targets = [1000]
    scale = 100

    dataset_path = '{}/data/{}_dataset.pkl'.format(os.pardir, dataset)
    trajectories = joblib.load(dataset_path)

    state_dim = trajectories[0]['observations'].shape[2:]
    act_dim = trajectories[0]['actions'].shape[2:]

    states, returns, traj_lens = [], [], []
    for trajectory in trajectories:
        states.append(trajectory['observations'])
        returns.append(trajectory['rewards'].sum())
        traj_lens.append(len(trajectory['actions']))
    returns, traj_lens = np.array(returns), np.array(traj_lens)

    states = np.concatenate(states, axis=0)
    assert states.shape[1:] == (1, 64, 64, 4), f"states.shape[1:] should be (1, 64, 64, 4)"

    # normalize through batch data(states.shape[0])
    # [0]: sequence length (1)
    state_mean, state_std = np.mean(states, axis=0)[0], np.std(states, axis=0)[0]

    print('=' * 60)
    print('Starting new experiment: {}_dataset'.format(dataset))
    print('=' * 60)
    print('Trajectories : {}, Timesteps : {}'.format(len(trajectories), len(states)))
    print('Return mean : {:.2f}, Return std : {:.2f}'.format(np.mean(returns), np.std(returns)))
    print('Return max : {:.2f}, Return min : {:.2f}'.format(np.max(returns), np.min(returns)))
    print('=' * 60)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']

    sorted_inds = np.argsort(returns)   # lowest to highest
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=batch_size, max_len=K):
        batch_inds = np.random.choice(
            np.arange(len(trajectories)),
            size=batch_size,
            replace=True,
            p=p_sample,
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            s.append(traj['observations'][si:si + max_len].reshape(1, -1, *state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, *act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1
            rtg.append(discount_cumsum(traj['rewards'][si:, 0], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, *state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / (state_std + 1e-5)
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, *act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, len = evaluate_episodes(
                        env,
                        algo,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        target_return=target_rew/scale,
                    )
                returns.append(ret)
                lengths.append(len)
            return {
                'target_{}_return_mean'.format(target_rew) : np.mean(returns),
                'target_{}_return_std'.format(target_rew) : np.std(returns),
                'traget_{}_length_mean'.format(target_rew) : np.mean(lengths),
                'target_{}_length_std'.format(target_rew) : np.std(lengths),
            }
        return fn

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=variant['embed_dim'],
        max_ep_len=max_ep_len,
        seq_len=K,
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_fn=variant['activation_fn'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        scheduler=scheduler,
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    date = datetime.datetime.now()
    group_name = '{}-{}-{}-{}'.format(exp_prefix, env_name, algo, dataset)
    exp_prefix = '{}-{}.{}.{}'.format(group_name, date.year, date.month, date.day)

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='MC_SeqModeling',
            config=variant,
            # mode='offline'
        )


    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(algo=algo, num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CarRacing-v0')
    parser.add_argument('--algo', type=str, default='mcdt')
    parser.add_argument('--dataset_type', type=str, default='medium')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_fn', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=3)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--num_steps_per_iter', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', type=bool, default=True)

    args = parser.parse_args()

    experiment('gym', variant=vars(args))