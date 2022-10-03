import os
import sys
import torch
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from log import Log

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = eval_fns

        self.start_time = time.time()

    def train_iteration(self, algo, num_steps, iter_num=0, print_logs=False):

        train_losses = []

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        Log.log['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                Log.log[f'evaluation/{k}'] = v

        # if algo == 'mcdt':
        #
        #     Log.eval_box_0 = np.array(Log.eval_box_0, dtype=object)
        #     Log.eval_box_1 = np.array(Log.eval_box_1, dtype=object)
        #     Log.eval_box_2 = np.array(Log.eval_box_2, dtype=object)
        #
        #     arr = np.ma.empty((1000, 100, 3))
        #     arr.mask = True
        #     arr[:Log.eval_box_0[0].shape[0], :Log.eval_box_0[0].shape[1], 0] = Log.eval_box_0[0]
        #     arr[:Log.eval_box_0[1].shape[0], :Log.eval_box_0[1].shape[1], 1] = Log.eval_box_0[1]
        #     arr[:Log.eval_box_0[2].shape[0], :Log.eval_box_0[2].shape[1], 2] = Log.eval_box_0[2]
        #     arr.mean(axis=2)
        #     Log.log['evaluation/action[0]'] = arr
        #
        #     arr = np.ma.empty((1000, 100, 3))
        #     arr.mask = True
        #     arr[:Log.eval_box_1[0].shape[0], :Log.eval_box_1[0].shape[1], 0] = Log.eval_box_1[0]
        #     arr[:Log.eval_box_1[1].shape[0], :Log.eval_box_1[1].shape[1], 1] = Log.eval_box_1[1]
        #     arr[:Log.eval_box_1[2].shape[0], :Log.eval_box_1[2].shape[1], 2] = Log.eval_box_1[2]
        #     arr.mean(axis=2)
        #     Log.log['evaluation/action[1]'] = arr
        #
        #     arr = np.ma.empty((1000, 100, 3))
        #     arr.mask = True
        #     arr[:Log.eval_box_2[0].shape[0], :Log.eval_box_2[0].shape[1], 0] = Log.eval_box_2[0]
        #     arr[:Log.eval_box_2[1].shape[0], :Log.eval_box_2[1].shape[1], 1] = Log.eval_box_2[1]
        #     arr[:Log.eval_box_2[2].shape[0], :Log.eval_box_2[2].shape[1], 2] = Log.eval_box_2[2]
        #     arr.mean(axis=2)
        #     Log.log['evaluation/action[2]'] = arr
        #
        #     Log.eval_box_0 = []
        #     Log.eval_box_1 = []
        #     Log.eval_box_2 = []

        Log.log['time/total'] = time.time() - self.start_time
        Log.log['time/evaluation'] = time.time() - eval_start
        Log.log['training/train_loss_mean'] = np.mean(train_losses)
        Log.log['training/train_loss_std'] = np.std(train_losses)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in Log.log.items():
                # if k == 'evaluation/action[0]' or k == 'evaluation/action[1]' or k == 'evaluation/action[2]':
                #     continue
                print(f'{k}: {v}')
            print('=' * 80)

        return Log.log

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        action_preds, rewards_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            Log.log['training/action_error'] = torch.mean((action_preds - action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()