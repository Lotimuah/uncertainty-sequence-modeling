import os
import sys
import torch
import torch.nn as nn
import transformers
from models.trajectory_gpt2 import GPT2Model
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from log import Log


class DecisionTransformer(nn.Module):
    def __init__(self,
                 state_dim,
                 act_dim,
                 hidden_size,
                 max_ep_len,
                 seq_len=None,
                 action_tanh=True,
                 **kwargs):
        super(DecisionTransformer, self).__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.Linear(256, hidden_size),
        )
        self.embed_action = nn.Linear(*self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, *self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_len = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        states = states.reshape(batch_size*seq_len, *self.state_dim).permute(0, 3, 1, 2)
        state_embeddings = self.embed_state(states)
        state_embeddings = state_embeddings.reshape(batch_size, seq_len, -1)
        action_embeddings = self.embed_action(actions)
        return_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        return_embeddings = return_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_len, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_len)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )

        x = transformer_outputs['last_hidden_state']

        x = x.reshape(batch_size, seq_len, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        # state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])

        return action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):

        states = states.reshape(1, -1, *self.state_dim)
        actions = actions.reshape(1, -1, *self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.seq_len is not None:
            states = states[:, -self.seq_len:]
            actions = actions[:, -self.seq_len:]
            returns_to_go = returns_to_go[:, -self.seq_len:]
            timesteps = timesteps[:, -self.seq_len:]

            attention_mask = torch.cat([torch.zeros(self.seq_len - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            states = torch.cat(
                [torch.zeros((states.shape[0], self.seq_len - states.shape[1], *self.state_dim), device=states.device),
                 states], dim=1).to(dtype=torch.float32)

            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.seq_len - actions.shape[1], *self.act_dim), device=actions.device),
                 actions], dim=1).to(dtype=torch.float32)

            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.seq_len - returns_to_go.shape[1], 1), device=returns_to_go.device),
                 returns_to_go], dim=1).to(dtype=torch.float32)

            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.seq_len - timesteps.shape[1]), device=timesteps.device),
                 timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        action_preds, return_preds = self.forward(states, actions, None, returns_to_go, timesteps, attention_mask, **kwargs)

        return action_preds[0, -1]
