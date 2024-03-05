import os
import torch as T
import torch.nn as nn
import torch.optim as optim

from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, chkpt_dir, fc1_dims=64, fc2_dims=64):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        input_dim = input_dims[0] if isinstance(input_dims, tuple) else input_dims
        self.actor = nn.Sequential(
                nn.Linear(input_dim, fc1_dims),
                nn.Tanh(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.Tanh(),
                nn.Linear(fc2_dims, n_actions)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action_mask=None):
        if action_mask is not None:
            dist = self.actor(state)
            mask_digit = T.where(action_mask.bool(), dist, T.tensor(-1e+8).to(self.device))
            dist = self.softmax(mask_digit)
            dist = Categorical(dist)
        else:
            dist = self.softmax(self.actor(state))
            dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, chkpt_dir, fc1_dims=64, fc2_dims=64):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        input_dim = input_dims[0] if isinstance(input_dims, tuple) else input_dims
        self.critic = nn.Sequential(
                nn.Linear(input_dim, fc1_dims),
                nn.Tanh(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.Tanh(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))