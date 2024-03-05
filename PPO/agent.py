import torch as T
import numpy as np

from .buffer import ReplayBuffer
from .network import ActorNetwork, CriticNetwork

class PPO:
    def __init__(self, input_dim, action_dim, save_dir, alpha=3e-5, batch_size=128, n_epochs=4, 
                       gamma=0.99, gae_lambda=0.95, policy_clip=0.2, beta=1e-3):

        self.actor = ActorNetwork(input_dims = input_dim, n_actions = action_dim, alpha = alpha, chkpt_dir = save_dir)
        self.critic = CriticNetwork(input_dims = input_dim, alpha = alpha, chkpt_dir = save_dir)
        self.memory = ReplayBuffer(batch_size)
        
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.beta = beta
       
    def remember(self, state, mask, action, reward, probs, vals, done):
        self.memory.store_memory(state, mask, action, reward, probs, vals, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, mask):
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)

        if mask is not None:
            mask = T.tensor(np.array([mask]), dtype=T.int64).to(self.actor.device)
            dist = self.actor(state, action_mask=mask)
        else:
            dist = self.actor(state)
            
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        policy_loss_list = []
        value_loss_list = []
        value_estimate_list = []
        entropy_list = []
        
        for _ in range(self.n_epochs):
            state_arr, mask_arr, action_arr, reward_arr, \
            old_prob_arr, vals_arr, dones_arr, batches = self.memory.generate_batches()

            values = T.tensor(vals_arr).to(self.actor.device)
            reward_arr = T.tensor(reward_arr).to(self.actor.device)
            dones_arr = T.tensor(dones_arr).to(self.actor.device)

            deltas = reward_arr[:-1] + self.gamma * values[1:] * (1 - dones_arr[:-1].float()) - values[:-1]
            advantage = T.zeros_like(reward_arr)

            gae = 0
            for t in reversed(range(len(deltas))):
                gae = gae * self.gamma * self.gae_lambda
                gae = gae + deltas[t]
                advantage[t] = gae

            advantage = advantage.to(self.actor.device)
            
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                
                if mask_arr[0] is not None:
                    masks = T.tensor(mask_arr[batch], dtype=T.int64).to(self.actor.device)
                    dist = self.actor(states, action_mask=masks)
                else:
                    dist = self.actor(states)
                    
                entropy = dist.entropy()
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = advantage[batch] * T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss + 0.001*entropy.mean()
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

        policy_loss_list.append(float(actor_loss))
        value_loss_list.append(float(critic_loss))
        value_estimate_list.append(float(T.mean(critic_value)))
        entropy_list.append(float(entropy.mean()))
                
        policy_loss_arr = np.array(policy_loss_list)
        value_loss_arr = np.array(value_loss_list)
        value_estimate_arr = np.array(value_estimate_list)
        entropy_arr = np.array(entropy_list)
        current_lr = self.actor.optimizer.param_groups[0]['lr']
        
        return np.mean(policy_loss_arr), np.mean(value_loss_arr), \
               np.mean(value_estimate_arr), np.mean(entropy_arr), current_lr