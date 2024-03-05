from PPO.agent import PPO
from PPO.lunar_lander import LunarLander

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time

def test():
    env = LunarLander(
        render_mode="human",
        gravity = -10.0,
        enable_wind = False,
        wind_power = 10.0,
        turbulence_power = 1.5
        ) 
    
    buffer_size = 2048
    batch_size = 32
    n_epochs = 4
    alpha = 1e-3
    save_dir = './save/'

    agent = PPO(input_dim = env.observation_space.shape, 
                action_dim = env.action_space.n, 
                save_dir = save_dir, 
                alpha=alpha, batch_size=batch_size, 
                n_epochs=n_epochs, gamma=0.99, 
                gae_lambda=0.95, policy_clip=0.2, 
                beta=1e-3)
    
    n_games = 10
    agent.load_models()
    
    for i in range(n_games):
        observation, info = env.reset()
        done = False
        reach_goal = False
        while (not done) :
            action, prob, val = agent.choose_action(observation,None)

            observation_, reward, done, reach_goal, info = env.step(action = action)
            
            observation = observation_
             

        
    env.close()
    
if __name__ =='__main__':
    test()
