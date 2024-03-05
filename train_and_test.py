from PPO.agent import PPO
from PPO.lunar_lander import LunarLander

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def curve (ax,epo,score_history,smoothed_rewards):
    ax.clear()
    ax.plot(epo, score_history, label='Reward')
    ax.plot(epo, smoothed_rewards, label='Smoothed Reward', color='r')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.set_title('Reward vs. Epoch')
    ax.legend()
    plt.savefig(f'save/reward_epoch.png')

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def train():

    env = LunarLander(
        render_mode="rgb",
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

    fig, ax = plt.subplots()
    agent = PPO(input_dim = env.observation_space.shape, 
                action_dim = env.action_space.n, 
                save_dir = save_dir, 
                alpha=alpha, batch_size=batch_size, 
                n_epochs=n_epochs, gamma=0.99, 
                gae_lambda=0.95, policy_clip=0.2, 
                beta=1e-3)
    
    
    n_games = 2001
    best_score = -500
    
    epo = []
    data = []
    score_history = []
    game_history =[]
    
    learn_iters = 0
    n_steps = 0

    for i in range(n_games):
        if i %100 ==0:
            env.render_mode = 'human'
        else : env.render_mode = 'rgb'
        observation, info = env.reset()
        game_steps = 0
        score = 0
        count = 0
        
        while (count<=1) :
            game_steps += 1
            n_steps += 1
            action, prob, val = agent.choose_action(observation,None)

            observation_, reward, done, reach_goal, info = env.step(action = action)
            
            if reward >200: reward = 200
            elif reward < -200: reward = -200
             
            if game_steps > 1000 : 
                reward = reward -50
                done = True 
                
            score += reward
            
            agent.remember(state=observation,
                           mask = None, action=action,
                           reward=reward, probs=prob, 
                           vals=val, done=done)
            observation = observation_
             
            if n_steps % buffer_size == 0:
                agent.learn()
                learn_iters += 1 
                
            if done == True:count+=1
            
            if reach_goal:
                reach_goal = 1
                game_history.append(1)
            else: 
                reach_goal = 0
                game_history.append(0)
            
            if score < -1000 : 
                score = -1000
                break
            
        epo.append(i)    
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        game_win_rate = np.mean(game_history[-100:])

        if score > best_score: 
            best_score = score
            agent.save_models()
            
        if i%1000 ==0:
            agent.save_models()
            
        print('Episode:', i, 'Score: %.1f' % score,'Best score: %.1f' % best_score ,'Avg score: %.1f' % avg_score,
                'Time_steps:', n_steps, 'Learning_steps:', learn_iters,'Game-win:',  reach_goal , 'Game-win-rate:%.3f' % game_win_rate )
        
        smoothed_rewards = smooth_curve(score_history)
        curve(ax,epo,score_history,smoothed_rewards)

        # save data
        data.append([i, score, best_score, avg_score, n_steps, learn_iters,reach_goal, game_win_rate])
        df = pd.DataFrame(data, columns=['Episode', 'Score', 'Best Score', 'Average Score', 'Time Steps', 'Learning Steps','Game-win','Game win rate'])
        df.to_csv('save/training_data.csv', index=False)
        
    env.close()
    
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
    train()
    test()
