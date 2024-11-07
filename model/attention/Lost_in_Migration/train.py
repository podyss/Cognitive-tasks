import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import PIL
from torchvision import models,transforms
from typing import List
from random import choice
from generate import generate
import ultralytics
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.model.env, n_eval_episodes=10)
            self.logger.record('reward', mean_reward)
            print(self.n_calls, 'reward: ',mean_reward)
        return True

class Env(gym.Env):
    def __init__(self, timelimit=10):
        self.action_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = Box(low=-10, high=20, shape=(5,3), dtype=np.float32)
        self.get_obs()
        self.count = 0
        self.timelimit = timelimit
        
    def get_obs(self):        
        self.obs, self.label = generate()
        return self.obs

    def get_ans(self):
        return self.label

    def reset(self, seed=0):
        self.count = 0
        self.get_obs()
        info = {}
        return self.obs, info

    def close(self):
        pass

    def step(self, action):
        action = int(action.argmax())
        if action==self.get_ans():
            reward = 0
        else:
            reward = -1
        self.count += 1
        done = 0
        if self.count == self.timelimit:
            done = 1
        truncated = 0
        info = {}
        self.get_obs()
        return self.obs, reward, done, truncated, info
    
    def test(self):
        for x in self.obs:
            print(x)
            inp = input()
            if inp == '=':
                action = 1
            elif inp == '<':
                action = 0
            else:
                action = 2
            if action == self.get_ans():
                reward = 1
            else:
                reward = 0
            self.count += 1
            print('reward: ',reward)

env = Env()
model = SAC('MlpPolicy', env, tensorboard_log='logs/')
# print(model.policy)
model.learn(total_timesteps=10000*10,progress_bar=True,callback=TensorboardCallback())
# model.save('model/ppo.pth')
