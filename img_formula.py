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


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.model.env, n_eval_episodes=10)
            self.logger.record('reward', mean_reward)
            print(self.n_calls, 'reward: ',mean_reward)
        return True
import ultralytics
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        extractors = {} 
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = nn.Sequential(
                    Conv(c1=3,c2=4,k=3,s=2),
                    Conv(c1=4,c2=8,k=3,s=2),
                    C2f(c1=8,c2=8,shortcut=True),C2f(c1=8,c2=8,shortcut=True),C2f(c1=8,c2=8,shortcut=True),
                    Conv(c1=8,c2=16,k=3,s=2),
                    C2f(c1=16,c2=16,shortcut=True),C2f(c1=16,c2=16,shortcut=True),C2f(c1=16,c2=16,shortcut=True),
                    C2f(c1=16,c2=16,shortcut=True),C2f(c1=16,c2=16,shortcut=True),C2f(c1=16,c2=16,shortcut=True),
                    Conv(c1=16,c2=32,k=3,s=2),
                    C2f(c1=32,c2=32,shortcut=True),C2f(c1=32,c2=32,shortcut=True),C2f(c1=32,c2=32,shortcut=True),
                    C2f(c1=32,c2=32,shortcut=True),C2f(c1=32,c2=32,shortcut=True),C2f(c1=32,c2=32,shortcut=True),
                    SPPF(c1=32,c2=32,k=5),

                    # Conv(c1=3,c2=64,k=3,s=2),
                    # Conv(c1=64,c2=128,k=3,s=2),
                    # C2f(c1=128,c2=128,shortcut=True),C2f(c1=128,c2=128,shortcut=True),C2f(c1=128,c2=128,shortcut=True),
                    # Conv(c1=128,c2=256,k=3,s=2),
                    # C2f(c1=256,c2=256,shortcut=True),C2f(c1=256,c2=256,shortcut=True),C2f(c1=256,c2=256,shortcut=True),
                    # C2f(c1=256,c2=256,shortcut=True),C2f(c1=256,c2=256,shortcut=True),C2f(c1=256,c2=256,shortcut=True),
                    # Conv(c1=256,c2=512,k=3,s=2),
                    # C2f(c1=512,c2=512,shortcut=True),C2f(c1=512,c2=512,shortcut=True),C2f(c1=512,c2=512,shortcut=True),
                    # C2f(c1=512,c2=512,shortcut=True),C2f(c1=512,c2=512,shortcut=True),C2f(c1=512,c2=512,shortcut=True),
                    # Conv(c1=512,c2=1024,k=3,s=2),
                    # C2f(c1=1024,c2=1024,shortcut=True),C2f(c1=1024,c2=1024,shortcut=True),C2f(c1=1024,c2=1024,shortcut=True),
                    # SPPF(c1=1024,c2=1024,k=5),
                    nn.Flatten(),
                )
                total_concat_size += subspace.shape[1] // 16 * subspace.shape[2] // 16 * 32
                # total_concat_size += subspace.shape[1] // 32 * subspace.shape[2] // 32 * 1024
 
        self.extractors = nn.ModuleDict(extractors)
 
        self._features_dim = total_concat_size
 
    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

class Env(gym.Env):
    def __init__(self, timelimit=10):
        self.action_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            spaces={
                "image": spaces.Box(-1, 1, [3,128,256], dtype=np.float32),
            },
        )
        self.obs = self.get_obs()
        self.count = 0
        self.timelimit = timelimit
        
    def get_obs(self):
        # y = np.random.randint(3)
        # self.index = np.random.randint(size=2,low=0,high=10)
        # if y == 1:
        #     self.index[1] = self.index[0]
        # while self.get_ans() != y:
        #     self.index = np.random.randint(size=2,low=0,high=10)

        from generate import generate_img
        img1, self.text1 = generate_img()
        img2, self.text2 = generate_img()
        def normalize(image):
            mean = np.mean(image)
            var = np.mean(np.square(image-mean))
            image = (image - mean)/np.sqrt(var)
            return image
        img1 = normalize(img1)
        img2 = normalize(img2)
        img1 = np.array(img1,dtype=np.float32).transpose((2,0,1))
        img2 = np.array(img2,dtype=np.float32).transpose((2,0,1))
        img = np.concatenate((img1,img2),axis=1)
        # self.obs = {"img1": img1, "img2": img2}
        self.obs = {"image":img}
        return self.obs

    def get_ans(self, ):
        res1 = eval(self.text1)
        res2 = eval(self.text2)
        if res1 == res2:
            return 1
        elif res1 < res2:
            return 0
        else:
            return 2

    def reset(self, seed=0):
        self.count = 0
        self.obs = self.observation_space.sample()
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
        obs = self.get_obs()
        return obs, reward, done, truncated, info
    
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
# model = SAC('MultiInputPolicy', env, tensorboard_log='simple/logs/', buffer_size=int(1e4),
#             policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor))
model = PPO('MultiInputPolicy', env, tensorboard_log='simple/logs/',
            policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor))
# print(model.policy)
model.learn(total_timesteps=10000*10,progress_bar=True,callback=TensorboardCallback())
model.save('simple/model/ppo.pth')