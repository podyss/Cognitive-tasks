import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, TD3, A2C
import numpy as np
from generate import generate
import sys
sys.path.insert(0, sys.path[0]+"/../../../")
from models import BaseEnv, TensorboardCallback


class Env(BaseEnv):
    def __init__(self, timelimit=10, k=4):
        action_space =spaces.Box(low=0, high=1, shape=(k,k,2), dtype=np.float32)
        observation_space = spaces.Box(low=0, high=1, shape=(k,k), dtype=np.float32)
        self.k = k
        super().__init__(action_space,observation_space,timelimit)

    def generate(self):
        return generate(self.k)
    
    def step(self, action):
        action = action.argmax(axis=-1)
        reward = -(action==self.get_label()).sum()
        self.count += 1
        done = 0
        if self.count == self.timelimit:
            done = 1
        truncated = 0
        info = {}
        self.get_obs()
        return self.obs, reward, done, truncated, info

env = Env(k=4)
model = SAC('MlpPolicy', env, tensorboard_log='logs/')
# print(model.policy)
model.learn(total_timesteps=10000*10,progress_bar=True,callback=TensorboardCallback())
# model.save('model/ppo.pth')