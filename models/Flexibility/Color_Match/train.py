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
    def __init__(self, timelimit=10):
        action_space =spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        observation_space = spaces.Box(low=0, high=4, shape=(2,2), dtype=np.float32)
        super().__init__(action_space,observation_space,timelimit)

    def generate(self):
        return generate()

env = Env()
model = SAC('MlpPolicy', env, tensorboard_log='logs/')
# print(model.policy)
model.learn(total_timesteps=10000*10,progress_bar=True,callback=TensorboardCallback(lim=10))
# model.save('model/ppo.pth')
