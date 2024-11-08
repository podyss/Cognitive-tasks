import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.model.env, n_eval_episodes=10)
            self.logger.record('reward', mean_reward)
            print(self.n_calls, 'reward: ',mean_reward)
        return True

class BaseEnv(gym.Env):
    def __init__(self, action_space, observation_space, timelimit=10):
        self.action_space = action_space
        self.observation_space = observation_space
        self.get_obs()
        self.count = 0
        self.timelimit = timelimit

    def generate(self):
        pass
        
    def get_obs(self):
        self.obs, self.label = self.generate()
        return self.obs

    def get_label(self):
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
        if action==self.get_label():
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