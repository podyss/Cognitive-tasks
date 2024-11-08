import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import numpy as np
from generate import generate
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.conv import Conv
import sys
sys.path.insert(0, sys.path[0]+"/../../../")
from models import BaseEnv, TensorboardCallback


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
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


class Env(BaseEnv):
    def __init__(self, timelimit=10):
        action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        observation_space = spaces.Dict(
            spaces={
                "image": spaces.Box(-1, 1, [3,128,256], dtype=np.float32),
            },
        )
        super().__init__(action_space,observation_space,timelimit)

    def generate(self):
        img1, text1 = generate()
        img2, text2 = generate()
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
        obs = {"image":img}
        res1 = eval(text1)
        res2 = eval(text2)
        if res1 == res2:
            label = 1
        elif res1 < res2:
            label = 0
        else:
            label = 2
        return obs, label

env = Env()
# model = SAC('MultiInputPolicy', env, tensorboard_log='logs/', buffer_size=int(1e4),
#             policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor))
model = PPO('MultiInputPolicy', env, tensorboard_log='logs/',
            policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor))
# print(model.policy)
model.learn(total_timesteps=10000*10,progress_bar=True,callback=TensorboardCallback())
# model.save('model/ppo.pth')
