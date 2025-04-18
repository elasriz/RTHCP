from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import gymnasium as gym


random_state = np.random.RandomState()


def env_factory(env_name: str, frequency= None, delay: int = 0, model_class = None, model_path=None, dir="out") -> Tuple:
    
    if env_name == "real_furuta":
        from envts.real_furuta import RealFurutaEnv
        env = RealFurutaEnv(frequency = frequency)
        observer = FurutaObserver()

    elif env_name == "delay_real_furuta":
        from envts.real_furuta import RealFurutaEnv
        env1 = RealFurutaEnv(frequency = frequency)
        from envts.DelayWrapper import DelayWrapper
        env = DelayWrapper(env1, delay)
        observer = FurutaObserver()

    elif env_name == "furuta":
        import envts.furuta
        env = gym.make("furuta")        
        observer = FurutaObserver()  

    elif env_name == "delay_furuta":
        import envts.furuta
        env1 = gym.make("furuta", frequency = frequency)         
        from envts.DelayWrapper import DelayWrapper
        env = DelayWrapper(env1, delay)
        observer = FurutaObserver()

    elif env_name == "im_furuta":
        import envts.fake_furuta        
        env = gym.make("im_furuta", model_class=model_class, model_path=model_path,  directory=dir)        
        observer = FurutaObserver()





    else:
        raise ValueError(f"Unknown env {env_name}")

    return env, observer


class Observer(object):
    def __init__(self):
        self.episodes = [[]]
        self.time = 0

    def reset(self):
        self.time = 0
        self.episodes.append([])

    def dt(self, env):
        return env.dt

    def log(self, env, state, action, reward=None, horloge=None):
        datapoint = self.observe(env, state)
        datapoint.update((f"action_{i}", action[i]) for i in range(action.size))
        if reward:
            datapoint.update((f"reward_{i}", reward[i]) for i in range(reward.size))
        if horloge:
            datapoint.update([(f"horloge", horloge)])
        datapoint[f"time"] = self.time
        self.time += self.dt(env)
        self.episodes[-1].append(datapoint)

    def dataframe(self):
        return pd.DataFrame(sum(self.episodes, []))


    def save(self, path="out/data/dataset.csv"):
        print(f"Save data to {path}")
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        self.dataframe().to_csv(path)

    def append_to(self, path="out/data/dataset.csv"):
        try:
            df_load = pd.read_csv(path)
            print(f"Append data to {path}")
            df = pd.concat([df_load, self.dataframe()])
            df.to_csv(path)
        except FileNotFoundError:
            self.save(path)

    def observe(self, env, state=None):
        raise NotImplementedError()

    def observe_array(self, env, state=None):
        return np.array(list(self.observe(env, state).values()))




class FurutaObserver(Observer):
    def observe(self, env, state=None):

        if state is None:
            state = env.unwrapped.state
            #state = env.reset()

        return OrderedDict([
            ("state_angle_1", state[0]),
            ("state_angle_2", state[1]),
            ("state_angular_vel_1", state[2]),
            ("state_angular_vel_2", state[3])
        ])
        
  
