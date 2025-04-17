"""
Continuous action version of the classic cart-pole system implemented by Rich Sutton et al.
"""
import gymnasium as gym
from gymnasium.envs.registration import register

from gymnasium.envs.classic_control import utils
from gymnasium.utils import seeding
from gymnasium.error import DependencyNotInstalled
from source.models import model_factory
from source.envs import Observer
import torch
import numpy as np
from typing import Any, Callable, List, Optional, Set
from pathlib import Path
import math
import os

from gymnasium import core, spaces



class Fakefuruta(gym.Env):
    def __init__(self,
                 observer: object =Observer, 
                 render_mode: Optional[str] = "rgb_array",                  
                 model_class: str = "model_class",
                 model_path: str = "model.tar",
                 directory: str = "model_directory",
                 **kwargs: dict) -> None:
        


        self.device = "cuda"
        self.observer = observer
        self.model_class = model_class
        self.model_path = model_path
        self.directory = Path(directory)
        self.kwargs = kwargs
        self.dynamics_model = None
        self.action_repeat = 1


        self.min_action = -1.0
        self.max_action = 1.0

        self.force_max = 1.0
        self.tau = 0.01  # seconds between state updates
        self.dt = 0.01


        self.MAX_VEL_1 = 15.0 
        self.MAX_VEL_2 = 30.0 # 5 * np.pi


        high = np.array(
            [np.inf, np.inf, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )

        low = -high
        
        self.observation_space = spaces.Box(shape=(4,), low=-high, high=high)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


        try:
            self.load_dynamics(model_class=self.model_class,
                               model_path=self.model_path,
                               **self.kwargs)
        except FileNotFoundError:
            print(model_class, model_path)
            print("modèle non trouvé")
            pass
        
        self.seed()

        self.state = None



    @property
    def name(self):
        return self.dynamics_model.__class__.__name__


    def load_dynamics(self, model_class, model_path, **kwargs):
        kwargs["action_size"] = 1
        kwargs["state_size"] = 4
        self.dynamics_model = model_factory(model_class, kwargs)

        print(self.directory)
        path = os.path.join(self.directory, model_path)

        self.dynamics_model.load_state_dict(torch.load(path, weights_only=True) )
        print(f"Loaded {model_class} from {path}.")
        self.dynamics_model.eval()
        self.dynamics_model.to(self.device)
        for param in self.dynamics_model.parameters():
            param.requires_grad = False

        return self

    def predict_transition(self, state, action):
        action = torch.repeat_interleave(action, self.action_repeat, dim=0)
        action = action.unsqueeze(0)
        with torch.no_grad():
            new_state = self.dynamics_model.integrate(state, action)
        return new_state[::self.action_repeat, ...].cpu().numpy()




    def seed(self, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):        


        force = min(max(action, -1.0), 1.0) * self.force_max
        u = torch.tensor(np.array([force]), dtype=torch.float32).to(self.device)

        state = torch.tensor(self.state, dtype=torch.float32).to(self.device)
        state = state.expand(1, -1)


        
        
        

      
        self.state = self.predict_transition(state, u)[0][0]

        self.state[0] = wrap_to_pi(self.state[0])
        self.state[1] = wrap_to_pi(self.state[1])

        self.state[2] = bound(self.state[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        self.state[3] = bound(self.state[3], -self.MAX_VEL_2, self.MAX_VEL_2)

        state = self.state #if np.isnan(self.state[0]):
        #    print(state)
        #    print(u)
        #    ab




        #reward = -( 100.0* (1.0  +  np.cos( state[1] ))**2 + 100.0* np.sin( state[1] )**2 + 50.0 * ( np.cos(state[0]) - 1)**2  + 0.5*state[2]**2 + 0.5*state[3]**2 + 1.0 * (action[0]**2))/400 #np.cos(self.state[0]) - np.cos(self.state[1])

        reward = np.exp(-0.5*( 1.0* (1.0  +  np.cos( self.state[1] ))**2 +  np.sin( self.state[1] )**2 + 0.5 * ( np.cos(self.state[0]) - 1)**2  + 0.0005*self.state[2]**2 + 0.00025*self.state[3]**2 + 0.00001 * (action[0]**2)))


        


        return self._get_ob(), reward, False, False, {} # np.array(self.state, dtype=np.float32), reward, False, False, {}

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(self.state, dtype=np.float32)
        #return np.array(
        #    [np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]], dtype=np.float32
        #)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.1, 0.1  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
            np.float32
        )
        return self._get_ob(), {}


def wrap_to_pi(x):

    return np.mod(x + np.pi, 2 * np.pi) - np.pi

def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
        m: The lower bound
        M: The upper bound
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)
    

register(
    id='im_furuta',
    entry_point='envts.fake_furuta:Fakefuruta',
    max_episode_steps=200,
)
