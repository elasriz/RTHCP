import copy
from pathlib import Path
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt

from source.envs import Observer
from source.models import model_factory
from source.random_policy import RandomPolicy
from source.training import Trainer
from source.TD3 import TD3, ExplorationNoise
from source.envs import env_factory

import os

logger = logging.getLogger(__name__)


class RTHCP(object):
    """
    Receding Horizon Trajectory-aware Hybrid Control with Planning (RTHCP).
    under inference delays

    This class combines model-based planning using CEM (Cross Entropy Method)
    with a learned policy (TD3) to compute optimal action sequences under model predictions.
    It optionally blends actions from the policy and noise.

    Attributes:
        env (object): Gym-compatible environment.
        observer (Observer): Observer used for normalizing state inputs.
        dynamics_model (nn.Module): Dynamics model used to simulate future trajectories.
        rl (TD3): The reinforcement learning policy.
        horizon (int): Planning horizon.
        receding_horizon (int): Number of actions to return in act() (number of delayed steps).
        action_repeat (int): Repeats each planned action this many times.
        population (int): Number of samples in CEM optimization.
        pi_population (int): Number of samples taken from the RL policy.
        selection (int): Number of top-performing trajectories selected in CEM.
        iterations (int): Number of CEM refinement iterations.
        use_Q (bool): Whether to use the critic for bootstrapping at the end of rollout.
        alpha (float): Weight for critic-based Q-value.
        model_class (str): Name of the model class to load.
        model_path (str): Template string for loading dynamics models.
        directory (Path): Directory to store/load models.
        gamma (float): Discount factor.
        test (bool): Whether the agent is in evaluation mode (no learning).
    """
    def __init__(self,
                 env: object,
                 observer: Observer, 
                 receding_horizon: int = 3,     
                 horizon: int = 4,
                 seed: int = 3,
                 action_repeat: int = 1,
                 population: int = 300,
                 pi_population: int = 20,
                 selection: int = 1,
                 iterations: int = 2,
                 device: str = "cuda",
                 model_class: str = "RTHCP_Furuta",
                 model_path: str = "{}_model_episode{}_seed{}.tar",
                 directory: str = "out",
                 model_fit_frequency: int = 5000,
                 test: bool = False,
                 use_Q: bool = True,
                 alpha: float = 1,
                 **kwargs: dict) -> None:
        


        self.pi_population = pi_population 
        self.use_Q = use_Q
        self.alpha = alpha


        self.test = test 
        self.seed = seed
        self.episode = None
        
        self.last_action = None

        self.env = env


        self.mbrl_path = "model_learning2/episode_15/models/{}_model_{}.tar"
        self.mbrl_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

        self.observer = observer
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.receding_horizon = receding_horizon
        self.population = population

        self.selection = selection
        self.iterations = iterations
 


        self.device = device
        self.model_class = model_class
        self.model_path = model_path
        self.directory = Path(directory)
        self.model_fit_frequency = model_fit_frequency
        
        self.kwargs = kwargs
        self.dynamics_model = None
        self.trainer = None
        self.planned_actions = []
        self.history = []
        self.steps = 0
        self.gamma = 0.99
   


  

        state_dim = 4 #self.env.observation_space.shape[0]
        action_dim = 1 #1     

        self.rl = TD3(state_dim=4, action_dim=1, actor_hidden_dim=[400,300], critic_hidden_dim=[400,300], tau=0.005, policy_noise=0.2)
        self.rl.exploration_noise = ExplorationNoise(action_dim=action_dim)  




        try:
            self.load_dynamics(model_class=self.model_class,
                           model_path=self.mbrl_path,
                           **self.kwargs)
        except FileNotFoundError:
            #print("file not found")
            pass
          
        self.exploration_policy = RandomPolicy(env, **kwargs)

    @property
    def name(self):
        return self.dynamics_model.__class__.__name__

    def load_dynamics(self, model_class, model_path, **kwargs):
        """
        Load the dynamics model from disk and move it to the correct device.

        Args:
            model_class (str): Name of the model class to load.
            model_path (str): Path format string to load model from.
            **kwargs: Additional arguments passed to model constructor.

        Returns:
            self: Returns the current object (for chaining).
        """        
        kwargs["action_size"] = 1 # 1
        kwargs["state_size"] = 4 #self.env.observation_space.shape[0]
        self.dynamics_model = model_factory(model_class, kwargs)
        try:
            path = self.directory / model_path.format(self.dynamics_model.__class__.__name__, self.episode, self.seed)
            self.dynamics_model.load_state_dict(torch.load(path, weights_only=True))
            #print(f"Loaded {model_class} from {path}.")
        except:
            #print("Loading model failed, no model saved found")    
            pass        

        if self.test:

            self.dynamics_model.eval()

        self.dynamics_model.to(self.device)
        return self

    def fit_dynamics(self):
        """
        Reinitialize and train the dynamics model using collected data.

        This will reload the model structure, train it using the observer data,
        and save it to disk.
        """        
        # Reset initial model and optimizer
        self.load_dynamics(model_class=self.model_class,
                           model_path=self.model_path,
                           **self.kwargs)
        self.trainer = Trainer(self.dynamics_model,
                               device=self.device,
                               directory=self.directory,
                               model_path=self.model_path,
                               seed=self.seed,
                               **self.kwargs)
        # Train
        self.trainer.train(df=self.observer.dataframe())
        self.trainer.save_model(self.episode)



    def reward_model(self, states, actions, gamma=None):
        """
        Compute reward for a batch of trajectories, optionally bootstrapping with critic Q-values.

        Args:
            states (Tensor): Predicted states [time x batch x state_dim].
            actions (Tensor): Actions taken [time x batch x action_dim].
            gamma (float, optional): Discount factor.

        Returns:
            Tensor: Reward per trajectory step [time x batch].
        """

        done = (
    	    torch.cos( states[..., 0] ) < -0.3
    	    )

        costs = 0.5* (1.0  +  torch.cos( states[..., 1] ))**2 + 1.0* torch.sin( states[..., 1] )**2 + 0.5 * ( torch.cos(states[..., 0]) - 1)**2  + 0.00005*states[..., 2]**2 + 0.000025*states[..., 3]**2 + 0.00001 * (actions[..., 0]**2)


        rewards = done.logical_not().float().clone().detach() * torch.exp(- costs)

        if gamma:
            time = torch.arange(rewards.shape[0], dtype=torch.float32, device=self.device).unsqueeze(-1).expand(rewards.shape)
            rewards *= torch.pow(gamma, time)
            
        if self.use_Q:
            
            states_new = states[-1].to(dtype=torch.float32, device=self.device)               
            #action_ = torch.tensor(self.rl.actor(states_new)).to(dtype=torch.float32, device=self.device)
            action_ = self.rl.actor(states_new).clone().detach().to(dtype=torch.float32, device=self.device)

            q1 = self.rl.critic1(states_new, action_)
            q2 = self.rl.critic2(states_new, action_)
            Q_value = self.alpha * (gamma**self.horizon) * torch.min(q1,q2).reshape(1,self.population + self.pi_population)

            rewards = torch.cat([rewards, Q_value], dim=0)

        return rewards


    def predict_trajectory(self, state, actions):
        """
        Roll out trajectories by applying a sequence of actions using the dynamics model.

        Args:
            state (Tensor): Initial state [batch x state_dim].
            actions (Tensor): Sequence of actions [horizon x batch x action_dim].

        Returns:
            Tensor: Predicted state trajectory [horizon x batch x state_dim].
        """

        
        actions = torch.repeat_interleave(actions, self.action_repeat, dim=0)
        states = self.dynamics_model.integrate(state, actions)
        return states[::self.action_repeat, ...]


    
    def plan(self, state, future_actions):
        """
        Cross Entropy Method (CEM) planning with initial rollout of provided future actions.

        Args:
            state (np.ndarray): The current observed state.
            future_actions (list or np.ndarray, optional): Sequence of actions applied before planning.

        Returns:
            dict: containing planned trajectories: states, actions, and time.
        """


        #action_space = self.env.action_space
        action_mean = torch.zeros(self.horizon, 1, 1, device=self.device)
        action_std = torch.ones(self.horizon, 1, 1, device=self.device) * 1.0


        state = torch.tensor(self.observer.observe_array(self.env, state), dtype=torch.float32).to(self.device)
        state = state.unsqueeze(0)

        # Apply future actions to compute the updated state
        future_actions_tensor = torch.tensor(future_actions, dtype=torch.float32, device=self.device).view(len(future_actions), 1, -1)
        state = self.dynamics_model.integrate(state, future_actions_tensor)[-1]  # Final state after future actions

        
        
        
        # Duplicate states for population planning
        state_pi = state.clone()
        state = state.expand(self.population + self.pi_population, -1) 
        
        # 1. Draw sample sequences of actions from RL policy 
        pi_actions = torch.empty(self.horizon, self.pi_population, 1, device=self.device)
        
        z = state_pi.repeat(self.pi_population, 1)
      

        with torch.no_grad():

            for t in range(self.horizon):

                pi_actions[t] = self.rl.sample_pi_actions(z) #torch.from_numpy(self.rl.select_action(z))
                z = self.dynamics_model.integrate(z, pi_actions[t].view(1,self.pi_population,1))[0]


            # 1.bis Draw sample sequences of actions from a normal distribution
            actions = action_mean + action_std * torch.randn(self.horizon, self.population, 1, device=self.device)
            actions = torch.clamp(actions, min=-1.0, max=1.0)

            actions = torch.cat([pi_actions, actions], dim=1)


            for _ in range(self.iterations):
                
                # 2. Unroll trajectories

                states = self.predict_trajectory(state, actions)
                # 3. Fit the distribution to the top-k performing sequences
                returns = self.reward_model(states, actions,self.gamma).sum(dim=0)
                _, best = returns.topk(self.selection, largest=True, sorted=False)
                states = states[:-1, :, :]  # Remove last predicted state, for which we have no action
                best_actions = actions[:, best, :]
                action_mean = best_actions.mean(dim=1, keepdim=True)
                action_std = best_actions.std(dim=1, unbiased=False, keepdim=True)
                actions = action_mean + action_std * torch.randn(self.horizon, self.population + self.pi_population, 1, device=self.device)
                actions = torch.clamp(actions, min=-1.0, max=1.0)

        times = self.observer.time + np.arange(states.shape[0]) * self.observer.dt(self.env) * self.action_repeat

        return {
            "states": states[:, best, :].cpu().numpy(),
            "actions": best_actions.cpu().numpy(),
            "time": times
        }





    def step(self):
        """
        Update the internal policy if not in test mode.

        - Advances the random exploration policy.
        - Performs a training step on the RL agent.
        """        
        if not self.test:
            self.exploration_policy.step()
            self.rl.update()
            


    def reset(self):
        """
        Reset internal planning actions and trajectory history.

        Called at the beginning of a new episode.
        """        
        self.planned_actions = []
        self.history = []

        
    def act(self, state, future_actions):
        """
        Selects a sequence of actions based on current observation and future action context.

        Args:
            state (np.ndarray): The current environment state.
            future_actions (list or np.ndarray, optional): Future actions to be applied first.

        Returns:
            list: A list of planned actions.
        """        
        self.step()
        trajectories = self.plan(state, future_actions)
        self.planned_actions = trajectories["actions"].mean(axis=1)
        self.planned_actions = np.repeat(self.planned_actions, self.action_repeat, axis=0)[:, :].tolist()
        self.planned_actions = self.planned_actions[:self.receding_horizon]
        best_actions = self.planned_actions
        
        return best_actions

        