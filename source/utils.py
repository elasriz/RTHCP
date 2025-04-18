
import numpy as np
import torch
import random
from source.envs import env_factory
from source.agents import agent_factory
import pandas as pd
from source.TD3 import TD3, ExplorationNoise
from tqdm import trange
import time
import copy




def seed_all(seed=None):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int, optional): The seed value. If None, a random seed is chosen.

    Returns:
        int: The seed used.
    """    
    if seed is None:
        seed = np.random.randint(2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed



def create_agent(env, observer, delay):
    """
    Initializes the agent for training.

    Args:
        env (gym.Env): The environment instance.
        observer (object): The observer object.
        delay (int): Delay value to be applied in the environment.

    Returns:
        object: The instantiated reinforcement learning agent.
    """
    
    agent = agent_factory(env, observer, agent_class="RTHCP", delay=delay)
    agent.rl = TD3(state_dim=4, action_dim=1, actor_hidden_dim=[400,300], critic_hidden_dim=[400,300], tau=0.005, policy_noise=0.2)
    agent.rl.exploration_noise = ExplorationNoise(action_dim=1)
    return agent


def create_env(env_name, delay):
    """
    Initializes the environment for training.

    Args:
        delay (int): Delay value to be applied in the environment.

    Returns:
        tuple: The initialized environment instance and observer.
    """
    env, observer = env_factory(env_name, frequency=50, delay=delay)

    return env, observer



def close_exp(env, timesleep):
    """
    Ensures proper closure of the environment and manages wait times between experiments.

    Args:
        env (gym.Env): The environment instance to be closed.
        timesleep (int): Waiting time (in seconds) before the next execution cycle.
    """
    try:

        # Avoid redundant closure attempts
        if hasattr(env, "already_closed") and env.already_closed:
            print("Environment has already been closed. Skipping redundant closure.")
        else:
            env.close()
        
        time.sleep(timesleep)  # Extended wait time to ensure proper hardware release

    except Exception as e:
        print(f"Error encountered during environment closure: {e}")


def evaluate(agent, delay, eval_episodes=10):
    """
    Evaluates the performance of the agent by executing multiple trials.

    Args:
        agent (object): The agent to evaluate.
        delay (int): Delay factor in the action execution.
        eval_episodes (int): Number of evaluation trials.

    Returns:
        float: Mean reward obtained across all evaluation trials.
    """
    
    episode_rewards = []
    episode = 1

    for _ in range(eval_episodes):

        # Reinitialize environment and agent for each trial to ensure independent evaluation
        
        env, _ = create_env(delay)
        agent.test = True

        # Reset the environment before starting the trial
        augmented_state, _, _, _ = env.reset()
        # wait for reset
        missed_states, _ = augmented_state
        while missed_states == [] :
            time.sleep(1)
            augmented_state, _, _, _ = env.reset()  
            missed_states, _ = augmented_state    

        done, terminated, future_actions = False, False, [[0.0] for _ in range(delay)]
        _ = agent.act(missed_states[-1], future_actions)

        t = 0
        ep_reward = 0
        env.step(future_actions, t0=t) #this first step to clean the transistions buffer and fill the action buffer
        new_future_actions = future_actions

        # Execute the trial until termination criteria are met
        while not (done or terminated):

            t+=1
            actions_sequence = agent.act(missed_states[-1], new_future_actions)
            env.step(actions_sequence, t0=t)
            augmented_state, new_missed_rewards, new_missed_terminated, new_missed_done = env.observe_wrapper(t0=t)
            new_missed_states, new_future_actions = augmented_state
            
            terminated, done = any(new_missed_terminated), any(new_missed_done)
            missed_states, future_actions = new_missed_states, new_future_actions
            ep_reward += sum(new_missed_rewards)

        episode += 1
        episode_rewards.append(ep_reward)
        print(f"Evaluation Episode : {episode}. Episode lenght : {t}. Return : {ep_reward}.")
        close_exp(env, 10)  # reseet the environement 
    return np.nanmean(episode_rewards)


def train_imagination(timesteps, model_class, agent, model_path, model_dir):
    """
    Trains the agent using imagined trajectories in a learned model environment.

    Args:
        timesteps (int): Number of training steps.
        model_class (str): The dynamics model class name.
        agent (object): The agent with policy and buffer.
        model_path (str): Path to the model file.
        model_dir (str): Directory of the model.

    Returns:
        object: Updated agent.rl instance after imagination training.
    """    

    im_env, _ = env_factory("im_furuta", model_class = model_class, model_path = model_path, dir = model_dir)

    replay_buffer_backup = copy.deepcopy(agent.rl.replay_buffer)

    done, terminated = False, False
    state, _ = im_env.reset()

    for timestep in trange(1,timesteps+1):

        agent.rl.update()
        action = agent.rl.select_action_pi(state)

        next_state, reward, terminated, done, _ = im_env.step(action)

        agent.rl.add_to_buffer(state, action, reward, next_state, terminated, done)

        state = next_state

        if done or terminated:
            state, _ = im_env.reset()
            done, terminated = False, False

    agent.rl.replay_buffer = replay_buffer_backup

    return agent.rl




def save_data(evaluating_rewards, seed, directory, filename="training_data.csv"):

    """
    Save evaluation results to a CSV file.

    Args:
        evaluating_rewards (list): List of evaluation rewards.
        seed (int): Random seed used during training.
        directory (Path): Directory to save the data.
        filename (str): Name of the CSV file.

    Returns:
        list: The original evaluating_rewards list.
    """

    training_results = [{
        "agent": "RTHCP",
        "episode": episode,
        "seed": seed,
        "total_reward": evaluating_rewards,
    } for episode, evaluating_rewards in enumerate(evaluating_rewards)]
    training_results = pd.DataFrame.from_records(training_results)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, 'a') as f:
        training_results.to_csv(f, sep=',', encoding='utf-8', header=f.tell() == 0, index=False)

    return evaluating_rewards

