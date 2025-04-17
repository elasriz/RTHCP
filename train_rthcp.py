import multiprocessing
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import trange

from source.utils import seed_all, create_agent, create_env, evaluate, save_data, close_exp, train_imagination

def main():
    """
    Main execution entry point.

    - Initializes multiprocessing with 'spawn' method (recommended on Windows and safe for subprocesses).
    - Starts the training process using the `train()` function.
    - Saves the evaluation rewards as a CSV file at the end.
    """


    multiprocessing.set_start_method("spawn", force=True)
    evaluating_rewards = train(delay = 3,
                            training_timesteps = 200000,
                            directory = "training_Furuta",
                              eval_frequency = 5000,
                                save_frequency = 5000,
                                  model_fit_frequency = 5000,
                                    imagination_frequency = 20000,
                                      resume_episode = 0, 
                                      seed = 1000
                                        )
    save_data(evaluating_rewards, seed = 1000, directory = Path("training_Furuta"), filename="training_data.csv")






def train(delay, training_timesteps, directory, eval_frequency, save_frequency, model_fit_frequency, imagination_frequency, resume_episode=0, seed=48):
    """
    Main training loop for a reinforcement learning agent with delayed actions.

    This function runs training over multiple timesteps, periodically evaluates the agent,
    saves policies, buffers, and trains a dynamics model and an imaginary environment.

    Args:
        delay (int): The fixed delay in the environment between action selection and effect.
        training_timesteps (int): Total number of training steps to execute.
        directory (str): Directory where models, buffers, and logs will be saved.
        eval_frequency (int): Number of steps between each evaluation phase.
        save_frequency (int): Number of steps between model/buffer saving.
        model_fit_frequency (int): Frequency (in steps) at which to refit the dynamics model.
        imagination_frequency (int): Frequency (in steps) to train using imagination rollouts.
        resume_episode (int): Episode number from which to resume training (if loading saved state).
        seed (int): Random seed for reproducibility.

    Returns:
        list: A list containing evaluation rewards (one per evaluation period).
    """

    evaluating_rewards = []
    seed = seed_all(seed)
    episode = resume_episode
    starting_step = 0

    next_model_training = model_fit_frequency
    next_imagination_training = imagination_frequency
    next_evaluation_step = eval_frequency
    next_save_step = save_frequency


    try:
        #Initialization
        #print(".... Initializing the Environment")
        env, observer = create_env(delay)


        #print(".... Initializing the Agent")
        agent = create_agent(env, observer, delay)

        # Set a directory for each episode
        agent.directory = Path(directory) / f"saved_models/"
        agent.episode = episode

        if not os.path.exists(f"./{directory}/saved_policy"):
            os.makedirs(f"./{directory}/saved_policy")

        if not os.path.exists(f"./{directory}/saved_buffers"):
            os.makedirs(f"./{directory}/saved_buffers")

        if not os.path.exists(f"./{directory}/saved_models"):
            os.makedirs(f"./{directory}/saved_models")


        if resume_episode != 0:
            print(".... Loading pretraied model") 
            rl_path = f"./{directory}/saved_models/"
            agent.load_dynamics(model_class=agent.model_class, model_path = rl_path + f"{agent.model_class}_episode{resume_episode}_seed{seed}")

            print(".... Loading pretraied policy") 
            rl_path = f"./{directory}/saved_policy/"
            agent.rl.load(rl_path + f"episode{resume_episode}_seed{seed}")

            print(".... Loading saved Buffer")
            rb_path = f"./{directory}/saved_buffers/"
            agent.rl.load_replay_buffer(rb_path + f"episode{resume_episode}_seed{seed}") 
            starting_step = len(agent.rl.replay_buffer)
            print(starting_step)


        # Set the agent
        agent.test = False
        agent.model_fit_frequency = model_fit_frequency
        observer.reset()
        agent.reset()

        done, terminated, future_actions = False, False, [[0.0] for _ in range(delay)]

        (missed_states,_), _, _, _ = env.reset()
        # wait for reset
        while missed_states == [] :
            time.sleep(1)
            (missed_states,_), _, _, _ = env.reset()

        _ = agent.act(missed_states[-1], future_actions)


        t = 0
        ep_reward = 0
        env.step(future_actions, t0=t) #this first step to clean the transistions buffer and fill the action buffer
        new_future_actions = future_actions

        for timestep in trange(starting_step,training_timesteps+1):
            t+=1
            # compute next action sequence
            actions_sequence = agent.act(missed_states[-1], new_future_actions)

            # send the action sequence to the wrapper
            env.step(actions_sequence, t0=t)
            # observe the wrapper 
            augmented_state, new_missed_rewards, new_missed_terminated, new_missed_done = env.observe_wrapper(t0=t)
            new_missed_states,new_future_actions = augmented_state

            

            ep_reward += sum(new_missed_rewards)
            terminated, done = any(new_missed_terminated), any(new_missed_done)


            if len(new_missed_states) > 1 or done or terminated: #do not store the first step (it is just a reset step)
                for i in range(len(new_missed_states)-1):
                    observer.log(env, new_missed_states[i], np.array(future_actions[i+1]))
                observer.log(env, new_missed_states[-1], np.array(new_future_actions[0]))



                agent.rl.add_to_buffer(missed_states[-1], np.array(future_actions[0]), new_missed_rewards[0], new_missed_states[0], new_missed_terminated[0], new_missed_done[0])
                for i in range(len(new_missed_states)-1):
                    agent.rl.add_to_buffer(new_missed_states[i], np.array(future_actions[i+1]), new_missed_rewards[i+1], new_missed_states[i+1], new_missed_terminated[i+1], new_missed_done[i+1])


            missed_states, future_actions = new_missed_states, new_future_actions


            if done or terminated:

                close_exp(env, 10)
                episode += 1
                print(f"Training Episode : {episode}. Episode lenght : {t}. timesteps : {timestep}. Return : {ep_reward}.")
                
                

                if timestep >= next_model_training  :

                    #print("training the model")
                    agent.fit_dynamics()
                    next_model_training += model_fit_frequency
                    agent.episode = episode
                    time.sleep(20)


                if timestep >= next_imagination_training:

                    #print("optimise policy with imagined trajectories ..................")
                    
                    agent.rl = train_imagination(timestep, agent.model_class, agent, agent.model_path.format(agent.model_class, agent.episode-1, agent.seed), agent.directory)
                    next_imagination_training += imagination_frequency
                    time.sleep(20)


                # Evaluate the agent's performance
                if timestep >= next_evaluation_step:
    
                    #print(" Evaluation ..................") 
                    agent.test = True
                    eval_ep_reward = evaluate(agent, delay, eval_episodes=10)    
                    evaluating_rewards.append(eval_ep_reward)
                    agent.test = False

                    next_evaluation_step += eval_frequency

                    time.sleep(20)


                if timestep >= next_save_step:                 
                    
                    #print("saving policy ..................")
                    agent.rl.save(f"./{str(directory)}/saved_policy/episode{episode}_seed{seed}")
                    agent.rl.save_replay_buffer(f"./{str(directory)}/saved_buffers/episode{episode}_seed{seed}")

                    next_save_step += save_frequency
                    time.sleep(20)


                # renitialize for next training episode
                env, observer = create_env(delay)

                agent.reset()
                observer.reset()

                done, terminated, future_actions = False, False, [[0.0] for _ in range(delay)]

                augmented_state, _, _, _ = env.reset()
                missed_states, _ = augmented_state
                while missed_states == [] :
                    time.sleep(1)
                    augmented_state, _, _, _ = env.reset()  
                    missed_states, _ = augmented_state        

                _ = agent.act(missed_states[-1], new_future_actions)    

                t = 0
                ep_reward = 0
                env.step(future_actions, t0=t) #this first step to clean the transistions buffer and fill the action buffer
                new_future_actions = future_actions


        return  evaluating_rewards
        

    except KeyboardInterrupt:
        print(f"🚨 Training session interrupted at episode {episode}!")
        return  evaluating_rewards

    except FileNotFoundError as e:
            print(f"⚠️ Missing file encountered: {e}. Skipping episode {episode}.")

    print(f"Training process completed after {episode+1} episodes!")

    return  evaluating_rewards




if __name__ == "__main__":
    main()
