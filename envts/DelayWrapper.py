from multiprocessing import Process, Queue, Manager, Value
import time
import numpy as np
import gymnasium as gym
import os
import psutil



class DelayWrapper:
    """
    A wrapper for Gym environments that introduces a fixed delay between
    action submission and execution using multiprocessing for asynchronous control.
    """

    def __init__(self, env: gym.Env, delay_value: int):
        """
        Initialize the DelayWrapper.

        Args:
            env (gym.Env): The Gym environment to wrap.
            delay_value (int): Number of steps of delay between action and execution.
        """
        self.env = env
        self.delay_value = delay_value
        self.action_queue = Queue(maxsize=self.delay_value)
        self.transitions_queue = Queue(maxsize=self.delay_value)
        self.already_closed = False
        self.dt = self.env.dt

        self.action_buffer = []

        manager = Manager()
        self.final_transition_queue = Queue(maxsize=1)
        self.terminated = manager.Value('b', False)
        self.done = manager.Value('b', False)

        self.env_process = None


    def _run_env(self):
        """
        Environment loop running in a separate process.
        Pulls actions from the queue, steps through the environment,
        and pushes transitions to another queue.
        """
        #print("Resetting wrapper environment")
        state, _ = self.env.reset()
        self.transitions_queue.put((state, 0, False, False, {}))

        while not (self.done.value or self.terminated.value):
            try:
                if self.action_queue.full():
                    for _ in range(self.delay_value):
                        action = self.action_queue.get()

                        if action is None:
                            break
                  
                        state, reward, terminated, done, info = self.env.step(action)
                        ##print(state)
                        self.done.value = done
                        self.terminated.value = terminated


                        self.transitions_queue.put((state, reward, terminated, done, info), block=True, timeout=0.01)
                        
                        if done or terminated:
                            #self.transitions_queue.put((state, reward, terminated, done, info))
                            transitions_finales = self.peek_last_items(self.transitions_queue, copy=True)
                            self.final_transition_queue.put(transitions_finales)
                            break

                        #self.transitions_queue.put((state, reward, terminated, done, info))
                        

            except Exception as e:
                #print(f"⚠️ Error in _run_env: {e}")
                pass


        """ for debug
        print("🛠 Final check before closing the child process...")
        if hasattr(self.env, "HW") and self.env.HW is not None:
            print("⚠️ Warning: self.env.HW still exists before `close()` in the child process.")
        else:
            print("⚠️ self.env.HW has already been deleted in the child process.") """


        time.sleep(1)
        self.close()
        


    def step(self, actions, t0=1):
        """
        Enqueue actions to be executed by the environment process.

        Args:
            actions (list): A list of actions to enqueue.
            t0 (int): If 0 (first step), clears the first item from the transitions queue.
        """
        self.action_buffer = actions

        if self.done.value or self.terminated.value:
            return

        if t0 == 0:
            _ = self.transitions_queue.get()

        for action in actions:
            try:
                self.action_queue.put(action, block=True)
            except Exception as e:
                pass
                #print(f"⚠️ Error sending action: {e}")


    def peek_last_items(self, queue, copy=False):
        """
        Retrieve all items from a queue without permanently removing them.

        Args:
            queue (Queue): The queue to inspect.
            copy (bool): Whether to reinsert the items after reading.

        Returns:
            list: The list of items from the queue.
        """
        if not copy:

            items = []
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                    items.append(item)
                except Exception:
                    break  # queue was empty between check and get
            return items  
                
        else:
            temp = []
            temp_queue = Queue()
            while not queue.empty():
                item = queue.get()
                temp.append(item)
                temp_queue.put(item)
            
            while not temp_queue.empty():
                queue.put(temp_queue.get())

            return temp

  

    def observe_wrapper(self, t0=1):
        """
        Retrieve the missed transitions and future actions to be executed.

        Args:
            t0 (int): If 0(reset), fetches from the current transitions queue,
                     otherwise waits until terminal condition or queue is full.

        Returns:
            tuple: Lists of missed states, missed rewards, missed terminations, missed done flags, and future actions.
        """
        missed_states, missed_rewards, missed_terminated, missed_done = [], [], [], []
        future_actions = self.action_buffer

        if t0 == 0 and not self.transitions_queue.empty():
            transitions = self.peek_last_items(self.transitions_queue, copy=True)
            for state, reward, terminated, done, _ in transitions:
                missed_states.append(state)
                missed_rewards.append(reward)
                missed_terminated.append(terminated)
                missed_done.append(done)
            return (missed_states, future_actions), missed_rewards, missed_terminated, missed_done

        if t0 > 0:
            while not (self.done.value or self.terminated.value):
                if self.transitions_queue.qsize() >= self.delay_value:
                    ##print("True")
                    transitions = self.peek_last_items(self.transitions_queue, copy=False)
                    ##print(len(transitions))
                    for state, reward, terminated, done, _ in transitions:
                        missed_states.append(state)
                        missed_rewards.append(reward)
                        missed_terminated.append(terminated)
                        missed_done.append(done)
                    return (missed_states, future_actions), missed_rewards, missed_terminated, missed_done

            if self.done.value or self.terminated.value:
                transitions = self.final_transition_queue.get()
                for state, reward, terminated, done, _ in transitions:
                    missed_states.append(state)
                    missed_rewards.append(reward)
                    missed_terminated.append(terminated)
                    missed_done.append(done)
                return (missed_states, future_actions), missed_rewards, missed_terminated, missed_done

        return (missed_states, future_actions), missed_rewards, missed_terminated, missed_done

                 
    def reset(self):
        """
        Reset the environment and launch a new subprocess for execution.

        Returns:
            tuple: Initial state and dummy reward, terminated, done, info values.
        """
        #print("Resetting DelayWrapper")
        self._drain_queue(self.action_queue)
        self._drain_queue(self.transitions_queue)
        self.terminated.value = False
        self.done.value = False

        self.parent_pid = os.getpid() # Store the parent PID once the process is created
        #print(f"🔍 Main process: {self.parent_pid}")

        if self.env_process and self.env_process.is_alive():
            #print(f"⚠️ Attempting to stop existing process {self.env_process.pid}")
            self.close()

        self.env_process = Process(target=self._run_env)
        self.env_process.start()
        #print(f"✅ New child process started: {self.env_process.pid}, Parent: {self.parent_pid}")

        time.sleep(1)  # sleep to let the process start


        try:
            actual_parent = psutil.Process(self.env_process.pid).ppid()
            #print(f"🧩 Child process {self.env_process.pid}, Actual parent: {actual_parent}, Expected parent: {os.getpid()}")

            if actual_parent != os.getpid():

                #print(f"⚠️ ERROR: Process {self.env_process.pid} has the wrong parent. It will not be joined.")
                self.env_process = None #  Remove reference to avoid invalid `join()`
                return
            
        except psutil.NoSuchProcess:
            pass

            #print(f"⚠️ The child process {self.env_process.pid} has already terminated.")


        for _ in range(30):
            (state,_), _, _, _ = self.observe_wrapper(t0=0)
            if state != [] :
                break
            time.sleep(0.1) # wait a moment and retry

        if state == [] :
            #print("Failed to reset environment")
            return (state,_), 0, False, False
        

        return (state,_), 0, False, False

    def _drain_queue(self, queue):
        """
        Empty all items from a queue.

        Args:
            queue (Queue): The queue to drain.
        """
        try:
            while not queue.empty():
                queue.get_nowait()
        except Exception as e:
            pass
            #print(f"Error draining queue: {e}")

    
    def close(self):
        """
        Properly close the child environment process and cleanup queues.
        """
        if self.already_closed:
            #print("⚠️ Already closed. Ignoring.")
            return
        self.already_closed = True
        
        #print("🔄 Closing child process...")
        try:
            if self.env_process is not None and self.env_process.is_alive():
                #self.action_queue.put(None)  # send stop signal (for the real robot)
                self.env_process.join(timeout=5)
                if self.env_process.is_alive():
                    #print("⚠️ Child process is not responding, attempting `terminate()`...")
                    self.env_process.terminate()
            #print("✅ Child process closed successfully.")

        except Exception as e:
            pass
            #print(f"⚠️ Error closing child process: {e}")

        finally:
            self._drain_queue(self.action_queue)
            self._drain_queue(self.transitions_queue)

            self.action_queue.close()
            self.action_queue.join_thread()
            self.transitions_queue.close()
            self.transitions_queue.join_thread()

            self.env_process = None

            #print("🧹 Final cleanup done.")

    def action_space(self):
        """Return the action space of the wrapped environment."""
        return self.env.action_space

    def observation_space(self):
        """Return the observation space of the wrapped environment."""
        return self.env.observation_space

    def is_done(self):
        """Check whether the current episode is done."""
        return self.done.value

