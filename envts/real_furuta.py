from quanser.hardware import HIL, HILError
from gymnasium import Env, spaces
import numpy as np
from math import pi
from time import time_ns, sleep, perf_counter
import sys
from loop_rate_limiters import RateLimiter
from typing import Any, Callable, List, Optional, Set

from numpy import cos, pi, sin, exp
import time
class RealFurutaEnv():
    """
    Environment class for controlling the Quanser Qube-Servo 2 (Furuta pendulum)
    using the Quanser HIL interface in real-time.

    This class provides a Gym-compatible API and manages low-level hardware interaction,
    including encoder reading, motor actuation, velocity filtering, and safe reset/close operations.
    """

    def __init__(self, frequency=None):
        """
        Initialize the real Furuta environment with the given frequency.

        Args:
            frequency (float): Desired control frequency (e.g., 50 or 100 Hz).
        """        
        #super().__init__()

        # Create a log file to store states
        old_stdout = sys.stdout
        #log_file = open("./debug_furuta.log","w")
        #sys.stdout = log_file


        # Initialize hardware interface
        # Initialization of the class
        self.HW = None
        self.hw_initialized = False
        self.already_closed = False

        self.state = None

        #self.rate = RateLimiter(frequency=100.0)
        self.alpha = None
        self.alpha_dot = None
        self.beta = None
        self.beta_dot = None

        self.alpha_last = None
        self.alpha_dot_last = None
        self.beta_last = None
        self.beta_dot_last = None

        self.alpha_last_last = self.alpha_last
        self.alpha_dot_last_last = self.alpha_dot_last
        self.beta_last_last = self.beta_last
        self.beta_dot_last_last = self.beta_dot_last




        self.enc0_offset, self.enc1_offset = None, None

        self.initial_time = None


        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        
        # Time step duration
        self.episode_duration = 12.0  # Episode duration in seconds
        self.episode_lenght = None
        self.remaining = None
        self.__frequency = frequency #100.0


        self.enc_coeff = 2 * pi / 2048

        
        if self.__frequency == 100.0:
            # pendulum angular velocity filter coefficients  fo frequency = 100 Hz dt = 0.01 s
            self.b_beta = [-48.7804878, 48.7804878]
            self.a_beta = [0.2195122, -0.73170732]

            # rotor angular velocity filter coefficients
            self.b_alpha = [-81.4479638, 81.4479638]
            self.a_alpha = [0.13122172, -0.31674208]
        

        
        if self.__frequency == 50.0:# pendulum angular velocity filter coefficients fo frequency = 50 Hz dt = 0.02 s
            self.b_beta = [-55.55555556, 55.55555556]
            self.a_beta = [0.11111111, 0.0]

            # rotor angular velocity filter coefficients
            self.b_alpha = [-79.6460177, 79.6460177]
            self.a_alpha = [0.15044248, 0.44247788] 
        


        # Voltage limits
        self.V_max = 4.0

    @property
    def dt(self) -> Optional[float]:
        """!
        Regulated period of the control loop in seconds, or `None` if there
        is no loop frequency regulation.
        """
        return 1.0 / self.__frequency if self.__frequency is not None else None

    @property
    def frequency(self) -> Optional[float]:
        """!
        Regulated frequency of the control loop in Hz, or `None` if there is
        no loop frequency regulation.
        """
        return self.__frequency
    
    # Write the analog output for motor voltage
    def cmd_voltage(self, voltage):
        """
        Apply a specified voltage to the motor.

        Args:
            voltage (float): Voltage to be applied in the range [-V_max, V_max].
        """        
        self.ao_buffer[0] = voltage
        try:
            self.HW.write_analog(self.ao_channels, len(self.ao_channels), self.ao_buffer)
        except HILError as h:
            print(h.get_error_message())


    # Read all encoder values
    def read_encoders(self):
        """
        Read the encoder values from the hardware.

        Returns:
            encoders (np.ndarray): Array containing raw encoder counts.
        """        
        try:
            self.HW.read_encoder(self.enc_channels, len(self.enc_channels), self.enc_buffer)
        except HILError as h:
            print(h.get_error_message()) 
        return self.enc_buffer
    

    # Set LED control buffers
    def set_LEDs(self, colour = [1, 0, 0]):
        """
        Set the status of the onboard LEDs.

        Args:
            colour (list): RGB values as a list of 3 floats (e.g., [0, 1, 0] for green).
        """
        LED_buffer = np.array(colour, dtype=np.float64)
        try:
            self.HW.write_other(self.LED_channels, len(self.LED_channels), LED_buffer)
        except HILError as h:
            print(h.get_error_message())


    def reset(self):

        """
        Initializes and connects to the hardware, sets up encoders, LEDs, and internal state.

        Returns:
            observation (np.ndarray): Initial state of the system.
            info (dict): Dictionary.
        """

        print("🔄 Initializing hardware...")

        self.HW = None
        self.hw_initialized = False
        self.already_closed = False  # 🔥 Prevents multiple close calls

        for attempt in range(3):  # 🔄 Attempts to reconnect to hardware
            try:
                self.HW = HIL()
                print(f"✅ Hardware inistialized : {self.HW}")

                boardIdentifier = "0"
                cardType = "qube_servo2_usb"
                self.HW.open(cardType, boardIdentifier)

                # ✅ Check after initialization
                print(f"🛠 Check after opening: {self.HW}")

                self.hw_initialized = True
                break
                #return self._get_ob(), {}  # 🔥 Return early if initialization succeeded

            except HILError as h:
                print(f"❌ Error while opening the hardware: {h.get_error_message()}")
                self.HW = None  # 🔥 Prevents inconsistent hardware state
                time.sleep(5)  # 🔥 Wait to avoid hardware conflict

        
        if self.hw_initialized == False:
        
            print("⚠️ Failed to initialize hardware after several attempts.")
            return None  # 🚨 Block reset if hardware is not responding


        # Configure channels
        self.ao_channels = np.array([0], dtype=np.int32)
        self.ao_buffer = np.zeros(len(self.ao_channels), dtype=np.float64)

        # Configure encoder ch annels
        self.enc_channels = np.array([0, 1], dtype=np.int32)
        self.enc_buffer = np.zeros(len(self.enc_channels), dtype=np.int32)

        

        # Configure LED channels
        self.LED_channels = np.array([11000, 11001, 11002], dtype=np.int32)
        self.LED_buffer = np.array([1,0,0], dtype=np.float64)



        self.set_LEDs([0, 0, 1])


        self.next_start_time = time_ns()
        self.initial_time = None
        #self.initial_time = self.next_start_time
        self.episode_lenght = self.episode_duration * self.__frequency
        self.remaining = self.episode_lenght



        # Enable motor amplifiers
        self.HW.write_digital(np.array([0], dtype=np.int32), 1, np.array([1], dtype=np.int8))
        
        
        
        # Reset encoder offsets

        [self.enc0_offset, self.enc1_offset] = self.read_encoders()
        t0 = time_ns()
        
        alpha = -self.enc_coeff * ( self.enc0_offset)
        beta = self.enc_coeff * (self.enc1_offset)

        # Angular velocities initialized to zero
        alpha_dot, beta_dot = 0.0, 0.0


        self.state = np.array([alpha, beta, alpha_dot, beta_dot], dtype=np.float32)

        self.alpha = alpha
        self.alpha_dot = alpha_dot
        self.beta = beta
        self.beta_dot = beta_dot

        self.alpha_last = alpha
        self.alpha_dot_last = alpha_dot
        self.beta_last = beta
        self.beta_dot_last = beta_dot

        self.alpha_last_last = self.alpha_last
        self.alpha_dot_last_last = self.alpha_dot_last
        self.beta_last_last = self.beta_last
        self.beta_dot_last_last = self.beta_dot_last

        return self._get_ob(), {}



    def step(self, a):
        """
        Apply an action to the environment and step forward by one time step.

        Args:
            a (np.ndarray): Action to apply (motor voltage normalized in [-1, 1]).

        Returns:
            observation (np.ndarray): The resulting state.
            reward (float): The computed reward.
            terminated (bool): Whether the episode has reached a terminal state.
            truncated (bool): Whether the maximum episode duration was reached.
            info (dict): Additional information (e.g., time in ms if needed).
        """
        if not self.initial_time:
            self.initial_time = time_ns()
            self.next_start_time  = self.initial_time + self.dt * 1e9

        action = np.clip(self.V_max * a[0], -self.V_max, self.V_max)
        self.cmd_voltage(action)
        self.next_start_time  += self.dt * 1e9


        in_step = True


        while in_step:
            try:
                # Wait for next loop start time
                if time_ns() > self.next_start_time:

                    [enc0, enc1] = self.read_encoders()
                    t0 = time_ns()


                    self.alpha_dot_last_last = self.alpha_dot_last
                    self.alpha_dot_last = self.alpha_dot

                    self.alpha_last_last = self.alpha_last
                    self.alpha_last = self.alpha

                    self.alpha = -self.enc_coeff * (enc0 - self.enc0_offset)
                    self.alpha_dot = -self.a_alpha[1]*self.alpha_dot_last - self.a_alpha[0]*self.alpha_dot_last_last + self.b_alpha[1]*self.alpha_last + self.b_alpha[0]*self.alpha_last_last


                    self.beta = self.enc_coeff * (enc1-self.enc1_offset)


                    self.beta_dot_last_last = self.beta_dot_last
                    self.beta_dot_last = self.beta_dot

                    self.beta_last_last = self.beta_last
                    self.beta_last = self.beta


                    self.beta_dot = -self.a_beta[1]*self.beta_dot_last - self.a_beta[0]*self.beta_dot_last_last + self.b_beta[1]*self.beta_last + self.b_beta[0]*self.beta_last_last

                    self.state = np.array([self.alpha, self.beta, self.alpha_dot, self.beta_dot])



                    # Define reward and termination logic (customize based on your task)
                    terminated = self._terminal()
                    reward = np.exp(-0.5*( 1.0* (1.0  +  np.cos( self.state[1] ))**2 +  np.sin( self.state[1] )**2 + 0.5 * ( np.cos(self.state[0]) - 1)**2  + 0.0005*self.state[2]**2 + 0.00025*self.state[3]**2 + 0.00001 * (a[0]**2)))

                    


                    self.remaining -= 1
                    truncated = bool(self.remaining == 0)
                    in_step = False


            finally:
                self.cmd_voltage(0.) # set voltage to 0 in case of a problem
                self.close()
                break        



        return self._get_ob(), reward.item(), terminated, truncated,  {}
    
    def _get_ob(self):
        """
        Retrieve the current observation/state of the environment.

        Returns:
            state (np.ndarray): Current state as [alpha, beta, alpha_dot, beta_dot].
        """        
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(self.state, dtype=np.float32)

    
    def _terminal(self):
        """
        Check if the pendulum is in a terminal (unsafe) state.

        Returns:
            bool: True if the episode should terminate.
        """        
        s = self.state

        return bool(np.cos(s[0]) < -0.4)
    
    def close(self):
        """
        Cleanly disconnect from the hardware, stop the motor, and release resources.
        """
        print("🔄 Closing the real environment...")

        if self.already_closed:
            print(" ⚠️ Already closed. Ignored.")
            return
        self.already_closed = True  # ✅ Mark the environment as closed

        if not self.hw_initialized or self.HW is None:
            print("⚠️ Hardware not initialized or already closed. Ignoring shutdown.")
            return  # 🔥 Avoid closing more than once

        print("🛠 Checking HW before closing...")

        try:

            if self.HW is not None:
                print("🛠️ Sending stop signal to hardware...")
                self.HW.write_digital(np.array([0], dtype=np.int32), 1, np.array([0], dtype=np.int8))
                self.set_LEDs(colour = [0, 1, 0])                
                print("🛠️ Closing the connection to hardware...")
                self.HW.close()
                self.HW = None  # 🔥 Explicit memory release

            self.hw_initialized = False  # 🔥 Prevent double shutdowns
            print("✅ Hardware connection closed properly.")

            time.sleep(2)  # 🔥 Increased to avoid conflict on next `reset()`

        except Exception as e:
            print(f"⚠️ Error while closing hardware: {e}")
