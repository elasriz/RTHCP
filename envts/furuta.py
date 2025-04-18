import gymnasium as gym
from gymnasium import core, spaces
from gymnasium.envs.registration import register
import numpy as np
from typing import Any, Callable, List, Optional, Set
from os import path
from time import time_ns, sleep, perf_counter
from gymnasium.envs.classic_control import utils

from typing import Optional

import numpy as np
from numpy import cos, pi, sin

from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding

import math


class FurutaEnv(gym.Env):

    torque_noise_max = 0.0
    action_arrow = None
    domain_fig = None

    def __init__(self, frequency=None):



        self.ROTOR_LENGTH = 0.085  # [m] Rotor length (Lr)
        self.PENDULUM_LENGTH = 0.129  # [m] Pendulum length (Lp)

        self.ROTOR_MASS = 0.095 #: [kg] Rotor Mass (mr)
        self.PENDULUM_MASS = 0.024  #: [kg] Pendulum Mass (mp)

        self.ROTOR_MOI = 2.3060e-04  #: [m] Rotor moment of inertia (J1)
        self.PENDULUM_MOI = 1.3313e-04  #: [m] Rotor moment of inertia (J2)

        self.MAX_VEL_1 = np.inf # 2 * np.pi
        self.MAX_VEL_2 = np.inf # 5 * np.pi

        self.friction_coeff_rotor = 5.0000e-04 # Coefficient of friction of the Rotor (cr)
        self.friction_coeff_pendulum = 5.0000e-05 # Coefficient of friction of the Rotor (cp)

        self.MOTOR_RESISTANCE = 8.4 # [Ohm]
        self.MOTOR_TORQUE_CONST = 0.042
        self.MOTOR_BEMF_CONST = 0.042


        high = np.array(
            [np.inf, np.inf, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )

        
        self.observation_space = spaces.Box(shape=(4,), low=-high, high=high)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.state = None

        self.seed()
        #self.dt = 0.01
        self.action_max = 4.0
        self.__frequency = frequency

        self.all_act = []
        self.all_alpha = []
        self.elapsed_time = None


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
    
    def all_actions(self):
        return self.all_act

    def all_alphas(self):
        return self.all_alpha  

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return np.array(self.state, dtype=np.float32)
        #return np.array(
        #    [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32
        #)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.

        self.next_start_time = time_ns()
        self.initial_time = None          
     
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.1, 0.1  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,)).astype(
            np.float32
        )

        return self._get_ob(), {}

    def seed(self, seed=None, options=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):


        if not self.initial_time:
            self.initial_time = time_ns()
            self.next_start_time  = self.initial_time + (self.dt - 0.002) * 1e9        

        s = self.state

        assert s is not None, "Call reset before using AcrobotEnv object."
        torque = np.array(self.action_max) * a

        s_augmented = np.append(s, torque)

        ns = symplectic(self._dsdt, s_augmented, self.dt)

        in_step = True
        while in_step:
            try:
        # Wait for next loop start time
                if time_ns() > self.next_start_time:

                    self.state = ns
                    terminated = self._terminal()
                    reward = np.exp(-0.5*( 1.0* (1.0  +  np.cos( s[1] ))**2 +  np.sin( s[1] )**2 + 0.5 * ( np.cos(s[0]) - 1)**2  + 0.0005*s[2]**2 + 0.00025*s[3]**2 + 0.00001 * (a[0]**2)))
                    in_step = False
                    self.next_start_time  = time_ns() + (self.dt - 0.002) * 1e9

            finally:

                break  

        return (self._get_ob(), reward, terminated, False, {})

    def _terminal(self):
        s = self.state
        return bool(np.cos(s[0]) < -0.4)
    
    def _dsdt(self, s_augmented):
        mr = self.ROTOR_MASS
        mp = self.PENDULUM_MASS
        Lr = self.ROTOR_LENGTH
        Lp = self.PENDULUM_LENGTH
        J1 = self.ROTOR_MOI
        J2 = self.PENDULUM_MOI
        cr = self.friction_coeff_rotor
        cp = self.friction_coeff_pendulum
        Rm = self.MOTOR_RESISTANCE
        kt = self.MOTOR_TORQUE_CONST
        km = self.MOTOR_BEMF_CONST
        g = 9.8
        a = s_augmented[4:]
        s = s_augmented[:4]

        alpha = s[0]
        beta = s[1]
        dalpha = s[2]
        dbeta = s[3]

        cosbeta = np.cos(beta)
        sinbeta = np.sin(beta)
        
        # Mass matrix
        m11 = J1 + 0.25*mp*(Lp**2)*(sinbeta**2)
        m22 = J2
        m12 = 0.5 * mp*Lr*Lp*cosbeta

        MassMat = np.array([[m11, m12],[m12, m22]])
        
        N_vec = 0.5*mp*Lp*sinbeta * np.array([Lp*dalpha*dbeta*cosbeta-Lr*(dbeta**2),
                                                   -0.5*Lp*(dalpha**2)*cosbeta])
        
        G_vec = np.array([0.,
                      0.5*mp*g*Lp*sinbeta])
    

        tau_mot = kt*(-a[0] - km*dalpha)/Rm 

        tau_vec = np.array([tau_mot - cr*dalpha,
                            -cp*dbeta])
        

        Minv = np.linalg.inv(MassMat)

        ddx = Minv @ (tau_vec - N_vec - G_vec)

        ddalpha, ddbeta = ddx
        
        return dalpha, dbeta, ddalpha, ddbeta, 0.0, 0.0

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

def rk4(derivs, y0, t):
    """
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.
    Example for 2D system:
        >>> def derivs(x):
        ...     d1 =  x[0] + 2*x[1]
        ...     d2 =  -3*x[0] + 4*x[1]
        ...     return d1, d2
        >>> dt = 0.0005
        >>> t = np.arange(0.0, 2.0, dt)
        >>> y0 = (1,2)
        >>> yout = rk4(derivs, y0, t)
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times
    Returns:
        yout: Runge-Kutta approximation of the ODE
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    ##print(yout[-1])
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]

def symplectic(derivs, y0, dt):
    
    y = np.asarray(derivs(y0))
    yout = y0[:4] + dt  * y[:4]
    yout[0] = y0[0] + dt  * yout[2]
    yout[1] = y0[1] + dt  * yout[3]

    return yout

register(
    id='furuta',
    entry_point='envts.furuta:FurutaEnv',
    max_episode_steps=500,
)
