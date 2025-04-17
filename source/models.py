import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import time
import math

def model_factory(model_class, kwargs) -> "DynamicsModel":
    model_class = globals()[model_class]
    model = model_class(**kwargs)
    return model



class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, output_dim, bias=True)
        for l in [self.linear1, self.linear2, self.linear3, self.linear4]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

    def forward(self, x):
        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        h = F.relu(self.linear3(h))
        return self.linear4(h)

class DynamicsModel(nn.Module):
    def __init__(self, integration_scheme="symplectic", dt=0.01, wrap_angle=True, **kwargs):
        super().__init__()
        self.integration_scheme = integration_scheme
        self.dt = dt
        self.wrap_angle = wrap_angle

    def derivative(self, state, action):
        raise NotImplementedError

    def get_Fa(self):
        raise NotImplementedError

    @staticmethod
    def wrap_state(state):
        wrapped = state.clone()

        return wrapped

    def forward(self, state, action):
        w_state = self.wrap_state(state) if self.wrap_angle else state

        if self.integration_scheme == "euler":
            return state + self.derivative(w_state, action) * self.dt

        elif self.integration_scheme == "symplectic":

            assert state.shape[1] % 2 == 0  # [Only for x, dx] systems
            p = state.shape[1] // 2
            new_state = state + self.derivative(w_state, action) * self.dt  # Euler
            new_state[:, :p] = state[:, :p] + new_state[:, p:] * self.dt

            #print("new state", new_state)
            return self.wrap_state(new_state)


                
    def integrate(self, state, actions, resets=None):
        """
        Integrate a trajectory
        :param state: initial state, of shape (batch x state size)
        :param actions: sequence of actions, of shape: (horizon x batch x action size)
        :param resets: dict of (timestep, state) such that the trajectory at time timestep is reset to state
        :return: resulting trajectory, of shape: (horizon x batch x state size)
        """
        states = torch.zeros((actions.shape[0], state.shape[0], state.shape[1])).to(state.device)
        with torch.no_grad():
            for t in range(actions.shape[0]):
                state = self.forward(state, actions[t, ...])
                if resets and t in resets:
                    state = resets[t]
                states[t, ...] = state
        return states



class MLPModel(DynamicsModel):
    def __init__(self, state_size, action_size, hidden_size=16,  dt=0.01, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.state_size, self.action_size = state_size, action_size
        self.mlp = MLP(state_size + action_size, hidden_size, state_size)

    def derivative(self, state, action):
        """
            Predict dx_t = MLP(x_t,u_t)
        :param x: a batch of states
        :param u: a batch of actions
        """

        xu = torch.cat((state, action), -1)
        return self.mlp(xu)


    
class FrictionlessFuruta(DynamicsModel):
    def __init__(self, device="cuda", integration_scheme="symplectic", dt=0.01, **kwargs):
        super().__init__(integration_scheme=integration_scheme, dt=dt, **kwargs)
        self.device = device

        self.g = 9.8
        #self.force = nn.Parameter(torch.tensor(4.0, dtype=torch.float32, device=self.device), requires_grad=True)
        self.mr = nn.Parameter(torch.tensor(0.095, dtype=torch.float32, device=self.device), requires_grad=False)
        self.mp = nn.Parameter(torch.tensor(0.024, dtype=torch.float32, device=self.device), requires_grad=False)
        self.Lr = nn.Parameter(torch.tensor(0.085, dtype=torch.float32, device=self.device), requires_grad=False)
        self.Lp = nn.Parameter(torch.tensor(0.129, dtype=torch.float32, device=self.device), requires_grad=False)
        self.Rm = nn.Parameter(torch.tensor(8.4, dtype=torch.float32, device=self.device), requires_grad=False)
        self.kt = nn.Parameter(torch.tensor(0.042, dtype=torch.float32, device=self.device), requires_grad=False)
        self.km = nn.Parameter(torch.tensor(0.042, dtype=torch.float32, device=self.device), requires_grad=False)
        self.J1 = nn.Parameter(torch.tensor(2.3060e-04, dtype=torch.float32, device=self.device), requires_grad=False)
        self.J2 = nn.Parameter(torch.tensor(1.3313e-04, dtype=torch.float32, device=self.device), requires_grad=False)


    def derivative(self, state, action): 

        alpha = state[:, 0]
        beta = state[:, 1]
        cosbeta = torch.cos(beta)
        sinbeta = torch.sin(beta)

        dx = torch.zeros_like(state)
        dx[:, 0] = state[:, 2]
        dx[:, 1] = state[:, 3]

        # Mass matrix elements with explicit shape matching
        m11 = self.J1 + 0.25 * self.mp * (self.Lp ** 2) * (sinbeta ** 2)

        #print("model m11", m11)

        m22 = self.J2.expand_as(m11)

        #print("model m22", m22)


        m12 = 0.5 * self.mp * self.Lr * self.Lp * cosbeta

        #print("model m12", m12)

        # Stack the mass matrix in a batch-friendly way
        MassMat = torch.stack([torch.stack([m11, m12], dim=-1), torch.stack([m12, m22], dim=-1)], dim=-2)


        # Nonlinear terms vector (N_vec) in a batch-friendly way

        N_vec = 0.5 * self.mp * self.Lp * sinbeta.unsqueeze(-1) * torch.stack([
            (self.Lp * state[:, 2] * state[:, 3] * cosbeta - self.Lr * (state[:, 3] ** 2)),
            (-0.5 * self.Lp * (state[:, 2] ** 2) * cosbeta)
        ], dim=-1)


        # Gravity vector (G_vec)
        G_vec = torch.stack([torch.zeros_like(sinbeta), 0.5 * self.mp * self.g * self.Lp * sinbeta], dim=-1)

        #print("model G_vec", G_vec)

        # Motor torque (tau_mot) and torque vector (tau_vec)
        tau_mot = self.kt * (-5.0*action[:, 0] - self.km * state[:, 2]) / self.Rm

        #print("model tau_mot", tau_mot)

        tau_vec = torch.stack([
            tau_mot.unsqueeze(-1),
            torch.zeros_like(tau_mot).unsqueeze(-1)
        ], dim=-1).squeeze(1)

        #print("model tau_vec", tau_vec)

        # Inverse of the Mass matrix for each batch element
        Minv = torch.linalg.inv(MassMat)

        #print("model Minv", Minv)

        # Compute ddx (acceleration vector) for each batch element
        ddx = torch.bmm(Minv, (tau_vec - N_vec - G_vec).unsqueeze(-1)).squeeze(-1)

        #print("model ddx", ddx)

        # Update dx with accelerations
        dx[:, 2], dx[:, 3] = ddx[:, 0], ddx[:, 1]

        return dx
    

    def get_params(self):

        return {"mr 0.095": self.mr.data,
                "mp 0.024": self.mp.data,
                "Lr 0.085": self.Lr.data,
                "Lp 0.129":self.Lp.data,
                "Rm 8.4":self.Rm.data,
                "kt 0.042": self.kt.data,
                "km 0.042": self.km.data,
                "J1 2.3060e-04":self.J1.data,
                "J2 1.3313e-04":self.J2.data,             
                
                }  


    
class AugmentedFurutaModel(DynamicsModel):
    """
    Augmented model prior+mlp
    """
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        super().__init__(integration_scheme=integration_scheme, **kwargs)
        self.device = device
        self.prior = FrictionlessFuruta(**kwargs)
        self.mlp = MLPModel(**kwargs)
        self.Fa = torch.tensor(0)
        self.Fp = torch.tensor(0)

    def derivative(self, state, action):
        self.Fp = self.prior.derivative(state, action)
        self.Fa = self.mlp.derivative(state, action)  # RTHCP augmentation 
        return self.Fp + self.Fa   # aphynity

    def get_Fa(self):
        return self.Fa, self.Fp  


             
class RTHCP_Furuta(AugmentedFurutaModel):
    """
    Augmented model prior+mlp   with L2 contstarint in the loss function
    """
    def __init__(self, device="cuda", integration_scheme="symplectic", **kwargs):
        super().__init__(**kwargs)
        self.use_l2con = True # use L2 constraint in the loss function
             
