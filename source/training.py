from pathlib import Path
import pandas as pd
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
import logging

from source.models import DynamicsModel


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self,
                 model: DynamicsModel,
                 device: str = "cuda",
                 epochs: int = 1000,
                 lr: float = 0.01,
                 lambda_0: float = 1,
                 Niter: int = 1,
                 aph_tau: int = 10,
                 data_path: str = "data/dataset.csv",
                 model_path: str = "{}_model_episode{}_seed{}.tar",
                 directory: str = "out/",
                 seed: int = 5,
                 **kwargs: dict) -> None:
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.data_path = data_path
        self.model_path = model_path
        self.directory = Path(directory)
        self.data = None
        self.train_data, self.test_data = None, None
        self.loss = torch.nn.MSELoss()
        self.losses = []
        self.lambda_0 = lambda_0
        self.Niter = Niter
        self.aph_tau = aph_tau
        self.seed = seed



    def load_data(self, df=None):
        if df is None:
            print(f"Loading data from {self.directory / self.data_path}")
            df = pd.read_csv(self.directory / self.data_path)
            #print(df)
            df = df



        states = df.filter(like='state').to_numpy()
        actions = df.filter(like='action').to_numpy()
        time = df["time"].to_numpy()


        # Transitions
        next_states = states[1:]
        time = time[:-1]
        states = states[:-1]
        actions = actions[:-1]/5.0

        # Remove env resets
        no_resets = np.where((0.015 > np.diff(time)) & (np.diff(time) > 0) )

        #print(no_resets)
        next_states, time, states, actions = next_states[no_resets], time[no_resets], states[no_resets], actions[no_resets]

        # To tensors
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        time = torch.tensor(time, dtype=torch.float).to(self.device)
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        self.data = (time, states, actions, next_states)
        #print(self.data[:5])

    def split_train_test(self, ratio=0.8):
        time, states, actions, next_states = self.data
        ind_list = np.arange(time.shape[0]).tolist()
        np.random.shuffle(ind_list)
        train_size = int(states.shape[0] * ratio)
        train_data = (time[ind_list[:train_size]], states[ind_list[:train_size]], actions[ind_list[:train_size]], next_states[ind_list[:train_size]])
        test_data = (time[ind_list[train_size:]], states[ind_list[train_size:]], actions[ind_list[train_size:]], next_states[ind_list[train_size:]])
        self.train_data, self.test_data = train_data, test_data

    def save_model(self, episode):

        out_path = self.directory / self.model_path.format(self.model.__class__.__name__, episode, self.seed)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saved model to {out_path}")
        torch.save(self.model.state_dict(), out_path)



    def compute_loss(self, data):
        times, states, actions, next_states = data
        predictions = self.model(states, actions)
        #Fa, Fp = self.model.get_Fa()
        #norm_Fa_ = torch.norm(Fa) / torch.norm(Fp)
        return torch.mean(torch.sum((predictions - next_states) ** 2, dim=1)) #, norm_Fa_

    def compute_constrained_loss(self, data, lambda_t):
        _, states, actions, next_states = data
        predictions = self.model(states, actions)
        lpred = self.loss(predictions, next_states)
        Fa, Fp = self.model.get_Fa()
        norm_Fa_ = torch.norm(Fa) / torch.norm(Fp)
        norm_Fa = norm_Fa_ #########################
        loss = lpred + norm_Fa / lambda_t
        return loss, lpred, norm_Fa

    def train(self, df=None):
        self.load_data(df)
        self.split_train_test()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.losses = np.full((self.epochs, 2), np.nan)
        epochs = tqdm.trange(self.epochs, desc="Train dynamics") if logger.getEffectiveLevel() == logging.DEBUG \
            else range(self.epochs)

        use_l2con = getattr(self.model, "use_l2con", False)
        logger.debug(f"Train with constrained : {use_l2con}")
        if use_l2con:
            lambda_t = self.lambda_0
            for epoch in epochs:
                for _ in range(self.Niter):
                    loss, lpred, norm_Fa = self.compute_constrained_loss(self.train_data, lambda_t)
                    validation_loss = self.compute_loss(self.test_data)
                    self.losses[epoch] = [loss.detach().cpu().numpy(), validation_loss.detach().cpu().numpy()]
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                lambda_t = lambda_t + self.aph_tau * lpred.detach()
                if epoch% 1000==0.0:
                    print(f"\n{epoch}: loss {loss.item()}, lpred {lpred.item()}, |Fa| {norm_Fa.item()}, lambda_t {lambda_t.item()}")
        else:
            for epoch in epochs:
                # Compute loss gradient and step optimizer
                loss = self.compute_loss(self.train_data)
                validation_loss = self.compute_loss(self.test_data)
                self.losses[epoch] = [loss.detach().cpu().numpy(), validation_loss.detach().cpu().numpy()]
