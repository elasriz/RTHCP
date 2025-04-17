import numpy as np




class RandomPolicy(object):
    name: str = "Random"

    def __init__(self, env: object, noise_tau: float = 200, sigma: float = 0.1, noise_schedule_tau=50, **kwargs: dict) -> None:
        self.env = env
        self.name = "Random"
        self.action_space = env.action_space
        self.noise_tau = noise_tau
        self.sigma = sigma
        self.time = 0
        self.noise_schedule_tau = noise_schedule_tau
        self.last_action = None

    def ornstein_uhlenbeck_noise(self):

        low = getattr(self.action_space, "minimum", getattr(self.action_space, "low", None))
        high = getattr(self.action_space, "maximum", getattr(self.action_space, "high", None))

        noise = np.random.uniform(low=low, high=high)\
            .astype(self.action_space.dtype, copy=False)
        return self.last_action - 1 / self.noise_tau * self.last_action + self.sigma * noise \
            if self.last_action is not None else self.sigma * noise


    def act(self, best_action=None):
        action = self.ornstein_uhlenbeck_noise()
        self.last_action = action
        if best_action is not None:
            action = self.schedule * best_action + (1 - self.schedule) * action
        return action

    def step(self):
        self.time += 1



        
    @property
    def schedule(self):
        return min(self.time / self.noise_schedule_tau, 1)