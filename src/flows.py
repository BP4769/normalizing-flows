import torch
import torch.nn as nn

from src.bijections.finite.base import Bijection


class Flow(nn.Module):
    def __init__(self,
                 n_dim: int,
                 bijection: Bijection,
                 device: torch.device):
        super().__init__()
        self.base = torch.distributions.MultivariateNormal(
            loc=torch.zeros(n_dim, device=device),
            covariance_matrix=torch.eye(n_dim, device=device)
        )
        self.bijection = bijection.to(device)

    def log_prob(self, x: torch.Tensor):
        z, log_det = self.bijection.forward(x)
        log_base = self.base.log_prob(z)
        return log_base + log_det

    def sample(self, n: int):
        z = self.base.sample(sample_shape=torch.Size((n,)))
        x, _ = self.bijection.inverse(z)
        return x