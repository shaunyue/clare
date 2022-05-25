import abc

import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Distribution, Normal

from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import BasePolicy
from offlinerl.utils.net.continuous import ActorProb


class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)

    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        self.mode = torch.tanh(normal_mean)

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def atanh(self,x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5 * torch.log(one_plus_x / one_minus_x)

    def log_prob(self, value, pre_tanh_value=None):
        """        
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = self.atanh(value)

        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def sample(self, return_pretanh_value=False):
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
    


    def rsample(self, return_pretanh_value=False):
        z = (
            self.normal_mean +
            self.normal_std *
            Normal(
                torch.zeros(self.normal_mean.size(), device=self.normal_mean.device),
                torch.ones(self.normal_std.size(), device=self.normal_mean.device)
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)
            # return torch.tanh(z)
    def sample_com(self):
        # mean std 分别为方差和均值
        normal = Normal(self.normal_mean, self.normal_std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        x_t.requires_grad_()
        # x_t = normal.sample()
        y_t = torch.tanh(x_t)
        # print("self.action_scale")
        # print(self.action_scale)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob

class TanhGaussianPolicy(ActorProb, BasePolicy):
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -5
    MEAN_MIN = -9.0
    MEAN_MAX = 9.0
    
    def atanh(self,x):
        one_plus_x = (1 + x).clamp(min=1e-6)
        one_minus_x = (1 - x).clamp(min=1e-6)
        return 0.5*torch.log(one_plus_x/ one_minus_x)

    def log_prob(self, obs, actions):
        raw_actions = self.atanh(actions)
        logits, h = self.preprocess(obs)
        
        mean = self.mu(logits)
        mean = torch.clamp(mean, self.MEAN_MIN, self.MEAN_MAX)
        if self._c_sigma:
            log_std = torch.clamp(
                self.sigma(logits), min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX
            )
            std = log_std.exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            log_std = (self.sigma.view(shape) + torch.zeros_like(mu))
            std = log_std.exp()

        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
        return log_prob.sum(-1)

    def forward(
            self,
            obs,
            state=None,
            infor={},
            reparameterize=True,

    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        logits, h = self.preprocess(obs, state)
        mean = self.mu(logits)
        
        if self._c_sigma:
            log_std = torch.clamp(
                self.sigma(logits), min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX
            )
            std = log_std.exp()
        else:
            print("no self._c_sigma --TanhGaussianPolicy")
            shape = [1] * len(mu.shape)
            shape[1] = -1
            log_std = (self.sigma.view(shape) + torch.zeros_like(mu))
            std = log_std.exp()
        
        return TanhNormal(mean, std)
    
    def policy_infer(self, obs):
        return self(obs).mode