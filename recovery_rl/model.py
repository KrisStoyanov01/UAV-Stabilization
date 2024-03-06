'''
Latent dynamics models are built on latent dynamics model used in
Goal-Aware Prediction: Learning to Model What Matters (ICML 2020). All
other networks are built on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
'''
Global utilities
'''


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# Soft update of target critic network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


# Hard update of target critic network
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


'''
Architectures for critic functions and policies for SAC model-free recovery
policies.
'''


# Q network architecture
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        # x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        # x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# L network architecture


class LNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(LNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        # x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        # x1 = -F.relu(x1)
        x1 = -torch.norm(x1, dim=1, keepdim=True)

        x2 = F.relu(self.linear4(xu))
        # x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        # x2 = -F.relu(x2)
        x2 = -torch.norm(x2, dim=1, keepdim=True)

        return x1, x2


# Q_risk network architecture
class QNetworkConstraint(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkConstraint, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_inputs + num_actions)
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = F.sigmoid(self.linear3(x1))

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = F.sigmoid(self.linear6(x2))

        return x1, x2


# Gaussian policy for SAC
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample(
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.cpu()
        self.action_bias = self.action_bias.cpu()
        return super(GaussianPolicy, self).cpu()


# Deterministic policy for model free recovery
class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.cpu()
        self.action_bias = self.action_bias.cpu()
        self.noise = self.noise.cpu()
        return super(DeterministicPolicy, self).cpu()


# Stochastic policy for model free recovery
class StochasticPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(StochasticPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = torch.nn.Parameter(
            torch.as_tensor([np.log(0.1)] * num_actions))
        self.min_log_std = np.log(1e-6)

        self.apply(weights_init_)
        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        # print(self.log_std)
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        log_std = log_std.unsqueeze(0).repeat([len(mean), 1])
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, state):
        dist = self.forward(state)
        action = dist.rsample()
        return action, dist.log_prob(action).sum(-1), dist.mean

    def to(self, device):
        self.action_scale = self.action_scale.cpu()
        self.action_bias = self.action_bias.cpu()
        return super(StochasticPolicy, self).cpu()


# Q network architecture for image observations
class QNetworkCNN(nn.Module):
    def __init__(self, num_actions, hidden_dim):
        super(QNetworkCNN, self).__init__()
        self.img_features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 2, kernel_size=5, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.state_features = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout()
        )
        self.state_img_features = nn.Sequential(
            nn.Linear(18 + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.action_features = nn.Sequential(
            nn.Linear(num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.state_action_1 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.state_action_2 = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weights_init_)

    def forward(self, obs_state, obs_img, action):
        state_features = self.state_features(obs_state)
        img_features = self.img_features(obs_img)
        img_features = img_features.view(-1, 18)

        cmb_states = torch.cat([state_features, img_features], dim=1)
        final_state_features = self.state_img_features(cmb_states)
        x0 = self.action_features(action)
        xu = torch.cat([final_state_features, x0], dim=1)
        x1 = self.state_action_1(xu)
        x2 = self.state_action_2(xu)
        return x1, x2


# Gaussian policy for SAC for image observations
class GaussianPolicyCNN(nn.Module):
    def __init__(self,
                 num_actions,
                 hidden_dim,
                 action_space=None):
        super(GaussianPolicyCNN, self).__init__()
        self.img_features = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=5, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 2, kernel_size=5, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.state_features = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout()
        )
        self.state_img_features = nn.Sequential(
            nn.Linear(18 + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, obs_state, obs_img):
        state_features = self.state_features(obs_state)
        img_features = self.img_features(obs_img)
        img_features = img_features.view(-1, 18)
        cmb_states = torch.cat([state_features, img_features], dim=1)
        final_state_features = self.state_img_features(cmb_states)

        # Now do normal SAC stuff
        x = F.relu(self.linear1(final_state_features))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, obs_state, obs_img):
        mean, log_std = self.forward(obs_state, obs_img)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample(
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.cpu()
        self.action_bias = self.action_bias.cpu()
        return super(GaussianPolicyCNN, self).cpu()
