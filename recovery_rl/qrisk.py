import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.optim import Adam

from recovery_rl.model import QNetworkConstraint, StochasticPolicy
from recovery_rl.utils import hard_update, soft_update


class QRiskWrapper:
    def __init__(self, obs_space, ac_space, hidden_size, logdir,
                 args, tmp_env):
        self.env_name = args.env_name
        self.logdir = logdir
        self.device = torch.device("cpu")
        self.ac_space = ac_space

        self.safety_critic = QNetworkConstraint(
            obs_space.shape[0], ac_space.shape[0],
            hidden_size).to(device=self.device)
        self.safety_critic_target = QNetworkConstraint(
            obs_space.shape[0], ac_space.shape[0],
            args.hidden_size).to(device=self.device)

        self.lr = args.lr
        self.safety_critic_optim = Adam(self.safety_critic.parameters(),
                                        lr=args.lr)
        hard_update(self.safety_critic_target, self.safety_critic)

        self.tau = args.tau_safe
        self.gamma_safe = args.gamma_safe
        self.updates = 0
        self.target_update_interval = args.target_update_interval
        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        self.policy = StochasticPolicy(obs_space.shape[0],
                                       ac_space.shape[0], hidden_size,
                                       ac_space).to(self.device)

        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.pos_fraction = args.pos_fraction if args.pos_fraction >= 0 else None
        self.tmp_env = tmp_env
        self.encoding = None

    def update_parameters(self,
                          memory=None,
                          policy=None,
                          batch_size=None,
                          plot=False):
        '''
        Trains safety critic Q_risk and model-free recovery policy which performs
        gradient ascent on the safety critic

        Arguments:
            memory: Agent's replay buffer
            policy: Agent's composite policy
            critic: Safety critic (Q_risk)
        '''
        if self.pos_fraction:
            batch_size = min(batch_size,
                             int((1 - self.pos_fraction) * len(memory)))
        else:
            batch_size = min(batch_size, len(memory))
        state_batch, action_batch, constraint_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size, pos_fraction=self.pos_fraction)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        constraint_batch = torch.FloatTensor(constraint_batch.astype(np.float32)).to(
            self.device).unsqueeze(1)

        if self.encoding:
            state_batch_enc = self.encoder(state_batch)
            next_state_batch_enc = self.encoder(next_state_batch)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = policy.sample(
                next_state_batch)
            if self.encoding:
                qf1_next_target, qf2_next_target = self.safety_critic_target(
                    next_state_batch_enc, next_state_action)
            else:
                qf1_next_target, qf2_next_target = self.safety_critic_target(
                    next_state_batch, next_state_action)
            min_qf_next_target = torch.max(qf1_next_target, qf2_next_target)
            next_q_value = constraint_batch + mask_batch * self.gamma_safe * (
                min_qf_next_target)

        if self.encoding:
            qf1, qf2 = self.safety_critic(
                state_batch_enc, action_batch
            )  # Two Q-functions to mitigate positive bias in the policy improvement step
        else:
            qf1, qf2 = self.safety_critic(
                state_batch, action_batch
            )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(
            qf1, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(
            qf2, next_q_value
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        self.safety_critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.safety_critic_optim.step()

        if self.updates % self.target_update_interval == 0:
            soft_update(self.safety_critic_target, self.safety_critic,
                        self.tau)
        self.updates += 1

    def get_value(self, states, actions, encoded=False):
        '''
            Arguments:
                states, actions --> list of states and list of corresponding 
                actions to get Q_risk values for
            Returns: Q_risk(states, actions)
        '''
        with torch.no_grad():
            if self.encoding and not encoded:
                q1, q2 = self.safety_critic(self.encoder(states), actions)
            else:
                q1, q2 = self.safety_critic(states, actions)
            return torch.max(q1, q2)

    def select_action(self, state, eval=False):
        '''
            Gets action from model-free recovery policy

            Arguments:
                Current state
            Returns:
                action
        '''
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.MF_recovery:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
            return action.detach().cpu().numpy()[0]
        elif self.Q_sampling_recovery:
            if not self.images:
                state_batch = state.repeat(1000, 1)
            else:
                state_batch = state.repeat(1000, 1, 1, 1)
            sampled_actions = torch.FloatTensor(
                np.array([self.ac_space.sample()
                          for _ in range(1000)])).to(self.device)
            q_vals = self.get_value(state_batch, sampled_actions)
            min_q_value_idx = torch.argmin(q_vals)
            action = sampled_actions[min_q_value_idx]
            return action.detach().cpu().numpy()
        else:
            assert False

    def __call__(self, states, actions):
        return self.safety_critic(states, actions)
