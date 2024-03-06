import datetime
import itertools
import os
import os.path as osp
import pickle

import numpy as np
import torch
from torch import nn, optim

from env.make_utils import make_env, register_env
from recovery_rl.replay_memory import (ConstraintReplayMemory, ReplayMemory,
                                       ReplayMemoryCNN)
from recovery_rl.sac import SAC
from recovery_rl.utils import linear_schedule

# TORCH_DEVICE = torch.device(
# 'cuda') if torch.cuda.is_available() else torch.device('cpu')

TORCH_DEVICE = torch.device('cpu')


# def torchify(x): return torch.FloatTensor(x).to('cuda')
def torchify(x): return torch.FloatTensor(x).to('cpu')


NUM_STEPS_TO_SAVE = 500


class Experiment:
    def __init__(self, exp_cfg):
        self.exp_cfg = exp_cfg
        self.name = self.exp_cfg.experiment_name + '_' + \
            self.exp_cfg.env_name + '_' + self.exp_cfg.method_name

        if not os.path.exists('./saved_model'):
            os.mkdir('saved_model')
        if not os.path.exists('./training_logs'):
            os.mkdir('training_logs')

        if not os.path.exists('./saved_model/' + self.name):
            os.mkdir('./saved_model/' + self.name)
        if not os.path.exists('./training_logs/' + self.name):
            os.mkdir('./training_logs/' + self.name)

        # Logging setup
        self.logdir = os.path.join('training_logs', self.name,
                                   self.exp_cfg.logdir, '{}_SAC_{}_{}_{}'.format(
                                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                       self.exp_cfg.env_name, self.exp_cfg.policy,
                                       self.exp_cfg.logdir_suffix))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        print("LOGDIR: ", self.logdir)
        pickle.dump(self.exp_cfg, open(
            os.path.join(self.logdir, "args.pkl"), "wb"))

        # Experiment setup
        self.experiment_setup()

        # Memory
        if self.exp_cfg.cnn:
            self.memory = ReplayMemoryCNN(
                self.exp_cfg.replay_size, self.exp_cfg.seed)
        else:
            self.memory = ReplayMemory(
                self.exp_cfg.replay_size, self.exp_cfg.seed)
        self.recovery_memory = ConstraintReplayMemory(
            self.exp_cfg.safe_replay_size, self.exp_cfg.seed)
        self.all_ep_data = []

        self.total_numsteps = 0
        self.updates = 0
        self.num_viols = 0
        self.num_successes = 0

        # Get demos
        self.task_demos = self.exp_cfg.task_demos

        # Get multiplier schedule for RSPO
        if self.exp_cfg.nu_schedule:
            self.nu_schedule = linear_schedule(self.exp_cfg.nu_start,
                                               self.exp_cfg.nu_end,
                                               self.exp_cfg.num_eps)
        else:
            self.nu_schedule = linear_schedule(self.exp_cfg.nu,
                                               self.exp_cfg.nu, 0)

    def experiment_setup(self):
        torch.manual_seed(self.exp_cfg.seed)
        np.random.seed(self.exp_cfg.seed)
        register_env(self.exp_cfg.env_name)
        env = make_env(self.exp_cfg.env_name)
        self.env = env
        self.env.seed(self.exp_cfg.seed)
        self.env.action_space.seed(self.exp_cfg.seed)
        agent = self.agent_setup(env)
        self.agent = agent

    def agent_setup(self, env):
        agent = SAC(env.observation_space,
                    env.action_space,
                    self.exp_cfg,
                    self.logdir,
                    tmp_env=make_env(self.exp_cfg.env_name))
        return agent

    def run(self):
        print("Start Training...")
        train_rollouts = []
        test_rollouts = []
        for i_episode in itertools.count(1):
            self.agent.i_episode = i_episode
            self.env.i_episode = i_episode
            train_rollout_info = self.get_train_rollout(i_episode)
            train_rollouts.append(train_rollout_info)
            if i_episode % NUM_STEPS_TO_SAVE == 0:
                PATH_policy = "./saved_model/" + self.name + "/policy_" + self.name + '_' + str(i_episode) + \
                    "_seed" + str(self.exp_cfg.seed) + ".pkl"
                PATH_critic = "./saved_model/" + self.name + "/critic_" + self.name \
                    + "_" + str(i_episode) + \
                    "_seed" + str(self.exp_cfg.seed) + ".pkl"
                print(PATH_policy)

                torch.save(self.agent.policy, PATH_policy)
                torch.save(self.agent.critic, PATH_critic)
            if i_episode > self.exp_cfg.num_eps:
                break
            self.dump_logs(train_rollouts, test_rollouts)

    def get_train_rollout(self, i_episode):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = self.env.reset()

        train_rollout_info = []
        ep_states = [state]
        ep_actions = []
        ep_constraints = []

        if i_episode % 10 == 0:
            print("SEED: ", self.exp_cfg.seed)
            print("LOGDIR: ", self.logdir)

        while not done:
            if len(self.memory) > self.exp_cfg.batch_size and (episode_steps % 2 == 0):
                # Number of updates per step in environment
                for i in range(self.exp_cfg.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(
                        self.memory,
                        min(self.exp_cfg.batch_size, len(self.memory)),
                        i_episode,
                        safety_critic=self.agent.safety_critic,
                        nu=self.nu_schedule(i_episode))
                    if not self.exp_cfg.disable_online_updates and len(
                            self.recovery_memory) > self.exp_cfg.batch_size and (self.num_viols / self.exp_cfg.batch_size) > self.exp_cfg.pos_fraction:
                        self.agent.safety_critic.update_parameters(
                            memory=self.recovery_memory,
                            policy=self.agent.policy,
                            batch_size=self.exp_cfg.batch_size,
                            plot=0)
                    self.updates += 1
            # Get action, execute action, and compile step results
            action = self.agent.select_action(state)
            next_state, reward, done, info = self.env.step(action)

            if info['constraint']:
                reward -= self.exp_cfg.constraint_reward_penalty

            train_rollout_info.append(info)
            episode_steps += 1
            episode_reward += reward
            self.total_numsteps += 1

            mask = float(not done)

            if episode_steps == self.env._max_episode_steps:
                break

            # Update buffers
            if not self.exp_cfg.disable_action_relabeling:
                self.memory.push(state, action, reward, next_state, mask)
            else:
                # absorbing state
                if info['constraint']:
                    for _ in range(20):
                        self.memory.push(
                            state, action, reward, next_state, mask)
                self.memory.push(state, action, reward, next_state, mask)

            if self.exp_cfg.DGD_constraints or self.exp_cfg.RCPO:
                self.recovery_memory.push(state, action,
                                          info['constraint'], next_state, mask)

            state = next_state
            ep_states.append(state)
            ep_actions.append(action)
            ep_constraints.append([info['constraint']])

        # Get success/violation stats
        if info['constraint']:
            self.num_viols += 1

        self.num_successes += int(info['success'])

        # Print performance stats
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".
              format(i_episode, self.total_numsteps, episode_steps,
                     round(episode_reward, 2)))
        print("Num Violations So Far: %d" % self.num_viols)
        print("Num Successes So Far: %d" % self.num_successes)
        if info["constraint"]:
            print("Reason: violate")
        elif info["success"]:
            print("Reason: success")
        else:
            print("Reason: timeout")
        print("=========================================")

        with open('logger_' + self.name + '.txt', 'a') as f:
            f.write("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}\n".format(
                i_episode, self.total_numsteps, episode_steps, round(episode_reward, 2)))
            f.write("Num Violations So Far: %d\n" % self.num_viols)
            f.write("Num Successes So Far: %d\n" % self.num_successes)
            if info["constraint"]:
                f.write("Reason: violate\n")
            elif info["success"]:
                f.write("Reason: success\n")
            else:
                f.write("Reason: timeout\n")
            f.write("=========================================")

        return train_rollout_info

    def dump_logs(self, train_rollouts, test_rollouts):
        data = {"test_stats": test_rollouts, "train_stats": train_rollouts}
        with open(osp.join(self.logdir, "run_stats.pkl"), "wb") as f:
            pickle.dump(data, f)
