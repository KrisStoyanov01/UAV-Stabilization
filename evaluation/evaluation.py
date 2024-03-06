import sys
import time

import eva_agents
import matplotlib.pyplot as plt
import numpy as np
import plot_eval
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from env.make_utils import make_env, register_env

sys.path.append(
    "/home/kris/drone-final/")
print(sys.path)


class Evaluator():
    def __init__(self, env_name, distur_test, print_one_trail_info):
        register_env(env_name)
        self.env = make_env(env_name)
        self.num_success = 0
        self.num_violation = 0
        self.agent = None
        self.get_action_time = 0
        self.distur_test = distur_test
        self.print_one_trail_info = print_one_trail_info

    def evaluation(self, num_episodes):
        self.num_success = 0
        self.num_violation = 0
        all_traj_info = []
        for i_episode in range(num_episodes):
            all_traj_info.append(self.get_test_rollout(i_episode))
        print("Success Rate: {}".format(self.num_success/num_episodes))
        print("Violation Rate: {}".format(self.num_violation/num_episodes))
        return all_traj_info

    def get_test_rollout(self, i_episode):
        avg_reward = 0.
        test_rollout_info = []
        state = self.env.reset()

        episode_reward = 0
        episode_steps = 0
        done = False

        if self.distur_test:
            self.env.distur_test = True
        cnt = 0
        while not done:
            cnt += 1
            now = time.time()
            action = self.agent.get_action(state)
            if self.distur_test and cnt % 20 == 0:
                action_sign = np.sign(action)
                action[0] = -11 * action_sign[0]
                action[1] = -2*np.pi * action_sign[1]

            self.get_action_time = max(time.time() - now, self.get_action_time)
            next_state, reward, done, info = self.env.step(action)  # Step
            done = done or episode_steps == self.env._max_episode_steps
            test_rollout_info.append(info)
            episode_reward += reward
            episode_steps += 1
            state = next_state
        avg_reward += episode_reward

        if info["constraint"]:
            self.num_violation += 1
        elif info["success"]:
            self.num_success += 1
        if self.print_one_trail_info:
            print("----------------------------------------")
            print("Test Rollout: {}".format(i_episode))
            print("Avg. Reward: {}".format(round(avg_reward, 2)))
            print("Final state: {}".format(state))
            print("Get action time: {}ms".format(self.get_action_time * 1000))
            if info["constraint"]:
                print("Terminate: Violation")
            elif info["success"]:
                print("Terminate: Success")
            else:
                print("Terminate: Timeout")
            print("----------------------------------------")
        return test_rollout_info


name = "kine_car" or "drone_xz"

VERSION = '0'
EXPERIMENT = 'kris' + VERSION + '_Drone_kris_LBAC_draw_clb'
ITERATION = 4000
SHAPE = 'sphere'

# radius of the sphere
R = 0.0

policy_map = {"CLF_CBF": [
    'policy_' + EXPERIMENT + '_' + str(ITERATION) + '_seed111111']}

dir = "/home/kris/drone-final/saved_model/" + EXPERIMENT + '/'

env_name = "Drone_kris"
distur_test = False
print_one_trail_info = True
num_episodes = 1


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


if __name__ == "__main__":
    evaluator = Evaluator(env_name=env_name, distur_test=distur_test,
                          print_one_trail_info=print_one_trail_info)
    for alg in policy_map:
        print("Evaluating: {}".format(alg))
        if alg == "MPC":
            if (env_name == "kine_car"):
                policy_map[alg].append(
                    eva_agents.Car_NMPC_Agent(control_freq=10))
            elif (env_name == "drone_xz"):
                policy_map[alg].append(
                    eva_agents.Drone_MPC_Agent(control_freq=10))
        elif alg == "CLF_CBF" and env_name == "drone_xz":
            policy_map[alg].append(
                eva_agents.Drone_CLF_CBF_Agent(control_freq=10))
        else:
            PATH = dir + policy_map[alg][0] + ".pkl"
            agent = eva_agents.RL_Agent(PATH)
            policy_map[alg].append(agent)
        evaluator.agent = policy_map[alg][-1]
        all_traj_info = evaluator.evaluation(num_episodes)
        # eval_plotter = plot_eval.Eval_Plotter(all_traj_info, policy_map[alg][0])
        # eval_plotter.plot_drone_traj()
        for test_rollout_info in all_traj_info:
            num_steps = len(test_rollout_info)
            x = []
            y = []
            z = []
            for i in range(num_steps):
                x.append(test_rollout_info[i]["state"][0])
                y.append(test_rollout_info[i]["state"][1])
                z.append(test_rollout_info[i]["state"][2])
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.grid()

        ax.scatter(-0.5, 0, 0.5, c='r', s=80)
        ax.scatter(0.5, 0, 0.5, c='g', s=80)

        ax.scatter(x, y, z, c='b', s=50)
        ax.set_title('3D Scatter Plot')

        if SHAPE == 'cylinder':
            Z = np.array([[-0.35, 0.35, 0], [0.35, -
                                             0.35, 0], [-0.35, -0.35, 0], [0.35, 0.35, 0], [-0.35, 0.35, 2], [0.35, -
                                                                                                              0.35, 2], [-0.35, -0.35, 2], [0.35, 0.35, 2]])


            faces = [[Z[0], Z[1], Z[2], Z[3]],
                     [Z[4], Z[5], Z[6], Z[7]],
                     [Z[0], Z[1], Z[5], Z[4]],
                     [Z[2], Z[3], Z[7], Z[6]],
                     [Z[1], Z[2], Z[6], Z[5]],
                     [Z[4], Z[7], Z[3], Z[0]]]


            Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, 0.1, 0.75)
            ax.plot_surface(Xc, Yc, Zc, alpha=0.5)
        elif SHAPE == 'sphere':
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = R*np.cos(u)*np.sin(v)
            y = R*np.sin(u)*np.sin(v)
            z = R*np.cos(v)

            # Plot the sphere
            ax.plot_surface(x, y, z, cmap=plt.cm.Blues)

        ax.set_xlabel('x', labelpad=20)
        ax.set_ylabel('y', labelpad=20)
        ax.set_zlabel('z', labelpad=20)
        plt.xlim(-0.8, 0.8)
        plt.ylim(-0.8, 0.8)

        plt.show()
