import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
import sys
sys.path.append("../")
from plotting import plotting_utils

class Eval_Plotter:
    def __init__(self, all_traj_info, alg):
        self.all_traj_info = all_traj_info
        self.alg = alg
    
    def plot_drone_traj(self):
        pos_err_multi = []
        theta_err_multi = []
        for test_rollout_info in self.all_traj_info:
            num_steps = len(test_rollout_info)
            x = []
            z = []
            for i in range(num_steps):
                x.append(test_rollout_info[i]["state"][0])
                z.append(test_rollout_info[i]["state"][1])
            x = np.array(x)
            z = np.array(z)
        fig, axs = plt.subplots(1, figsize=(8, 8))
        axs.plot(x, z)
        plt.show()
    
    def plot_car_state(self, idx):
        pos_err_multi = []
        theta_err_multi = []
        for test_rollout_info in self.all_traj_info:
            num_steps = len(test_rollout_info)
            x_e = []
            y_e = []
            theta_e = []
            for i in range(num_steps):
                x_e.append(test_rollout_info[i]["state"][0])
                y_e.append(test_rollout_info[i]["state"][1])
                theta_e.append(test_rollout_info[i]["state"][2])
            x_e = np.array(x_e)
            y_e = np.array(y_e)
            pos_e = LA.norm(np.array([x_e, y_e]), axis=0)
            theta_e = np.array(theta_e)
            pos_err_multi.append(pos_e)
            theta_err_multi.append(theta_e)
        data = [np.array(pos_err_multi), np.array(theta_err_multi)]
        fig, axs = plt.subplots(1, figsize=(16, 8))
        err_mean, err_lb, err_ub = plotting_utils.get_stats(data[idx])
        axs.fill_between(
            range(err_mean.shape[0]),
            err_lb,
            err_ub,
            alpha = 0.25,
            label = self.alg
        )
        axs.plot(err_mean)
        axs.set_xlabel("Time Step", fontsize=42)
        axs.set_ylabel("State", fontsize=42)
        plt.show()
    
    def plot_action(self, idx):
        idx_0_multi = []
        idx_1_multi = []
        for test_rollout_info in self.all_traj_info:
            num_steps = len(test_rollout_info)
            idx_0 = []
            idx_1 = []
            for i in range(num_steps):
                idx_0.append(test_rollout_info[i]["action"][0])
                idx_1.append(test_rollout_info[i]["action"][1])
            idx_0_multi.append(np.array(idx_0))
            idx_1_multi.append(np.array(idx_1))
        
        data = [np.array(idx_0_multi), np.array(idx_1_multi)]
        fig, axs = plt.subplots(1, figsize=(16, 8))
        err_mean, err_lb, err_ub = plotting_utils.get_stats(data[idx])
        axs.fill_between(
            range(err_mean.shape[0]),
            err_lb,
            err_ub,
            alpha = 0.25,
            label = self.alg
        )
        axs.plot(err_mean)
        axs.set_xlabel("Time Step", fontsize=42)
        axs.set_ylabel("Action", fontsize=42)
        # axs.set_ylim([-0.4, 0.4])
        plt.show()


        


    