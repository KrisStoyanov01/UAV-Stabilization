import pickle
import os
import matplotlib.pyplot as plt

EXPERIMENT = 'Reward_10_over_distance_Drone_3D_LBAC_draw_clb'

PATH = './training_logs/' + EXPERIMENT + '/drone_xz'

if __name__ == '__main__':
    dirs = os.walk(PATH)

    for dir in dirs:
        rewards = []
        try:
            with open(dir[0] + '/run_stats.pkl', 'rb') as f:
                data = pickle.load(f)
                for stats in data['train_stats']:
                    for entry in stats:
                        rewards.append(entry['reward'])
            plt.plot(rewards)
            plt.show()
        except:
            pass
