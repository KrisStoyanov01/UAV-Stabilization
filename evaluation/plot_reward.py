import numpy as np
import matplotlib.pyplot as plt

lower_bound = [-1.2, -1.2, 0]
upper_bound = [1.2, 1.2, 1]
R = 0.25

GOAL = [1, 0, 0.5]
Z = 0.25
NUM_DIVISIONS = 100
PENALTY = 1


def penalize(x, y):
    rew = 0

    if x ** 2 + y ** 2 - R ** 2 <= 0:
        rew -= PENALTY
    if x <= -1.2 or x >= 1.2 or y <= -1.2 or y >= 1.2 or Z <= 0.015 or Z >= 1:
        rew -= PENALTY

    return rew


def reward(x, y):
    return penalize(x, y) - np.sqrt((x-GOAL[0])**2+5*(y-GOAL[1])**2+(Z-GOAL[2])**2)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    Xs, Ys, Zs = [], [], []
    step = (upper_bound[0] - lower_bound[0]) / (NUM_DIVISIONS-1)

    for i in range(NUM_DIVISIONS):
        for j in range(NUM_DIVISIONS):
            x = lower_bound[0] + i * step
            y = lower_bound[1] + j * step

            Xs.append(x)
            Ys.append(y)
            Zs.append(reward(x, y))

    ax.scatter(Xs, Ys, Zs)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
