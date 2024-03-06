import gym
import numpy as np
import numpy.linalg as LA
import pybullet as p
import random
from gym import Env, spaces, utils
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym.envs.registration import register


START_POS = np.array([[0.0, 0.0, 0.2]])
ALLOWED_OFFSET = 0.15

# controlls the frequency
DIV = 8

# the model can be considere trained when
# a specific proportions of the previous iterations
# are successful
SUCCESS_PROPORTION = 0.95
FAILURE_PROPORTION = 1.0 - SUCCESS_PROPORTION
PREVIOUS_CHECKED_ITERATIONS = 100
MODEL_FINISHED_TRAINING = False


class Drone_kris(Env, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        simulation_freq_hz = 240 / DIV
        control_freq_hz = 240 / DIV
        AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz)
        INIT_XYZS = START_POS
        INIT_RPYS = np.array([[0, 0, 0]])
        PHY = Physics.PYB
        self.drone_env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            physics=PHY,
            neighbourhood_radius=10,
            freq=simulation_freq_hz,
            aggregate_phy_steps=AGGR_PHY_STEPS,
            gui=True,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        self.episode_steps = 0
        self._max_episode_steps = 100
        self.consecutive_successes_count = 0
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()

        self.obs_drone_state = None

        self.i_episode = 0

    def step(self, action):
        # count one more made step
        self.episode_steps += 1

        # wind strength
        wind_strength = random.uniform(0.0001, 0.0005)

        # wind direction
        wind_direction = np.array([1, 2, 1])

        # calculate the displacement of the drone
        displacement = wind_strength * wind_direction

        action = action.astype(float)
        action += displacement

        old_state = self.obs_drone_state.copy()
        action_mag_per = min(
            LA.norm(action), self.drone_env.SPEED_LIMIT) / self.drone_env.SPEED_LIMIT
        drone_action = {str(0): np.array(
            [action[0], action[1], action[2], action_mag_per])}

        for _ in range(5):
            obs, reward, done, info = self.drone_env.step(drone_action)
            self.obs_drone_state = self.get_obs(obs)
            success_flag = self.success()
            unsafe_flag = self.unsafe()
            done = False
            if success_flag or unsafe_flag:
                done = True
                break

        pos = np.array(
            [self.obs_drone_state[0], self.obs_drone_state[1], self.obs_drone_state[2]])

        reward = - 20*np.sqrt((pos[0]-START_POS[0][0])**2 +
                              (pos[1]-START_POS[0][1])**2 +
                              (pos[2]-START_POS[0][2])**2)

        if success_flag:
            reward += 8000

        if unsafe_flag:
            self.obs_drone_state = old_state

        info = {
            "constraint": unsafe_flag,
            "reward": reward,
            "state": old_state,
            "next_state": self.obs_drone_state,
            "action": action,
            "success": success_flag,
        }
        return self.obs_drone_state, reward, done, info

    def reset(self):

        # a new episode starts, so reset the step count
        self.episode_steps = 0

        self.drone_env.INIT_XYZS = START_POS.reshape(1, -1)
        obs = self.drone_env.reset()
        self.obs_drone_state = self.get_obs(obs)
        return self.obs_drone_state.copy()

    def unsafe(self):

        x = self.obs_drone_state[0]
        y = self.obs_drone_state[1]
        z = self.obs_drone_state[2]

        # box around the scene
        if x <= -0.8 or x >= 0.8 or y <= -0.8 or y >= 0.8 or z <= 0.015 or z >= 0.75:
            return True

        #  check if the drone is too far from the original position
        if LA.norm(self.obs_drone_state[:3] - START_POS) > ALLOWED_OFFSET:
            return True

        return False

    def success(self):

        # check if we have kept stable for the needed number of steps
        if (self.episode_steps >= self._max_episode_steps):
            self.consecutive_successes_count += 1
            if (self.consecutive_successes_count >= round(SUCCESS_PROPORTION * PREVIOUS_CHECKED_ITERATIONS)):
                MODEL_FINISHED_TRAINING = True
                print('-------------MODEL CAN BE CONSIDERED TRAINED-------------')
            return True

        return False

    def get_action_space(self):
        act_lower_bound = np.array([-0.25, -0.25, -0.25])
        act_upper_bound = np.array([0.25, 0.25, 0.25])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def get_observation_space(self):
        obs_lower_bound = np.array([-0.8, -0.8, 0.015, -1, -1, -1])
        obs_upper_bound = np.array([0.8, 0.8, 0.75, 1, 1, 1])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def get_obs(self, obs):
        obs_drone_state = np.zeros(6)
        obs_drone_state[0] = obs[str(0)]["state"][0]
        obs_drone_state[1] = obs[str(0)]["state"][1]
        obs_drone_state[2] = obs[str(0)]["state"][2]
        obs_drone_state[3] = obs[str(0)]["state"][10]
        obs_drone_state[4] = obs[str(0)]["state"][11]
        obs_drone_state[5] = obs[str(0)]["state"][12]
        return obs_drone_state


if __name__ == "__main__":
    register(id="d3d-v1", entry_point="env.Drone_kris:Drone_kris")
    env = gym.make('d3d-v1')
    obs = env.reset()
    rew = 0

    for i in range(100000):
        obs, reward, done, info = env.step(np.array([0, 0, 0]))
        rew += reward

        if done:
            print(obs)
            break
