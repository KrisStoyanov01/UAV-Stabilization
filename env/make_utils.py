import gym
from gym.envs.registration import register

ENV_ID = {
    "Drone_3D": "Drone_3D-v0",
    "Drone_kris": "Drone_kris-v0",

}

ENV_CLASS = {
    "Drone_3D": "Drone_3D",
    "Drone_kris": "Drone_kris",
}


def register_env(env_name):
    env_id = ENV_ID[env_name]
    env_class = ENV_CLASS[env_name]
    register(id=env_id, entry_point="env." + env_name + ":" + env_class)


def make_env(env_name):
    env_id = ENV_ID[env_name]
    return gym.make(env_id)
