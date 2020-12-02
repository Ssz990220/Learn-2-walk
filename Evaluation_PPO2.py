import gym
import pybulletgym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv
import time

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

if __name__ == '__main__':
    env_id = "HumanoidPyBulletEnv-v0"
    num_cpu = 1
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    model = PPO2.load("HumanoidPyBulletEnv-v0_PPO2_2020_11_3016_29_44")
    ob = env.reset()
    reward = 0

    while True:
        action, _states = model.predict(ob)
        ob, r, done, info = env.step(action)
        reward += r
        time.sleep(0.01)
        if done:
            ob = env.reset()
            print('r is {}'.format(r))
            print('Episode reward is {}'.format(reward))
            reward = 0