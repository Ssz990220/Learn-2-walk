import gym
import pybulletgym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import SAC
import time

if __name__ == '__main__':
    env = gym.make('Walker2DPyBulletEnv-v0')
    model = SAC.load("_7880000_steps")
    env.render()
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