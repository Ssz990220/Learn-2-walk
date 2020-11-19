import gym
import pybulletgym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1
import time

if __name__ == '__main__':
    env = gym.make('HumanoidPyBulletEnv-v0')
    model = PPO1.load("_18247680_steps")
    env.render()
    ob = env.reset()
    reward = 0

    while True:
        action, _states = model.predict(ob)
        ob, r, done, info = env.step(action)
        reward += r
        time.sleep(0.05)
        if done:
            ob = env.reset()
            print('r is {}'.format(r))
            print('Episode reward is {}'.format(reward))
            reward = 0