import os
from typing import Union, Optional, Callable, Tuple, List

import gym
import pybulletgym
import numpy as np
import wandb
# from rrc_simulation.grasping_envs.SlideEnv import PickEnv
from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines.common.vec_env import sync_envs_normalization, VecEnv
from mpi4py import MPI


class EvalCallback_wandb(EvalCallback):

    def __init__(self, eval_env: Union[gym.Env, VecEnv],
                 callback_on_new_best: Optional[BaseCallback] = None,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 log_path: str = None,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):

        super(EvalCallback_wandb, self).__init__(eval_env=eval_env, callback_on_new_best=callback_on_new_best,
                                                 n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
                                                 log_path=log_path,
                                                 best_model_save_path=best_model_save_path, deterministic=deterministic,
                                                 render=render,
                                                 verbose=verbose)

    def _on_step(self) -> bool:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.rank = rank
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and rank == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            episode_rewards, episode_lengths = evaluate_policy(self.model, self.eval_env,
                                                               n_eval_episodes=self.n_eval_episodes,
                                                               render=self.render,
                                                               deterministic=self.deterministic,
                                                               return_episode_rewards=True,
                                                               rank=self.rank)
            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results, ep_lengths=self.evaluations_length)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print("Eval num_timesteps={}, "
                      "episode_reward={:.2f} +/- {:.2f}".format(self.num_timesteps, mean_reward, std_reward))
                print("Episode length: {:.2f} +/- {:.2f}".format(mean_ep_length, std_ep_length))

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
        else:
            pass

        return True


def evaluate_policy(
        model: "BaseRLModel",
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
        rank: int = 0
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
    episode_rewards, episode_lengths = [], []
    success = 0
    drop = 0
    time_exceed = 0
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_episode_length = np.mean(episode_lengths)
    wandb.log({'mean Episode Length': mean_episode_length})
    mean_reward = np.mean(episode_rewards)
    wandb.log({'reward': mean_reward})
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: {:.2f} < {:.2f}".format(mean_reward,
                                                                                                     reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
