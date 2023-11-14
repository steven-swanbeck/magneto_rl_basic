#!/usr/bin/env python3
# %%
import numpy as np
from magneto_env import MagnetoEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# %%

# def make_env(env_id: str, rank: int, seed: int = 0):
def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        # env = gym.make(env_id, render_mode="human")
        env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=10)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# %%
def train_ppo (env, path, rel_path, timesteps):
    # . Training    
    checkpoint_callback = CheckpointCallback(
        # save_freq=10,
        save_freq=100000,
        save_path=path + rel_path + 'weights/',
        name_prefix='magneto',
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # - Start from scratch or load specified weights
    # model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    model = RecurrentPPO("MlpLstmPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # model = PPO.load(path + rel_path + 'breakpoint.zip', env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    print(model.policy)
    
    # # - Training
    try:
        model.learn(total_timesteps=timesteps, callback=checkpoint_callback, progress_bar=True)
        
        for i in range(10):
            obs, _ = env.reset()
            over = False
            counter = 0
            while not over:
                action, _states = model.predict(obs)
                obs, rewards, over, _, _ = env.step(action)
                env.render()
                counter += 1
        env.close()
        
    finally:
        # pass
        model.save(path + rel_path + 'breakpoint.zip')

def main ():
    path = '/home/steven/magneto_ws/outputs/'
    env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=10)
    # rel_path = 'leader-follower/no_magnetic_seeds/2_seeding_magnetism/'
    rel_path = 'recurrent/leader_follower/vec_env/'
    
    # num_cpu = 4
    # vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # . Training
    # train_ppo(env, path, rel_path, 20000000)
    train_ppo(env, path, rel_path, 100000)
    # train_ppo(vec_env, path, rel_path, 100000)
    

# %%
if __name__ == "__main__":
    main()

# %%
