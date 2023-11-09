#!/usr/bin/env python3
# %%
from magneto_env import MagnetoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from magneto_policy_learner import CustomActorCriticPolicy

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
    model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./magneto_tensorboard/")
    # model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./magneto_tensorboard/")
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
        model.save(path + rel_path + 'breakpoint.zip')

def main ():
    path = '/home/steven/magneto_ws/outputs/'
    env = MagnetoEnv(render_mode="human", sim_mode="grid", magnetic_seeds=5)
    rel_path = 'lstm/'
    
    # . Training
    train_ppo(env, path, rel_path, 300000)

if __name__ == "__main__":
    main()

# %%
