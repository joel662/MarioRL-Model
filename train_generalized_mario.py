# === Fully Updated Mario Training Script with Fixed Callback ===

import os
import numpy as np
import cv2
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from gym.wrappers import TimeLimit
import torch
import random
from multiprocessing import Value

# Allowed Actions
ALLOWED_ACTIONS = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['down'],
    ['down', 'right']
]

# Preprocess grayscale + resize
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]

# Reward shaping
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, section_width=100, shared_max_section=None):
        super().__init__(env)
        self.section_width = section_width
        self.prev_x = 0

    def reset(self, **kwargs):
        self.prev_x = 0
        self.visited_sections = set()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        curr_x = info.get('x_pos', 0)
        delta_x = max(0, curr_x - self.prev_x)
        self.prev_x = curr_x

        shaped_reward = delta_x * 0.1
        if info.get('flag_get'):
            shaped_reward += 300.0 + info.get('time', 0) * 0.05
            print("\U0001F3C1 Mario completed the level!")
        if done and not info.get('flag_get'):
            shaped_reward -= 50.0

        return obs, shaped_reward, done, info

# Save best model callback
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, save_name, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, f"{save_name}.zip")
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            try:
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                print(f"\U0001F9EA Step {self.n_calls}: Loaded {len(y)} rewards")
                if len(y) > 0:
                    mean_reward = np.mean(y[-100:])
                    print(f"\U0001F4CA Mean: {mean_reward:.2f} | Best: {self.best_mean_reward:.2f}")
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print(f"\U0001F525 New best mean reward: {mean_reward:.2f}. Saving model.")
                        self.model.save(self.save_path)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")
        return True

# Dynamic env generator

def make_dynamic_env(levels, rank, base_log_dir, max_steps, shared_max_section, seed=0):
    def _init():
        set_random_seed(seed + rank)
        world, stage = random.choice(levels)
        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
        env = JoypadSpace(env, ALLOWED_ACTIONS)
        env = CustomRewardWrapper(env, shared_max_section=shared_max_section)
        env = MaxAndSkipEnv(env, skip=4)
        env = PreprocessFrame(env)
        env = TimeLimit(env, max_episode_steps=max_steps)
        monitor_dir = os.path.join(base_log_dir, f"monitor_{rank}")
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
        return env
    return _init

# === Training Script ===
if __name__ == "__main__":
    log_dir = "./logs/general_world1"
    os.makedirs(log_dir, exist_ok=True)

    num_cpu = 12
    total_timesteps = 25_000_000
    steps_per_chunk = 2_500_000
    max_steps = 3000
    levels = [(1, 1), (1, 2), (1, 3), (1, 4)]
    shared_max_section = Value('i', 0)

    print("\U0001F3AE Initializing multi-level VecEnv (World 1 only)...")
    env_fns = [
        make_dynamic_env(levels, i, log_dir, max_steps, shared_max_section)
        for i in range(num_cpu)
    ]
    env = VecMonitor(SubprocVecEnv(env_fns), filename=log_dir)
    env = VecFrameStack(env, n_stack=4)

    print("\U0001F680 Creating PPO model")
    model = PPO(
        "CnnPolicy", env,
        verbose=1,
        tensorboard_log="./tb_logs/general_world1/",
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        clip_range=0.1,
        gae_lambda=0.95,
        ent_coef=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,
        log_dir=log_dir,
        save_name="best_model_world1"
    )

    for chunk in range(total_timesteps // steps_per_chunk):
        print(f"⏱️ Training chunk {chunk+1}")
        model.learn(
            total_timesteps=steps_per_chunk,
            callback=callback,
            reset_num_timesteps=(chunk == 0),
            tb_log_name="PPO-World1-Generalized"
        )
        model.save(os.path.join(log_dir, f"ppo_world1_step{(chunk+1)*steps_per_chunk}"))
        print("\U0001F4BE Model saved")

    model.save(os.path.join(log_dir, "ppo_final_world1"))
    print("✅ Finished training generalized agent for World 1")
