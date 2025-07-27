import os
import cv2
import gym
import torch
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym.wrappers import TimeLimit
from multiprocessing import Value
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

# === ACTIONS ===
ALLOWED_ACTIONS = [
    ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'],
    ['A'], ['down'], ['down', 'right']
]

# === Frame Preprocessing ===
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]

# === Reward Shaping ===
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_x = 0

    def reset(self, **kwargs):
        self.prev_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        curr_x = info.get('x_pos', 0)
        delta_x = max(0, curr_x - self.prev_x)
        self.prev_x = curr_x

        shaped_reward = delta_x * 0.1
        if info.get('flag_get'):
            shaped_reward += 300.0 + info.get('time', 0) * 0.05
            print("🏁 Mario completed the level!")
        if done and not info.get('flag_get'):
            shaped_reward -= 50.0
        return obs, shaped_reward, done, info

# === Save Best Model Callback ===
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
                if len(y) > 0:
                    mean_reward = np.mean(y[-100:])
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        print(f"🔥 New best mean reward: {mean_reward:.2f}. Saving model.")
                        self.model.save(self.save_path)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")
        return True

# === Create a single environment ===
def make_env(world, stage, rank, log_dir, max_steps, seed=0):
    def _init():
        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
        env = JoypadSpace(env, ALLOWED_ACTIONS)
        env = CustomRewardWrapper(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = PreprocessFrame(env)
        env = TimeLimit(env, max_episode_steps=max_steps)
        monitor_path = os.path.join(log_dir, f"monitor_{rank}")
        os.makedirs(os.path.dirname(monitor_path), exist_ok=True)
        env = Monitor(env, monitor_path)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# === MAIN ===
if __name__ == "__main__":
    log_dir = "./logs/general_world1_dynamic"
    model_path = os.path.join(log_dir, "best_model_world1_dynamic.zip")

    num_cpu = 12
    steps_per_chunk = 2_000_000
    total_timesteps = 6_000_000
    max_steps = 8000
    levels = [(1, 1), (1, 2), (1, 3), (1, 4)]

    # Assign workers to levels round-robin
    env_fns = [
        make_env(*levels[i % len(levels)], i, log_dir, max_steps)
        for i in range(num_cpu)
    ]

    vec_env = VecFrameStack(VecMonitor(SubprocVecEnv(env_fns)), n_stack=4)

    # === Load model and override entropy coefficient ===
    model = PPO.load(
        model_path,
        env=vec_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        custom_objects={"ent_coef": 0.05}  
    )

    # === Callback ===
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,
        log_dir=log_dir,
        save_name="best_model_world1_dynamic_resumed"
    )

    # === Resume training ===
    for chunk in range(total_timesteps // steps_per_chunk):
        print(f"\n⏱️ Training chunk {chunk+1}/{total_timesteps // steps_per_chunk}")
        model.learn(
            total_timesteps=steps_per_chunk,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="PPO-World1-Dynamic-Resumed"
        )
        chunk_save_path = os.path.join(log_dir, f"ppo_world1_dynamic_resumed_step{(chunk+1)*steps_per_chunk}")
        model.save(chunk_save_path)
        print(f"💾 Saved model chunk: {chunk_save_path}")

    final_path = os.path.join(log_dir, "ppo_final_world1_dynamic_resumed")
    model.save(final_path)
    print("✅ Resumed training complete.")
