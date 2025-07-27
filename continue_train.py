# === Mario Adaptive Resumed Training Script ===

import os
import cv2
import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym.wrappers import TimeLimit
from multiprocessing import Value, Manager
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack, DummyVecEnv, VecVideoRecorder
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

# === OBS PREPROCESS ===
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]

# === CUSTOM REWARD ===
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, section_width=100, shared_max_section=None, level=None, level_stats=None):
        super().__init__(env)
        self.section_width = section_width
        self.prev_x = 0
        self.level = level
        self.level_stats = level_stats

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
            print(f"🏁 Mario completed the level! ({self.env.unwrapped._world}-{self.env.unwrapped._stage})")
        if done:
            if self.level and self.level_stats is not None:
                stats = self.level_stats[self.level]
                stats['attempted'] += 1
                if info.get('flag_get'):
                    stats['completed'] += 1
                self.level_stats[self.level] = stats
            if not info.get('flag_get'):
                shaped_reward -= 50.0

        return obs, shaped_reward, done, info

# === CALLBACK TO SAVE BEST MODEL AND VIDEO ===
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, save_name, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_name = save_name
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
                        self.model.save(self.save_path)
                        print(f"🔥 New best mean reward: {mean_reward:.2f} -> Model saved!")
                        self.save_rollout_videos(self.num_timesteps)
            except Exception as e:
                print(f"⚠️ Callback error: {e}")
        return True

    def save_rollout_videos(self, current_step):
        levels_to_render = [(1, 1), (1, 2), (1, 3), (1, 4)]
        video_dir = os.path.join(self.log_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

        for world, stage in levels_to_render:
            def make_env():
                env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
                env = JoypadSpace(env, ALLOWED_ACTIONS)
                env = CustomRewardWrapper(env)
                env = MaxAndSkipEnv(env, skip=4)
                env = PreprocessFrame(env)
                env = TimeLimit(env, max_episode_steps=3000 if (world, stage) == (1, 3) else 8000)
                return env

            env = DummyVecEnv([make_env])
            env = VecFrameStack(env, n_stack=4)
            env = VecVideoRecorder(
                env,
                video_folder=video_dir,
                record_video_trigger=lambda step: step == 0,
                video_length=8000,
                name_prefix=f"best_{world}-{stage}_step{current_step}",
            )

            obs = env.reset()
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                if done.any():
                    break
            env.close()

            print(f"🎥 Saved video for level {world}-{stage} at step {current_step}")

# === DYNAMIC SAMPLING ===
def compute_sampling_weights(levels, level_stats, min_weight=0.2):
    weights = []
    for lvl in levels:
        stats = level_stats[lvl]
        rate = stats['completed'] / max(1, stats['attempted'])
        weight = max(min_weight, 1.0 - rate ** 2)
        weights.append(weight)
    return weights

def make_adaptive_env(rank, base_log_dir, default_max_steps, shared_max_section, level_stats, level_counts, weights, seed=0):
    def _init():
        set_random_seed(seed + rank)

        if rank < len(levels):
            level = levels[rank]
        else:
            level = random.choices(levels, weights=weights)[0]

        level_counts[level] += 1
        world, stage = level
        print(f"🤌 Worker {rank} assigned to World-{world}-{stage}")

        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
        env = JoypadSpace(env, ALLOWED_ACTIONS)
        env = CustomRewardWrapper(env, shared_max_section=shared_max_section, level=level, level_stats=level_stats)
        env = MaxAndSkipEnv(env, skip=4)
        env = PreprocessFrame(env)

        custom_max_steps = 3000 if level == (1, 3) else default_max_steps
        env = TimeLimit(env, max_episode_steps=custom_max_steps)

        monitor_dir = os.path.join(base_log_dir, f"monitor_{rank}")
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(monitor_dir, f"monitor.csv"))

        return env
    return _init

# === MAIN ===
if __name__ == "__main__":
    log_dir = "./logs/general_world1_dynamic"
    os.makedirs(log_dir, exist_ok=True)
    num_cpu = 12
    total_timesteps = 20_000_000
    steps_per_chunk = 2_000_000
    default_max_steps = 8000
    levels = [(1, 1), (1, 2), (1, 3), (1, 4)]
    shared_max_section = Value('i', 0)

    manager = Manager()
    level_stats = manager.dict({lvl: {'completed': 0, 'attempted': 1} for lvl in levels})
    level_counts = manager.dict({lvl: 0 for lvl in levels})

    weights = compute_sampling_weights(levels, level_stats)
    env_fns = [
        make_adaptive_env(i, log_dir, default_max_steps, shared_max_section, level_stats, level_counts, weights)
        for i in range(num_cpu)
    ]
    vec_env = VecMonitor(SubprocVecEnv(env_fns), filename=log_dir)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # === LOAD previous best model and override entropy ===
    model_path = os.path.join(log_dir, "best_model_world1_dynamic.zip")
    model = PPO.load(
        model_path,
        env=vec_env,
        device="cuda" if torch.cuda.is_available() else "cpu",
        custom_objects={"ent_coef": 0.05}  # Increase exploration
    )

    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000,
        log_dir=log_dir,
        save_name="best_model_world1_dynamic_resumed",
    )

    for chunk in range(total_timesteps // steps_per_chunk):
        print(f"⏱️ Training chunk {chunk+1}/{total_timesteps // steps_per_chunk}")

        model.learn(
            total_timesteps=steps_per_chunk,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="PPO-World1-Dynamic2-Resumed"
        )
        model.save(os.path.join(log_dir, f"ppo_world1_dynamic_resumed_step{(chunk+1)*steps_per_chunk}"))

        print("\n📊 Level Completion Rates:")
        all_solved = True
        for lvl in levels:
            stats = level_stats[lvl]
            rate = stats['completed'] / max(1, stats['attempted'])
            print(f"  Level {lvl}: {stats['completed']} / {stats['attempted']} → {rate:.2%}")
            if rate < 0.95:
                all_solved = False

        if all_solved:
            print("🚩 Early stopping: All levels solved >95% consistently.")
            break

        print("\n📊 Level Sampling Counts:")
        for lvl in levels:
            print(f"  Level {lvl}: {level_counts[lvl]} times")

        for lvl in levels:
            level_stats[lvl] = {'completed': 0, 'attempted': 1}

        weights = compute_sampling_weights(levels, level_stats)
        env_fns = [
            make_adaptive_env(i, log_dir, default_max_steps, shared_max_section, level_stats, level_counts, weights)
            for i in range(num_cpu)
        ]
        vec_env = VecMonitor(SubprocVecEnv(env_fns), filename=log_dir)
        vec_env = VecFrameStack(vec_env, n_stack=4)
        model.set_env(vec_env)

    model.save(os.path.join(log_dir, "ppo_final_world1_dynamic_resumed2"))
    print("✅ Resumed training complete")
