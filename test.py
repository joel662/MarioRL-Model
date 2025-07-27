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
from multiprocessing import Value

# ==== GPU Check ====
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print("\u2705 Using GPU:", torch.cuda.get_device_name(0))
else:
    print("\u26A0\uFE0F CUDA not available, using CPU")

ALLOWED_ACTIONS = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['down'],
    ['down', 'right']
]


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]

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
            print("🏁 Mario completed the level!")

        if done and not info.get('flag_get'):
            shaped_reward -= 50.0

        return obs, shaped_reward, done, info

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
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"\U0001F525 New best mean reward: {mean_reward:.2f}. Saving model.")
                    self.model.save(self.save_path)
        return True

def make_env(world, stage, rank, log_dir, max_steps, shared_max_section, seed=0):
    def _init():
        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
        env = JoypadSpace(env, ALLOWED_ACTIONS)
        env = CustomRewardWrapper(env, shared_max_section=shared_max_section)
        env = MaxAndSkipEnv(env, skip=4)
        env = PreprocessFrame(env)
        env = TimeLimit(env, max_episode_steps=max_steps)
        env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    num_cpu = 12
    total_timesteps = 20_000_000
    steps_per_chunk = 5_000_000
    max_steps = 5000
    worlds_and_stages = [ (1, 3)]

    prev_world = None
    prev_stage = None

    for world, stage in worlds_and_stages:
        try:
            print(f"\n\U0001F3AE Training on World {world}-{stage}")

            global_max_section = Value('i', 0)  # Shared memory counter

            env_fns = [
                make_env(world, stage, i, log_dir, max_steps, global_max_section)
                for i in range(num_cpu)
            ]
            env = VecMonitor(SubprocVecEnv(env_fns))
            env = VecFrameStack(env, n_stack=4)

            model = PPO(
                "CnnPolicy", env,
                verbose=1,
                tensorboard_log="./tb_logs/",
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=4,
                clip_range=0.1,
                gae_lambda=0.95,
                ent_coef=0.01,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            if prev_world is not None and prev_stage is not None:
                prev_model_path = os.path.join(log_dir, f"ppo_final_W{prev_world}-{prev_stage}.zip")
                if os.path.exists(prev_model_path):
                    print(f"\U0001F4E6 Loading weights from World {prev_world}-{prev_stage}")
                    model.set_parameters(prev_model_path)
                else:
                    print(f"\u26A0\uFE0F Previous model not found: {prev_model_path}")

            callback = SaveOnBestTrainingRewardCallback(
                check_freq=1000,
                log_dir=log_dir,
                save_name=f"best_model_W{world}-{stage}"
            )

            for chunk in range(total_timesteps // steps_per_chunk):
                print(f"\u23F1\uFE0F Chunk {chunk+1} of {total_timesteps // steps_per_chunk}")
                model.learn(
                    total_timesteps=steps_per_chunk,
                    callback=callback,
                    reset_num_timesteps=False,
                    tb_log_name=f"PPO-W{world}-{stage}.2"
                )
                chunk_save_path = os.path.join(log_dir, f"ppo_W{world}-{stage}_step{(chunk+1)*steps_per_chunk}")
                model.save(chunk_save_path)
                print(f"\U0001F4BE Saved intermediate model: {chunk_save_path}")

            model.save(os.path.join(log_dir, f"ppo_final_W{world}-{stage}"))
            print(f"\u2705 Done training World {world}-{stage}")

            prev_world = world
            prev_stage = stage

        except Exception as e:
            print(f"\u274C Failed training World {world}-{stage}: {e}")