import os
import cv2
import gym
import torch
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

# === Setup ===
ALLOWED_ACTIONS = [
    ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'],
    ['A'], ['down'], ['down', 'right']
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
    def __init__(self, env, stats):
        super().__init__(env)
        self.prev_x = 0
        self.stats = stats

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
            self.stats['completed'] += 1
            print("🏁 Mario completed the level!")
        if done:
            self.stats['attempted'] += 1
            if not info.get('flag_get'):
                shaped_reward -= 50.0
        return obs, shaped_reward, done, info

class SaveBestCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            try:
                x, y = ts2xy(load_results(os.path.dirname(self.save_path)), 'timesteps')
                if len(y) > 0:
                    mean_reward = np.mean(y[-100:])
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.model.save(self.save_path)
                        print(f"🔥 New best mean reward: {mean_reward:.2f}. Model saved.")
            except Exception as e:
                print(f"Callback error: {e}")
        return True

def make_env(world, stage, stats):
    env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
    env = JoypadSpace(env, ALLOWED_ACTIONS)
    env = CustomRewardWrapper(env, stats=stats)
    env = MaxAndSkipEnv(env, skip=4)
    env = PreprocessFrame(env)
    env = TimeLimit(env, max_episode_steps=8000)
    env = Monitor(env, f"./logs/mario_{world}{stage}")
    return DummyVecEnv([lambda: env])

if __name__ == "__main__":
    levels = [(1, 1), (1, 2), (1, 3), (1, 4)]
    total_timesteps = 8_000_000
    curriculum_threshold = 0.95
    max_epochs_per_level = 5

    for world, stage in levels:
        print(f"\n📘 Starting curriculum training for World-{world} Stage-{stage}")
        stats = {'completed': 0, 'attempted': 1}  # to avoid divide by zero
        env = make_env(world, stage, stats=stats)
        env = VecFrameStack(env, n_stack=4)

        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tb_logs/",
                    learning_rate=1e-4, n_steps=2048, batch_size=256,
                    n_epochs=4, clip_range=0.1, gae_lambda=0.95,
                    ent_coef=0.01, device="cuda" if torch.cuda.is_available() else "cpu")

        save_path = f"./models/mario_{world}{stage}_best"
        callback = SaveBestCallback(check_freq=1000, save_path=save_path)

        # Curriculum loop: stop early if level mastered
        for epoch in range(max_epochs_per_level):
            print(f"📈 Epoch {epoch + 1}/{max_epochs_per_level}")
            stats['completed'], stats['attempted'] = 0, 1
            model.learn(total_timesteps=2_000_000, callback=callback)

            completion_rate = stats['completed'] / max(1, stats['attempted'])
            print(f"✅ Completion rate: {completion_rate:.2%}")

            if completion_rate >= curriculum_threshold:
                print(f"🎓 Level {world}-{stage} mastered! Moving to next level.\n")
                break
            else:
                print("🔁 Continuing training for this level...\n")

        model.save(f"./models/mario_{world}{stage}_final")
        print(f"🏁 Saved final model for World-{world} Stage-{stage}")
