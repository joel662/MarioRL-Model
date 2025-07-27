import gym
import gym_super_mario_bros
import cv2
import numpy as np
import torch
from tqdm import trange
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym.wrappers import TimeLimit

# --- Action set ---
ALLOWED_ACTIONS = [
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['down'],
    ['down', 'right']
]

# --- Grayscale + Resize frame ---
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None]

# --- Build a correctly wrapped evaluation env ---
def make_eval_env(world, stage, max_steps=10000):
    def _init():
        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0")
        env = JoypadSpace(env, ALLOWED_ACTIONS)
        env = MaxAndSkipEnv(env, skip=4)
        env = PreprocessFrame(env)
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env

    vec_env = DummyVecEnv([_init])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env

# --- Evaluate a model over N episodes ---
def evaluate_model(model_path, world=1, stage=1, episodes=100, max_steps=10000):
    env = make_eval_env(world, stage, max_steps)
    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

    flag_success = 0
    rewards = []

    for i in trange(episodes, desc=f"Evaluating W{world}-{stage}"):
        obs = env.reset()
        done = False
        total_reward = 0
        

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
            env.render()
        rewards.append(total_reward)
        info_dict = info[0]
        print(f"🎯 Episode {i+1}: Total Reward = {total_reward:.2f} | x_pos = {info_dict.get('x_pos', 0)} | flag = {info_dict.get('flag_get', False)}")

        if info_dict.get('flag_get'):
            flag_success += 1
            time_left = info_dict.get("time", 0)
            print(f"🏁 Mario completed the level with {time_left} seconds left!")

    success_rate = flag_success / episodes * 100
    avg_reward = np.mean(rewards)

    print(f"\n🎮 World {world}-{stage} Evaluation")
    print(f"✅ Flag success rate: {success_rate:.2f}%")
    print(f"🏅 Avg shaped reward: {avg_reward:.2f}")

# --- Run evaluation ---
if __name__ == "__main__":
    # ppo_world1_dynamic_step10000000
    # best_model_world1_dynamic
    model_path = "./logs/general_world1_dynamic/ppo_world1_dynamic_step10000000.zip"  # Change this for different models
    evaluate_model(model_path, world=1, stage=4, episodes=100)