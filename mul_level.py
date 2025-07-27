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
def evaluate_model(model_path, world, stage, max_steps=10000, max_tries=50):
    print(f"\n🕹️ Evaluating: World {world}-{stage}")
    env = make_eval_env(world, stage, max_steps)
    model = PPO.load(model_path, env=env, device="cuda" if torch.cuda.is_available() else "cpu")

    for i in range(1, max_tries + 1):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
            env.render()

        info_dict = info[0]
        print(f"🎯 Episode {i}: Total Reward = {total_reward:.2f} | x_pos = {info_dict.get('x_pos', 0)} | flag = {info_dict.get('flag_get', False)}")

        if info_dict.get('flag_get'):
            time_left = info_dict.get("time", 0)
            print(f"🏁 Mario completed the level in episode {i} with {time_left} seconds left!")
            return True  # Success

    print(f"\n❌ Agent failed to complete World {world}-{stage} in {max_tries} tries.")
    return False  # Failure

# --- Evaluate W1-1 to W1-4 in sequence ---
if __name__ == "__main__":
    world = 1
    for stage in range(1, 5):
        model_path = f"./logs/W{world}-{stage}.zip"
        success = evaluate_model(model_path, world, stage)

        if not success:
            print(f"\n⛔ Stopping evaluation. Agent failed at World {world}-{stage}.")
            break
        else:
            print(f"\n✅ Agent completed World {world}-{stage}. Moving to next stage...")
