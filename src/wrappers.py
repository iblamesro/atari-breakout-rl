
import gymnasium as gym
import numpy as np
import cv2

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

def preprocess_frame(frame, out_size=84):
    # frame: (H,W,3) uint8
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = None
        h, w = 84, 84
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(k, h, w), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = preprocess_frame(obs)
        self.frames = np.stack([frame]*self.k, axis=0)
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = preprocess_frame(obs)
        self.frames = np.concatenate([self.frames[1:], frame[None, ...]], axis=0)
        return self.frames.copy(), reward, terminated, truncated, info

class RewardClip(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))

def make_env(env_id="ALE/Breakout-v5", seed=0):
    env = gym.make(env_id, frameskip=4, full_action_space=False)
    env = NoopResetEnv(env)
    env = RewardClip(env)
    env = FrameStack(env, k=4)
    env.reset(seed=seed)
    return env
