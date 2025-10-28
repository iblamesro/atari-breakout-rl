import gymnasium as gym
import numpy as np
import cv2
import ale_py  # Nécessaire pour enregistrer les envs Atari

# ---------- NoopReset Wrapper ----------
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


# ---------- Frame Preprocessing ----------
def preprocess_frame(frame, out_size=84):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)


# ---------- Frame Stack Wrapper ----------
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
        return self.frames.copy(), float(np.sign(reward)), terminated, truncated, info


# ---------- Environment Factory ----------
def make_env(env_id="ALE/Breakout-v5", seed=0):
    """
    Prépare un environnement Atari standardisé pour l'entraînement et l'évaluation.
    """
    env = gym.make(env_id, frameskip=4, full_action_space=False)
    env = NoopResetEnv(env)
    env = FrameStack(env, k=4)
    env.reset(seed=seed)
    return env
