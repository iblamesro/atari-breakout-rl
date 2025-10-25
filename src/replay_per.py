
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=1000000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.frame = 1

    def __len__(self):
        return len(self.buffer)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Buffer is empty")
        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (np.array(states, dtype=np.uint8),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.uint8),
                np.array(dones, dtype=np.float32),
                indices,
                weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + 1e-6
