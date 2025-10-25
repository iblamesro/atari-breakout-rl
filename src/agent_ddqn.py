
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .replay_per import PrioritizedReplayBuffer

class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DDQNAgent:
    def __init__(self, n_actions, device=None,
                 lr=1e-4, gamma=0.99, batch_size=32,
                 buffer_capacity=200000, target_update_freq=10000,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 min_replay_size=50000, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay_frames=1_000_000):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size

        self.online = DQN(in_channels=4, n_actions=n_actions).to(self.device)
        self.target = DQN(in_channels=4, n_actions=n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha,
                                              beta_start=per_beta_start, beta_frames=per_beta_frames)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        # epsilon linear schedule
        self.eps_start = epsilon_start
        self.eps_final = epsilon_final
        self.eps_decay = epsilon_decay_frames
        self.frame_idx = 0

        self.steps = 0

    def epsilon(self):
        frac = min(1.0, self.frame_idx / self.eps_decay)
        return self.eps_start + frac * (self.eps_final - self.eps_start)

    @torch.no_grad()
    def act(self, state):
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.n_actions)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.online(state_t)
        return int(q.argmax(dim=1).item())

    def store(self, s, a, r, ns, d):
        self.memory.push(s, a, r, ns, d)

    def train_step(self):
        if len(self.memory) < max(self.min_replay_size, self.batch_size):
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        # current Q
        q_values = self.online(states_t)
        q_selected = q_values.gather(1, actions_t.view(-1,1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online(next_states_t)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states_t)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        td_error = target_q - q_selected
        loss_el = self.loss_fn(q_selected, target_q)
        loss = (loss_el * weights_t).mean()

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optim.step()

        # update PER priorities
        self.memory.update_priorities(indices, td_error.detach().cpu().numpy())

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())
