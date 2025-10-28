import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .replay_per import PrioritizedReplayBuffer


# ======== DUELING DQN NETWORK (OPTIMISÉ) ========
class DuelingDQN(nn.Module):
    """
    Architecture Dueling DQN améliorée avec:
    - 4 couches convolutionnelles (au lieu de 3) pour plus de capacité
    - Batch normalization pour stabilité
    - Plus de neurones dans les couches FC (512)
    """
    def __init__(self, in_channels=4, n_actions=4, deep_arch=True):
        super().__init__()
        
        if deep_arch:
            # Architecture PROFONDE (recommandée pour haute performance)
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
            self.bn4 = nn.BatchNorm2d(128)
            conv_out_size = 5 * 5 * 128  # 84x84 -> 20x20 -> 9x9 -> 7x7 -> 5x5
            self.deep = True
        else:
            # Architecture STANDARD (compatible avec anciens modèles)
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            conv_out_size = 7 * 7 * 64
            self.deep = False

        # Streams Value et Advantage avec plus de neurones
        self.fc_val = nn.Linear(conv_out_size, 512)
        self.fc_adv = nn.Linear(conv_out_size, 512)
        self.val = nn.Linear(512, 1)
        self.adv = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x / 255.0
        
        if self.deep:
            # Forward avec batch norm
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
        else:
            # Forward standard
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)

        val = F.relu(self.fc_val(x))
        adv = F.relu(self.fc_adv(x))
        val = self.val(val)
        adv = self.adv(adv)

        # Dueling: Q(s,a) = V(s) + (A(s,a) - mean(A))
        return val + adv - adv.mean(1, keepdim=True)


# ======== DOUBLE DUELING DQN AGENT (OPTIMISÉ) ========
class DDQNAgent:
    """
    Agent Double DQN avec Dueling architecture et améliorations:
    - Learning rate adaptatif (ReduceLROnPlateau)
    - Gradient clipping pour stabilité
    - Hyperparamètres optimisés par défaut
    - Support architecture profonde optionnelle
    """
    def __init__(self, n_actions, device=None,
                 lr=2.5e-4, gamma=0.99, batch_size=64,
                 buffer_capacity=500000, target_update_tau=0.005,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=2000000,
                 min_replay_size=80000, epsilon_start=1.0, epsilon_final=0.02, 
                 epsilon_decay_frames=2_000_000, grad_clip=10.0, deep_arch=False):

        # ======== 🔥 AUTOMATIC DEVICE SELECTION (MPS / CUDA / CPU) ========
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("⚡ Using Apple Metal GPU (MPS)!")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("⚡ Using NVIDIA GPU (CUDA)!")
            else:
                self.device = torch.device("cpu")
                print("💻 Using CPU only.")
        else:
            self.device = device

        # ======== INITIALIZATION ========
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_tau = target_update_tau
        self.min_replay_size = min_replay_size
        self.grad_clip = grad_clip

        # Réseaux principal & cible
        self.online = DuelingDQN(in_channels=4, n_actions=n_actions, deep_arch=deep_arch).to(self.device)
        self.target = DuelingDQN(in_channels=4, n_actions=n_actions, deep_arch=deep_arch).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        
        if deep_arch:
            print("🏗️  Using DEEP architecture (4 conv layers)")
        else:
            print("🏗️  Using standard architecture (3 conv layers)")

        # Optimiseur avec meilleur learning rate initial
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr, eps=1e-5, amsgrad=True)
        
        # Scheduler adaptatif (réduit LR quand performance stagne)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )

        # Mémoire avec Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(
            buffer_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames
        )

        # Fonction de perte Huber
        self.loss_fn = nn.HuberLoss(reduction='none')

        # Epsilon pour exploration (valeurs optimisées)
        self.eps_start = epsilon_start
        self.eps_final = epsilon_final
        self.eps_decay = epsilon_decay_frames
        self.frame_idx = 0
        self.steps = 0


    # ======== EPSILON DECAY ========
    def epsilon(self):
        return self.eps_final + (self.eps_start - self.eps_final) * np.exp(-1. * self.frame_idx / (self.eps_decay / 2))


    # ======== ACTION SELECTION ========
    @torch.no_grad()
    def act(self, state):
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.n_actions)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device).float()
        q = self.online(state_t)
        return int(q.argmax(dim=1).item())


    # ======== MEMORY STORAGE ========
    def store(self, s, a, r, ns, d):
        self.memory.push(s, a, r, ns, d)


    # ======== SOFT UPDATE (Target Network) ========
    def soft_update(self):
        tau = self.target_update_tau
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


    # ======== TRAINING STEP (OPTIMISÉ) ========
    def train_step(self):
        if len(self.memory) < max(self.min_replay_size, self.batch_size):
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Q-values actuelles
        q_values = self.online(states_t)
        q_selected = q_values.gather(1, actions_t.view(-1, 1)).squeeze(1)

        # Q-values cibles (Double DQN)
        with torch.no_grad():
            next_q_online = self.online(next_states_t)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states_t)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        # TD error et perte pondérée
        td_error = target_q - q_selected
        loss_el = self.loss_fn(q_selected, target_q)
        loss = (loss_el * weights_t).mean()

        # Rétropropagation avec GRADIENT CLIPPING
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=self.grad_clip)
        self.optim.step()

        # Mise à jour des priorités PER
        self.memory.update_priorities(indices, td_error.detach().cpu().numpy())

        # Mise à jour douce du réseau cible
        self.soft_update()
        self.steps += 1

        return float(loss.item())
    
    
    # ======== UPDATE LEARNING RATE SCHEDULER ========
    def update_scheduler(self, eval_score):
        """Met à jour le learning rate basé sur le score d'évaluation."""
        self.scheduler.step(eval_score)
