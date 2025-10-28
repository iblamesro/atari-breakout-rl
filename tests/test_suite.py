"""
Tests unitaires pour le projet Atari Breakout DDQN.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import torch
import gymnasium as gym
import ale_py  # Nécessaire pour enregistrer les envs Atari

from src.replay_per import PrioritizedReplayBuffer
from src.agent_ddqn import DDQNAgent, DuelingDQN
from src.wrappers import make_env


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Tests pour PrioritizedReplayBuffer."""
    
    def test_initialization(self):
        """Test que le buffer s'initialise correctement."""
        buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.capacity, 1000)
        self.assertEqual(buffer.alpha, 0.6)
    
    def test_push_and_len(self):
        """Test que push ajoute des éléments et len fonctionne."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        action = 1
        reward = 1.0
        next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        self.assertEqual(len(buffer), 1)
        
        # Ajouter plus d'éléments
        for _ in range(50):
            buffer.push(state, action, reward, next_state, done)
        self.assertEqual(len(buffer), 51)
    
    def test_sample(self):
        """Test que sample retourne le bon format."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        
        # Remplir le buffer
        for i in range(50):
            state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
            action = i % 4
            reward = float(i % 2)
            next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
            done = float(i % 10 == 0)
            buffer.push(state, action, reward, next_state, done)
        
        # Sample un batch
        states, actions, rewards, next_states, dones, indices, weights = buffer.sample(32)
        
        self.assertEqual(states.shape, (32, 4, 84, 84))
        self.assertEqual(actions.shape, (32,))
        self.assertEqual(rewards.shape, (32,))
        self.assertEqual(next_states.shape, (32, 4, 84, 84))
        self.assertEqual(dones.shape, (32,))
        self.assertEqual(len(indices), 32)
        self.assertEqual(len(weights), 32)
    
    def test_update_priorities(self):
        """Test que les priorités se mettent à jour."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        
        for i in range(20):
            state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
            buffer.push(state, 0, 0.0, state, 0.0)
        
        # Sample et update
        _, _, _, _, _, indices, _ = buffer.sample(10)
        td_errors = np.random.randn(10)
        
        buffer.update_priorities(indices, td_errors)
        
        # Vérifier que les priorités ont changé
        self.assertTrue(np.any(buffer.priorities[:20] != buffer.priorities[0]))
    
    def test_overflow(self):
        """Test que le buffer gère le dépassement de capacité."""
        buffer = PrioritizedReplayBuffer(capacity=10)
        
        for i in range(20):
            state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
            buffer.push(state, 0, 0.0, state, 0.0)
        
        # Le buffer ne devrait pas dépasser sa capacité
        self.assertEqual(len(buffer), 10)


class TestDuelingDQN(unittest.TestCase):
    """Tests pour le réseau DuelingDQN."""
    
    def test_initialization(self):
        """Test que le réseau s'initialise correctement."""
        model = DuelingDQN(in_channels=4, n_actions=4)
        self.assertIsNotNone(model)
    
    def test_forward_pass(self):
        """Test qu'un forward pass fonctionne."""
        model = DuelingDQN(in_channels=4, n_actions=4)
        model.eval()
        
        # Input batch de 2 images
        x = torch.randn(2, 4, 84, 84)
        
        with torch.no_grad():
            output = model(x)
        
        # Vérifier la forme de sortie
        self.assertEqual(output.shape, (2, 4))
    
    def test_output_range(self):
        """Test que les Q-values sont dans une plage raisonnable."""
        model = DuelingDQN(in_channels=4, n_actions=4)
        model.eval()
        
        x = torch.randn(1, 4, 84, 84)
        
        with torch.no_grad():
            output = model(x)
        
        # Les Q-values devraient être raisonnables (pas NaN ou Inf)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestDDQNAgent(unittest.TestCase):
    """Tests pour l'agent DDQN."""
    
    def test_initialization(self):
        """Test que l'agent s'initialise correctement."""
        agent = DDQNAgent(n_actions=4)
        self.assertEqual(agent.n_actions, 4)
        self.assertIsNotNone(agent.online)
        self.assertIsNotNone(agent.target)
        self.assertIsNotNone(agent.optim)
    
    def test_act(self):
        """Test que act retourne une action valide."""
        agent = DDQNAgent(n_actions=4)
        state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        
        action = agent.act(state)
        
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 4)
    
    def test_store(self):
        """Test que store ajoute à la mémoire."""
        agent = DDQNAgent(n_actions=4)
        
        state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        action = 1
        reward = 1.0
        next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        done = False
        
        initial_size = len(agent.memory)
        agent.store(state, action, reward, next_state, done)
        
        self.assertEqual(len(agent.memory), initial_size + 1)
    
    def test_epsilon_decay(self):
        """Test que epsilon décroit correctement."""
        agent = DDQNAgent(n_actions=4, epsilon_start=1.0, epsilon_final=0.1, epsilon_decay_frames=100)
        
        eps_start = agent.epsilon()
        self.assertAlmostEqual(eps_start, 1.0, places=1)
        
        # Simuler 50 frames
        agent.frame_idx = 50
        eps_mid = agent.epsilon()
        self.assertLess(eps_mid, eps_start)
        
        # Simuler 100 frames
        agent.frame_idx = 100
        eps_end = agent.epsilon()
        self.assertLess(eps_end, eps_mid)


class TestEnvironment(unittest.TestCase):
    """Tests pour l'environnement wrappé."""
    
    def test_make_env(self):
        """Test que make_env crée un environnement valide."""
        env = make_env("ALE/Breakout-v5", seed=0)
        self.assertIsNotNone(env)
        
        # Test reset
        state, info = env.reset()
        self.assertEqual(state.shape, (4, 84, 84))
        self.assertEqual(state.dtype, np.uint8)
        
        # Test step
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        self.assertEqual(next_state.shape, (4, 84, 84))
        self.assertIsInstance(reward, float)
        
        env.close()


class TestCheckpointCompatibility(unittest.TestCase):
    """Tests de compatibilité des checkpoints."""
    
    def test_load_converted_checkpoint(self):
        """Test qu'on peut charger un checkpoint converti."""
        if not os.path.exists('runs/final_dueling.pt'):
            self.skipTest("Checkpoint converti non disponible")
        
        agent = DDQNAgent(n_actions=4)
        state_dict = torch.load('runs/final_dueling.pt', map_location='cpu')
        
        # Devrait charger sans erreur en mode strict
        agent.online.load_state_dict(state_dict, strict=True)
        
        # Test forward pass
        x = torch.randn(1, 4, 84, 84).to(agent.device)
        with torch.no_grad():
            output = agent.online(x)
        
        self.assertEqual(output.shape, (1, 4))


def run_tests():
    """Lance tous les tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
