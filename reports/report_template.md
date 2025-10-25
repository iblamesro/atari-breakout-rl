
# Atari Breakout with Double DQN and Prioritized Replay

## 1. Introduction & Problem Overview
Briefly motivate RL on Atari, define objective and evaluation.

## 2. Environment & Reward
- Observation pipeline: grayscale, 84x84, frame stack (k=4)
- Action space
- Reward clipping and termination

## 3. Method
- DQN recap
- Double DQN (selection vs evaluation) with fixed targets
- Prioritized Experience Replay (alpha, beta anneal)

## 4. Training Setup
- Network architecture
- Hyperparameters (lr, gamma, batch, replay size, target update, epsilon schedule, clip)
- Implementation details (warmup, update frequency, seeds)
- Hardware

## 5. Experiments & Results
- Setups: DQN, DDQN, DDQN+PER (2 seeds)
- Curves: reward (raw & moving avg), loss, epsilon
- Table: final score, area under learning curve, wall-clock
- Analysis: stability, sample efficiency, what ablations show

## 6. Conclusion
Key findings, limitations, and future work (dueling heads, n-step, NoisyNets, sticky actions).

## 7. Team Roles
- Member A — research & reading
- Member B — implementation & infra
- Member C — experiments & analysis
