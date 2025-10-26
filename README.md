# Atari Breakout - Double DQN Agent

## 1. Project Overview

This project implements a **Double Deep Q-Network (DDQN)** agent capable of learning to play the Atari **Breakout** game using **Reinforcement Learning**.
The objective is to train an autonomous agent to maximize its game score by interacting with the environment through trial and error.

The project was developed as part of the **Reinforcement Learning course** at the Albert School (MSc Data for Finance).
It follows the Gymnasium (OpenAI) Atari API and uses **PyTorch** for model training.

---

## 2. Environment Description

* **Environment:** `ALE/Breakout-v5` (Atari Learning Environment)
* **Observation space:** 4 stacked grayscale frames (84×84 pixels each)
* **Action space:** 4 discrete actions (move left, move right, fire, no-op)
* **Reward:** +1 for breaking a brick, 0 otherwise
* **Episode termination:** when the ball is lost or the game ends

---

## 3. Algorithm Description

The implemented algorithm is a **Double Deep Q-Network (DDQN)** combined with a **Prioritized Experience Replay (PER)** buffer to improve sample efficiency.

### Key Components

* **Online network:** Learns to approximate the Q-values (state–action value function)
* **Target network:** Provides stable Q-value targets (updated every few steps)
* **Prioritized Replay Buffer:** Samples important experiences more often, based on TD error magnitude
* **Epsilon-greedy policy:** Balances exploration and exploitation during training

### Network Architecture

1. **Convolutional layers:** Extract visual features from game frames
2. **Fully connected layers:** Compute action-value (Q) estimates
3. **Activation function:** ReLU
4. **Optimizer:** Adam

---

## 4. Training Setup

### Command

```bash
python -m src.train_breakout_ddqn_per --env ALE/Breakout-v5 --total-frames 500000 --seed 1
```

### Key Hyperparameters

| Parameter             | Value               |
| --------------------- | ------------------- |
| Learning rate         | 1e-4                |
| Discount factor (γ)   | 0.99                |
| Replay buffer size    | 200,000             |
| Batch size            | 32                  |
| Epsilon start         | 1.0                 |
| Epsilon final         | 0.05                |
| Epsilon decay         | 1,000,000 frames    |
| Target update         | Every 10,000 frames |
| Total training frames | 500,000             |

During training, performance metrics such as average reward, epsilon value, and loss are logged and saved in `runs/training_log.csv`.

---

## 5. Evaluation and Results

### Evaluate the trained agent

```bash
python -m src.watch_agent
```

The trained model achieves consistent performance, typically scoring between **8 and 12 points per episode** after 500,000 training frames.
The agent successfully learns basic ball–paddle coordination and demonstrates stable gameplay behavior.

Training curves (reward vs frames and loss vs frames) can be generated using:

```bash
python plot_training.py
```

Resulting figures are stored or displayed for analysis.

---

## 6. File Structure

```
atari-breakout-rl/
│
├── src/
│   ├── agent_ddqn.py          # Double DQN model and training logic
│   ├── replay_per.py          # Prioritized Experience Replay implementation
│   ├── train_breakout_ddqn_per.py  # Training script
│   ├── wrappers.py            # Environment preprocessing (frames, normalization, etc.)
│   └── watch_agent.py         # Script to visualize trained agent
│
├── runs/                      # Logs and model checkpoints
│   ├── training_log.csv
│   └── final.pt
│
├── plot_training.py           # Generates training performance plots
├── requirements.txt           # Required dependencies
└── README.md                  # Project documentation
```

---

## 7. Dependencies

### Installation

```bash
pip install -r requirements.txt
```

### Main Libraries

* Python 3.10+
* PyTorch
* Gymnasium
* ALE-py
* NumPy
* OpenCV
* TQDM
* Matplotlib
* Pandas

---

## 8. Authors and Roles

| Member                  | Role                                                                 |
| ----------------------- | -------------------------------------------------------------------- |
| **Sara Ben Abdelkader** | Implementation of DDQN algorithm, training management, visualization |
| **Saber Dhib**          | Hyperparameter tuning, results analysis, and reporting               |


