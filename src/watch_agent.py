import os
import sys
import time
import cv2
import torch
import numpy as np
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

# Pour s'assurer que Python trouve les modules dans src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent_ddqn import DDQNAgent
from src.wrappers import make_env


def watch_agent(model_path="runs/final.pt", env_id="ALE/Breakout-v5", seed=1, episodes=3):
    """
    Watch a trained agent play Atari Breakout.
    """
    print("🎮 Watching trained agent on:", env_id)

    # Environnement avec rendu vidéo
    env = gym.make(env_id, render_mode="rgb_array", frameskip=4, full_action_space=False)
    env.reset(seed=seed)

    # Chargement de l’agent
    agent = DDQNAgent(env.action_space.n)
    agent.online.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.online.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frame_count = 0

        print(f"\n▶️ Episode {ep + 1}/{episodes}")
        while not done:
            # Action de l’agent (pas d’exploration)
            action = agent.act(state)

            # Interaction avec l’environnement
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            frame_count += 1

            # Rendu OpenCV
            frame = env.render()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame, f"Score: {int(total_reward)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Breakout Agent 🎮", frame)
            
            # Fermer avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True
                break

            state = next_state

        print(f"✅ Episode {ep + 1} finished — Total Reward: {total_reward:.1f} | Frames: {frame_count}")
        time.sleep(1)

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    watch_agent()
