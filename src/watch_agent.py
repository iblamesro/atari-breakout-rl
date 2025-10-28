import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # ✅ Doit être tout en haut

import cv2, time, torch
import gymnasium as gym
import ale_py

from src.agent_ddqn import DDQNAgent
from src.wrappers import make_env


def watch_agent(model_path="runs/ckpt_500000.pt", env_id="ALE/Breakout-v5", seed=1, episodes=3, render_fps=30, epsilon=0.01):
    """
    Affiche un agent DDQN entraîné jouant à Atari Breakout.
    Appuyer sur 'q' pour quitter.
    
    Args:
        model_path: Chemin vers le checkpoint
        env_id: ID de l'environnement
        seed: Random seed
        episodes: Nombre d'épisodes à jouer
        render_fps: FPS d'affichage (30 = fluide, 60 = rapide)
        epsilon: Taux d'exploration (0.01 = quasi déterministe)
    """
    print(f"\n{'='*60}")
    print(f"🎮 BREAKOUT AGENT DEMO")
    print(f"{'='*60}")
    print(f"📁 Model: {model_path}")
    print(f"🎬 Episodes: {episodes}")
    print(f"🎞️  FPS: {render_fps}")
    print(f"{'='*60}\n")

    # Chargement de l'environnement avec rendu RGB
    env = gym.make(env_id, render_mode="rgb_array", frameskip=4, full_action_space=False)
    env.reset(seed=seed)

    # Chargement de l'agent
    agent = DDQNAgent(env.action_space.n)
    
    # Chargement robuste du checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Si c'est un checkpoint complet
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'eval_mean' in checkpoint and checkpoint['eval_mean'] is not None:
            print(f"📊 Model performance: {checkpoint['eval_mean']:.1f}±{checkpoint.get('eval_std', 0):.1f}")
        if 'frame' in checkpoint:
            print(f"🎯 Trained for {checkpoint['frame']:,} frames")
    else:
        state_dict = checkpoint
    
    try:
        agent.online.load_state_dict(state_dict)
        print(f"✅ Loaded checkpoint successfully\n")
    except Exception as e:
        print(f"⚠️  Warning: Loading with strict=False due to: {e}")
        agent.online.load_state_dict(state_dict, strict=False)
    
    agent.online.eval()
    agent.eps_final = epsilon  # Epsilon très bas pour évaluation

    # Contrôle du rythme de rendu
    frame_delay = 1.0 / render_fps
    
    returns = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        frame_count = 0
        start_time = time.time()

        print(f"▶️  Episode {ep + 1}/{episodes}")

        while not done:
            # Action du modèle
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            frame_count += 1

            # Rendu du jeu
            frame = env.render()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Score et infos
                cv2.putText(frame, f"Score: {int(total_reward)}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Episode {ep+1}/{episodes}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.imshow("Breakout DDQN 🧠", frame)

            # Maintenir FPS constant
            elapsed = time.time() - start_time
            expected = frame_count * frame_delay
            if elapsed < expected:
                time.sleep(expected - elapsed)

            # Quitter avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True
                break

            state = next_state

        returns.append(total_reward)
        print(f"✅ Episode {ep+1} terminé — Score : {total_reward:.1f}")
        time.sleep(0.5)

    env.close()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print(f"� RESULTS")
    print(f"{'='*60}")
    print(f"Mean score: {sum(returns)/len(returns):.1f}")
    print(f"Best score: {max(returns):.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch a trained Breakout DDQN agent")
    parser.add_argument('--model-path', type=str, default='runs/ckpt_500000.pt', 
                       help='Path to model checkpoint')
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5', 
                       help='Gym environment id')
    parser.add_argument('--episodes', type=int, default=3, 
                       help='Number of episodes to play')
    parser.add_argument('--seed', type=int, default=1, 
                       help='Random seed')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Render FPS (30=smooth, 60=fast)')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='Exploration rate (0.01 = almost deterministic)')
    args = parser.parse_args()

    watch_agent(model_path=args.model_path, env_id=args.env, seed=args.seed, 
               episodes=args.episodes, render_fps=args.fps, epsilon=args.epsilon)
