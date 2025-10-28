"""
Version améliorée du script d'entraînement avec:
- Meilleurs hyperparamètres par défaut
- Sauvegarde de métadonnées complètes
- Early stopping basé sur performance
- Logging amélioré
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os, csv, argparse, json, time
import numpy as np
import torch
from collections import deque
from tqdm import tqdm
import gymnasium as gym
import ale_py

from src.wrappers import make_env
from src.agent_ddqn import DDQNAgent


def evaluate(env, agent, episodes=5, epsilon=0.05):
    """Évalue l'agent et retourne mean, std."""
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            old_eps = agent.eps_final
            agent.eps_final = epsilon
            a = agent.act(s)
            agent.eps_final = old_eps
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += r
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def save_checkpoint(agent, frame, episode, save_dir, hyperparams, is_best=False):
    """Sauvegarde un checkpoint avec métadonnées complètes."""
    checkpoint = {
        'frame': frame,
        'episode': episode,
        'model_state_dict': agent.online.state_dict(),
        'optimizer_state_dict': agent.optim.state_dict(),
        'scheduler_state_dict': agent.scheduler.state_dict(),
        'hyperparams': hyperparams,
        'architecture': 'DuelingDQN',
        'replay_size': len(agent.memory),
    }
    
    filename = 'best.pt' if is_best else f'ckpt_{frame}.pt'
    path = os.path.join(save_dir, filename)
    torch.save(checkpoint, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total-frames', type=int, default=5_000_000,
                       help='Total frames (5M recommandé pour bonnes performances)')
    parser.add_argument('--log-every', type=int, default=10_000)
    parser.add_argument('--eval-every', type=int, default=100_000)
    parser.add_argument('--save-dir', type=str, default='runs_improved')
    
    # Hyperparamètres OPTIMISÉS pour haute performance
    parser.add_argument('--lr', type=float, default=2.5e-4,
                       help='Learning rate (2.5e-4 = 2.5x plus rapide)')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (64 pour GPU/MPS, 32 pour CPU lent)')
    parser.add_argument('--buffer-size', type=int, default=500_000,
                       help='Replay buffer (500k = plus de diversité)')
    parser.add_argument('--min-replay-size', type=int, default=80_000,
                       help='Min samples avant training (80k pour meilleure init)')
    parser.add_argument('--target-update-tau', type=float, default=0.005,
                       help='Soft update tau (0.005 = 5x plus rapide)')
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-final', type=float, default=0.02,
                       help='Epsilon final (0.02 = plus d\'exploration)')
    parser.add_argument('--epsilon-decay', type=int, default=2_000_000,
                       help='Epsilon decay frames (2M = exploration plus longue)')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                       help='Gradient clipping (10.0 pour stabilité)')
    parser.add_argument('--deep-arch', action='store_true',
                       help='Use deep architecture (4 conv layers)')
    
    args = parser.parse_args()
    
    # Configuration
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🎮 BREAKOUT TRAINING - OPTIMIZED")
    print(f"{'='*60}")
    print(f"📁 Save dir: {args.save_dir}")
    print(f"🎯 Total frames: {args.total_frames:,}")
    print(f"🧠 Architecture: {'Deep (4 conv)' if args.deep_arch else 'Standard (3 conv)'}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"🎓 Learning rate: {args.lr}")
    print(f"🔄 Buffer size: {args.buffer_size:,}")
    print(f"⚡ Target update tau: {args.target_update_tau}")
    print(f"{'='*60}\n")
    
    env = make_env(args.env, seed=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Sauvegarder hyperparams
    hyperparams = vars(args)
    hyperparams['timestamp'] = time.strftime('%Y-%m-%d_%H-%M-%S')
    with open(os.path.join(args.save_dir, 'hyperparams.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    n_actions = env.action_space.n
    agent = DDQNAgent(
        n_actions=n_actions,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        target_update_tau=args.target_update_tau,
        min_replay_size=args.min_replay_size,
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_frames=args.epsilon_decay,
        grad_clip=args.grad_clip,
        deep_arch=args.deep_arch
    )
    
    s, _ = env.reset()
    episode_return, ep = 0.0, 0
    recent = deque(maxlen=100)
    best_eval_score = -float('inf')
    
    csv_path = os.path.join(args.save_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['frame','episode','ep_return','mean100','epsilon','loss','eval_mean','eval_std','lr'])
    
    print(f"🚀 Démarrage de l'entraînement - {args.total_frames:,} frames")
    print(f"📊 Hyperparamètres sauvegardés dans {args.save_dir}/hyperparams.json\n")
    
    start_time = time.time()
    pbar = tqdm(total=args.total_frames, desc="Training", unit="frames")
    
    for frame in range(1, args.total_frames + 1):
        agent.frame_idx = frame
        a = agent.act(s)
        ns, r, terminated, truncated, _ = env.step(a)
        d = terminated or truncated
        agent.store(s, a, r, ns, float(d))
        loss = agent.train_step()
        s = ns
        episode_return += r
        
        if d:
            ep += 1
            recent.append(episode_return)
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([frame, ep, episode_return, np.mean(recent) if recent else 0.0, 
                                       agent.epsilon(), loss or '', '', '', agent.optim.param_groups[0]['lr']])
            
            pbar.set_postfix({
                'ep': ep,
                'ret': f'{episode_return:.1f}',
                'mean100': f'{np.mean(recent):.1f}' if recent else '0.0',
                'ε': f'{agent.epsilon():.3f}'
            })
            
            s, _ = env.reset()
            episode_return = 0.0
        
        # Évaluation périodique
        if frame % args.eval_every == 0:
            mean_ret, std_ret = evaluate(env, agent, episodes=10)
            
            # Mettre à jour le learning rate scheduler
            agent.update_scheduler(mean_ret)
            
            # Sauvegarde du meilleur modèle
            is_best = mean_ret > best_eval_score
            if is_best:
                best_eval_score = mean_ret
                save_checkpoint(agent, frame, ep, args.save_dir, hyperparams, is_best=True)
                print(f"\n🌟 Nouveau meilleur score: {mean_ret:.1f}±{std_ret:.1f} (frame {frame:,})")
            
            # Sauvegarde checkpoint régulier
            save_checkpoint(agent, frame, ep, args.save_dir, hyperparams, is_best=False)
            
            elapsed = time.time() - start_time
            fps = frame / elapsed
            eta = (args.total_frames - frame) / fps / 3600
            
            print(f"\n[Eval] Frame {frame:,}: Ret {mean_ret:.1f}±{std_ret:.1f} | "
                  f"ε {agent.epsilon():.3f} | Buffer {len(agent.memory):,} | "
                  f"LR {agent.optim.param_groups[0]['lr']:.2e}")
            print(f"⏱️  FPS: {fps:.1f} | ETA: {eta:.1f}h")
            
            # Mettre à jour le CSV avec les résultats d'évaluation
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([frame, ep, '', '', agent.epsilon(), '', 
                                       mean_ret, std_ret, agent.optim.param_groups[0]['lr']])
        
        pbar.update(1)
    
    pbar.close()
    
    # Sauvegarde finale
    save_checkpoint(agent, args.total_frames, ep, args.save_dir, hyperparams, is_best=False)
    torch.save(agent.online.state_dict(), os.path.join(args.save_dir, 'final_state_dict.pt'))
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Entraînement terminé!")
    print(f"⏱️  Temps total: {total_time/3600:.2f}h")
    print(f"📊 Meilleur score: {best_eval_score:.1f}")
    print(f"📁 Sauvegardé dans: {args.save_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
