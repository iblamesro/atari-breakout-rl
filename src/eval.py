"""
Script d'évaluation headless pour mesurer les performances de l'agent sans GUI.
Génère des statistiques détaillées et peut sauvegarder les résultats en CSV.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import csv
from datetime import datetime
from tqdm import tqdm

from src.wrappers import make_env
from src.agent_ddqn import DDQNAgent


def evaluate_agent(agent, env, n_episodes=30, epsilon=0.01, verbose=True):
    """
    Évalue l'agent sur n_episodes et retourne des statistiques détaillées.
    
    Args:
        agent: Agent DDQN entraîné
        env: Environnement Gymnasium
        n_episodes: Nombre d'épisodes d'évaluation
        epsilon: Epsilon pour l'évaluation (0.01 = quasi-greedy)
        verbose: Afficher les résultats intermédiaires
    
    Returns:
        dict: Statistiques (mean, std, min, max, median, scores)
    """
    scores = []
    episode_lengths = []
    
    # Sauvegarder l'epsilon original
    original_eps = agent.eps_final
    agent.eps_final = epsilon
    
    iterator = tqdm(range(n_episodes), desc="Évaluation") if verbose else range(n_episodes)
    
    for ep in iterator:
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            state = next_state
        
        scores.append(total_reward)
        episode_lengths.append(steps)
        
        if verbose:
            iterator.set_postfix({
                'score': f'{total_reward:.1f}',
                'mean': f'{np.mean(scores):.2f}',
                'steps': steps
            })
    
    # Restaurer epsilon
    agent.eps_final = original_eps
    
    # Calculer statistiques
    stats = {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'median': float(np.median(scores)),
        'q25': float(np.percentile(scores, 25)),
        'q75': float(np.percentile(scores, 75)),
        'mean_length': float(np.mean(episode_lengths)),
        'scores': scores,
        'lengths': episode_lengths,
    }
    
    return stats


def print_stats(stats, model_path):
    """Affiche les statistiques de manière lisible."""
    print("\n" + "="*60)
    print(f"📊 RÉSULTATS D'ÉVALUATION")
    print(f"Modèle: {model_path}")
    print("="*60)
    print(f"Score moyen:      {stats['mean']:>8.2f} ± {stats['std']:.2f}")
    print(f"Score médian:     {stats['median']:>8.2f}")
    print(f"Score min/max:    {stats['min']:>8.1f} / {stats['max']:.1f}")
    print(f"Quartiles (25/75):{stats['q25']:>8.2f} / {stats['q75']:.2f}")
    print(f"Longueur moyenne: {stats['mean_length']:>8.1f} steps")
    print("="*60)


def save_results(stats, model_path, output_csv):
    """Sauvegarde les résultats dans un fichier CSV."""
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'model', 'mean_score', 'std_score', 'median_score', 
                        'min_score', 'max_score', 'q25', 'q75', 'mean_length', 'n_episodes'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            model_path,
            stats['mean'],
            stats['std'],
            stats['median'],
            stats['min'],
            stats['max'],
            stats['q25'],
            stats['q75'],
            stats['mean_length'],
            len(stats['scores'])
        ])
        
        # Sauvegarder les scores individuels
        writer.writerow([])
        writer.writerow(['episode', 'score', 'length'])
        for i, (score, length) in enumerate(zip(stats['scores'], stats['lengths'])):
            writer.writerow([i+1, score, length])
    
    print(f"\n💾 Résultats sauvegardés: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Évaluer un agent DDQN Breakout")
    parser.add_argument('--model-path', type=str, required=True, help='Chemin du checkpoint')
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5', help='Environnement')
    parser.add_argument('--n-episodes', type=int, default=30, help='Nombre d\'épisodes')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Epsilon pour évaluation')
    parser.add_argument('--seed', type=int, default=42, help='Seed aléatoire')
    parser.add_argument('--output', type=str, default=None, help='Fichier CSV de sortie')
    parser.add_argument('--quiet', action='store_true', help='Mode silencieux')
    args = parser.parse_args()
    
    # Configuration
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"🎮 Initialisation de l'environnement: {args.env}")
    env = make_env(args.env, seed=args.seed)
    
    print(f"🧠 Chargement du modèle: {args.model_path}")
    agent = DDQNAgent(n_actions=env.action_space.n)
    
    # Chargement robuste
    state_dict = torch.load(args.model_path, map_location='cpu')
    try:
        agent.online.load_state_dict(state_dict, strict=True)
        print("✓ Chargement strict réussi")
    except Exception as e:
        print(f"⚠️  Chargement strict échoué, tentative avec strict=False")
        agent.online.load_state_dict(state_dict, strict=False)
    
    agent.online.eval()
    
    # Évaluation
    print(f"\n🔬 Évaluation sur {args.n_episodes} épisodes (ε={args.epsilon})...")
    stats = evaluate_agent(agent, env, n_episodes=args.n_episodes, 
                          epsilon=args.epsilon, verbose=not args.quiet)
    
    # Affichage et sauvegarde
    print_stats(stats, args.model_path)
    
    if args.output:
        save_results(stats, args.model_path, args.output)
    
    env.close()
    return 0


if __name__ == '__main__':
    exit(main())
