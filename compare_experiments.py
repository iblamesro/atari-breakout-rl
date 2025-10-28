"""
Script pour comparer les résultats de plusieurs expériences
Usage: python compare_experiments.py --dir experiments
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_experiment_results(exp_dir):
    """Charge les résultats d'une expérience."""
    try:
        log_path = os.path.join(exp_dir, 'training_log.csv')
        df = pd.read_csv(log_path)
        
        # Info du run
        info_path = os.path.join(exp_dir, 'info.txt')
        info = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.strip().split(':', 1)
                        info[key.strip()] = val.strip()
        
        # Statistiques
        episodes = df[df['ep_return'].notna()]
        if len(episodes) > 0:
            final_mean = episodes['mean100'].iloc[-1] if 'mean100' in episodes.columns else 0
            max_score = episodes['ep_return'].max()
            
            # Eval scores
            eval_df = df[df['eval_mean'].notna()]
            best_eval = eval_df['eval_mean'].max() if not eval_df.empty else 0
        else:
            final_mean = 0
            max_score = 0
            best_eval = 0
        
        return {
            'name': os.path.basename(exp_dir),
            'df': df,
            'info': info,
            'final_mean': final_mean,
            'max_score': max_score,
            'best_eval': best_eval,
            'total_episodes': len(episodes)
        }
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement de {exp_dir}: {e}")
        return None


def compare_experiments(experiments_dir, output_dir='reports/experiments'):
    """Compare toutes les expériences."""
    
    experiments_dir = Path(experiments_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print(f"🔬 COMPARAISON DES EXPÉRIENCES")
    print(f"{'='*60}\n")
    
    # Trouver tous les dossiers d'expériences
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    
    if not exp_dirs:
        print(f"❌ Aucune expérience trouvée dans {experiments_dir}")
        return
    
    print(f"📁 Trouvé {len(exp_dirs)} expériences:\n")
    
    # Charger toutes les expériences
    experiments = []
    for exp_dir in sorted(exp_dirs):
        result = load_experiment_results(exp_dir)
        if result:
            experiments.append(result)
            print(f"  ✅ {result['name']}")
    
    if not experiments:
        print("\n❌ Aucune expérience valide chargée")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 RÉSULTATS COMPARATIFS")
    print(f"{'='*60}\n")
    
    # Tableau comparatif
    print(f"{'Expérience':<30} {'Mean100':<12} {'Max Score':<12} {'Best Eval':<12} {'Episodes':<10}")
    print("-" * 80)
    
    for exp in experiments:
        print(f"{exp['name']:<30} {exp['final_mean']:<12.2f} {exp['max_score']:<12.1f} "
              f"{exp['best_eval']:<12.2f} {exp['total_episodes']:<10}")
    
    print("")
    
    # Trouver la meilleure expérience
    best_exp = max(experiments, key=lambda x: x['final_mean'])
    print(f"🏆 MEILLEURE EXPÉRIENCE: {best_exp['name']}")
    print(f"   Score final (mean100): {best_exp['final_mean']:.2f}")
    print(f"   Max score: {best_exp['max_score']:.1f}")
    print(f"   Best eval: {best_exp['best_eval']:.2f}")
    
    # Graphiques de comparaison
    print(f"\n🎨 Génération des graphiques...")
    
    # 1. Courbes d'apprentissage
    plt.figure(figsize=(14, 8))
    
    for exp in experiments:
        df = exp['df']
        episodes_df = df[df['ep_return'].notna()]
        if not episodes_df.empty and 'mean100' in episodes_df.columns:
            plt.plot(episodes_df['frame'], episodes_df['mean100'], 
                    label=exp['name'], linewidth=2, alpha=0.8)
    
    plt.xlabel('Training Frames', fontsize=12)
    plt.ylabel('Mean 100 Episodes Score', fontsize=12)
    plt.title('Comparaison des Courbes d\'Apprentissage', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {output_dir}/comparison_learning_curves.png")
    
    # 2. Bar chart des performances finales
    plt.figure(figsize=(12, 6))
    
    names = [exp['name'] for exp in experiments]
    means = [exp['final_mean'] for exp in experiments]
    
    bars = plt.bar(range(len(names)), means, color='steelblue', alpha=0.7)
    
    # Colorer la meilleure en vert
    best_idx = means.index(max(means))
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(1.0)
    
    plt.xticks(range(len(names)), [n.replace('_', '\n') for n in names], rotation=45, ha='right')
    plt.ylabel('Score Final (Mean 100)', fontsize=12)
    plt.title('Performance Finale par Expérience', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_final_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {output_dir}/comparison_final_scores.png")
    
    # 3. Sauvegarder le tableau en CSV
    summary_data = []
    for exp in experiments:
        summary_data.append({
            'Expérience': exp['name'],
            'Mean100_final': exp['final_mean'],
            'Max_score': exp['max_score'],
            'Best_eval': exp['best_eval'],
            'Total_episodes': exp['total_episodes'],
            **exp['info']
        })
    
    df_summary = pd.DataFrame(summary_data)
    csv_path = output_dir / 'experiments_summary.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"  ✅ {csv_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ COMPARAISON TERMINÉE")
    print(f"{'='*60}\n")
    print(f"📁 Résultats: {output_dir}/")
    print(f"\n💡 Recommandation: Continuez avec les paramètres de '{best_exp['name']}'")
    print(f"   pour un entraînement plus long (1-2M frames)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple training experiments")
    parser.add_argument("--dir", type=str, default="experiments",
                       help="Directory containing experiment folders")
    parser.add_argument("--output", type=str, default="reports/experiments",
                       help="Output directory for comparison plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"❌ Dossier non trouvé: {args.dir}")
        print("\n💡 Lancez d'abord: bash quick_experiments.sh")
    else:
        compare_experiments(args.dir, args.output)
