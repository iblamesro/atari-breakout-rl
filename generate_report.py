"""
Script pour générer un rapport complet après l'entraînement
Génère tous les plots et statistiques nécessaires pour la présentation
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def generate_full_report(optimized_path="runs_optimized", baseline_path="runs", output_dir="reports"):
    """Génère le rapport complet avec tous les plots et statistiques."""
    
    print(f"\n{'='*60}")
    print(f"📊 GÉNÉRATION DU RAPPORT COMPLET")
    print(f"{'='*60}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. GÉNÉRER LES PLOTS OPTIMIZED
    print("🎨 Étape 1/5 : Génération des plots optimisés...")
    os.system(f"python plot_training.py --input {optimized_path}/training_log.csv --output {output_dir}/plots_optimized")
    
    # 2. GÉNÉRER LES PLOTS BASELINE
    print("\n🎨 Étape 2/5 : Génération des plots baseline...")
    os.system(f"python plot_training.py --input {baseline_path}/training_log.csv --output {output_dir}/plots_baseline")
    
    # 3. COMPARAISON
    print("\n📊 Étape 3/5 : Génération de la comparaison...")
    os.system(f"python compare_models.py --baseline {baseline_path}/training_log.csv --ultimate {optimized_path}/training_log.csv --output {output_dir}/comparison")
    
    # 4. STATISTIQUES FINALES
    print("\n📈 Étape 4/5 : Calcul des statistiques finales...")
    
    # Charger les logs
    df_opt = pd.read_csv(f"{optimized_path}/training_log.csv")
    df_base = pd.read_csv(f"{baseline_path}/training_log.csv")
    
    # Hyperparamètres
    with open(f"{optimized_path}/hyperparams.json", "r") as f:
        hyperparams = json.load(f)
    
    # Statistiques optimized
    episodes_opt = df_opt[df_opt["ep_return"].notna()]
    final_scores_opt = episodes_opt.tail(1000)["ep_return"].values if len(episodes_opt) >= 1000 else episodes_opt["ep_return"].values
    
    eval_opt = df_opt[df_opt["eval_mean"].notna()]
    best_eval_opt = eval_opt["eval_mean"].max() if not eval_opt.empty else 0
    
    # Statistiques baseline
    episodes_base = df_base[df_base["ep_return"].notna()]
    final_scores_base = episodes_base.tail(1000)["ep_return"].values if len(episodes_base) >= 1000 else episodes_base["ep_return"].values
    
    eval_base = df_base[df_base["eval_mean"].notna()]
    best_eval_base = eval_base["eval_mean"].max() if not eval_base.empty else 0
    
    # Créer le rapport texte
    report = []
    report.append("="*60)
    report.append("RAPPORT D'ENTRAÎNEMENT - BREAKOUT ATARI")
    report.append("="*60)
    report.append("")
    
    report.append("📊 CONFIGURATION")
    report.append("-" * 60)
    report.append(f"Total frames: {hyperparams['total_frames']:,}")
    report.append(f"Architecture: {'Deep (4 conv)' if hyperparams.get('deep_arch', False) else 'Standard (3 conv)'}")
    report.append(f"Learning rate: {hyperparams['lr']}")
    report.append(f"Batch size: {hyperparams['batch_size']}")
    report.append(f"Buffer size: {hyperparams['buffer_size']:,}")
    report.append(f"Target update tau: {hyperparams['target_update_tau']}")
    report.append(f"Epsilon final: {hyperparams['epsilon_final']}")
    report.append(f"Epsilon decay: {hyperparams['epsilon_decay']:,} frames")
    report.append("")
    
    report.append("🎯 RÉSULTATS - MODÈLE OPTIMISÉ")
    report.append("-" * 60)
    report.append(f"Total épisodes: {len(episodes_opt)}")
    report.append(f"Score moyen (1000 derniers eps): {np.mean(final_scores_opt):.2f} ± {np.std(final_scores_opt):.2f}")
    report.append(f"Score max (épisode): {np.max(episodes_opt['ep_return']):.1f}")
    report.append(f"Score min (épisode): {np.min(episodes_opt['ep_return']):.1f}")
    report.append(f"Meilleur score (évaluation): {best_eval_opt:.2f}")
    report.append("")
    
    report.append("📉 RÉSULTATS - BASELINE")
    report.append("-" * 60)
    report.append(f"Total épisodes: {len(episodes_base)}")
    report.append(f"Score moyen (1000 derniers eps): {np.mean(final_scores_base):.2f} ± {np.std(final_scores_base):.2f}")
    report.append(f"Score max (épisode): {np.max(episodes_base['ep_return']):.1f}")
    report.append(f"Score min (épisode): {np.min(episodes_base['ep_return']):.1f}")
    report.append(f"Meilleur score (évaluation): {best_eval_base:.2f}")
    report.append("")
    
    report.append("🚀 AMÉLIORATION")
    report.append("-" * 60)
    improvement = ((np.mean(final_scores_opt) - np.mean(final_scores_base)) / np.mean(final_scores_base)) * 100 if np.mean(final_scores_base) > 0 else 0
    report.append(f"Amélioration du score moyen: {improvement:+.1f}%")
    report.append(f"Gain absolu: {np.mean(final_scores_opt) - np.mean(final_scores_base):+.2f} points")
    report.append("")
    
    report.append("📁 FICHIERS GÉNÉRÉS")
    report.append("-" * 60)
    report.append(f"✅ Plots optimisés: {output_dir}/plots_optimized/")
    report.append(f"✅ Plots baseline: {output_dir}/plots_baseline/")
    report.append(f"✅ Comparaison: {output_dir}/comparison/")
    report.append(f"✅ Meilleur modèle: {optimized_path}/best.pt")
    report.append("")
    
    report.append("🎬 PROCHAINES ÉTAPES")
    report.append("-" * 60)
    report.append("1. Tester visuellement:")
    report.append(f"   python -m src.watch_agent --model-path {optimized_path}/best.pt --episodes 5")
    report.append("")
    report.append("2. Évaluer quantitativement:")
    report.append(f"   python -m src.eval --model-path {optimized_path}/best.pt --n-episodes 30")
    report.append("")
    report.append("3. Préparer la présentation avec les plots dans reports/")
    report.append("")
    report.append("="*60)
    
    # Sauvegarder le rapport
    report_text = "\n".join(report)
    report_path = output_dir / "RAPPORT_FINAL.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n✅ Rapport sauvegardé: {report_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate complete training report")
    parser.add_argument("--optimized", type=str, default="runs_optimized",
                       help="Path to optimized run directory")
    parser.add_argument("--baseline", type=str, default="runs",
                       help="Path to baseline run directory")
    parser.add_argument("--output", type=str, default="reports",
                       help="Output directory for report")
    
    args = parser.parse_args()
    
    generate_full_report(args.optimized, args.baseline, args.output)
