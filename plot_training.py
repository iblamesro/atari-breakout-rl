"""
Script pour générer les graphiques d'entraînement
Usage: python plot_training.py --input runs_optimized/training_log.csv --output reports/plots
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_training_curves(csv_path, output_dir="reports/plots"):
    """Génère les graphiques d'entraînement depuis un fichier CSV."""
    
    # Charger les données
    print(f"📊 Chargement des données depuis {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Nettoyer les données
    df["mean100"].fillna(0, inplace=True)
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["eval_mean"] = pd.to_numeric(df["eval_mean"], errors="coerce")
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎨 Génération des graphiques...")
    
    # 1. Courbe de récompenses (épisodes + moyenne glissante)
    plt.figure(figsize=(14, 6))
    episodes_df = df[df["ep_return"].notna()]
    plt.plot(episodes_df["frame"], episodes_df["ep_return"], 
             alpha=0.3, color="lightblue", label="Episode Return")
    plt.plot(episodes_df["frame"], episodes_df["mean100"], 
             linewidth=2, color="blue", label="Mean 100 Episodes")
    
    # Ajouter les points d'évaluation si disponibles
    eval_df = df[df["eval_mean"].notna()]
    if not eval_df.empty:
        plt.scatter(eval_df["frame"], eval_df["eval_mean"], 
                   color="red", s=100, marker="o", label="Evaluation", zorder=5)
    
    plt.xlabel("Training Frames", fontsize=12)
    plt.ylabel("Episode Return", fontsize=12)
    plt.title("Training Progress - Reward Curve", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rewards_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_dir}/rewards_curve.png")
    
    # 2. Epsilon decay
    plt.figure(figsize=(14, 5))
    epsilon_df = df[df["epsilon"].notna()]
    plt.plot(epsilon_df["frame"], epsilon_df["epsilon"], 
             color="orange", linewidth=2)
    plt.xlabel("Training Frames", fontsize=12)
    plt.ylabel("Epsilon (ε)", fontsize=12)
    plt.title("Exploration Rate (Epsilon) Decay", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epsilon_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {output_dir}/epsilon_curve.png")
    
    # 3. Loss curve
    plt.figure(figsize=(14, 5))
    loss_df = df[df["loss"].notna()]
    if not loss_df.empty:
        # Moyenne glissante pour lisser la courbe
        window = min(100, len(loss_df) // 10)
        if window > 1:
            loss_smooth = loss_df["loss"].rolling(window=window, min_periods=1).mean()
            plt.plot(loss_df["frame"], loss_smooth, 
                    color="red", linewidth=2, label=f"Loss (smoothed {window})")
        else:
            plt.plot(loss_df["frame"], loss_df["loss"], 
                    color="red", alpha=0.7, label="Loss")
        
        plt.xlabel("Training Frames", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Loss", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {output_dir}/loss_curve.png")
    
    # 4. Scores d'évaluation avec barres d'erreur
    if not eval_df.empty and "eval_std" in eval_df.columns:
        plt.figure(figsize=(14, 6))
        eval_df_clean = eval_df[eval_df["eval_mean"].notna()]
        
        plt.errorbar(eval_df_clean["frame"], eval_df_clean["eval_mean"],
                    yerr=eval_df_clean["eval_std"], 
                    marker='o', markersize=8, linewidth=2, 
                    capsize=5, color="green", label="Evaluation Score")
        
        plt.xlabel("Training Frames", fontsize=12)
        plt.ylabel("Evaluation Score (mean ± std)", fontsize=12)
        plt.title("Evaluation Performance Over Training", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eval_scores.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ {output_dir}/eval_scores.png")
    
    # 5. Learning rate (si disponible)
    if "lr" in df.columns:
        lr_df = df[df["lr"].notna()]
        if not lr_df.empty:
            plt.figure(figsize=(14, 5))
            plt.plot(lr_df["frame"], lr_df["lr"], 
                    color="purple", linewidth=2)
            plt.xlabel("Training Frames", fontsize=12)
            plt.ylabel("Learning Rate", fontsize=12)
            plt.title("Learning Rate Schedule", fontsize=14, fontweight="bold")
            plt.yscale("log")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/lr_schedule.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  ✅ {output_dir}/lr_schedule.png")
    
    # Statistiques finales
    print(f"\n📈 Statistiques:")
    print(f"  • Total frames: {df['frame'].max():,}")
    print(f"  • Total episodes: {df[df['episode'].notna()]['episode'].max():.0f}")
    print(f"  • Meilleur score (mean100): {episodes_df['mean100'].max():.2f}")
    if not eval_df.empty:
        print(f"  • Meilleur score (eval): {eval_df['eval_mean'].max():.2f}")
    
    print(f"\n✅ Tous les graphiques sauvegardés dans {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training plots")
    parser.add_argument("--input", type=str, default="runs_optimized/training_log.csv",
                       help="Path to training log CSV")
    parser.add_argument("--output", type=str, default="reports/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Fichier non trouvé: {args.input}")
        print(f"Fichiers disponibles:")
        for root, dirs, files in os.walk("."):
            for f in files:
                if f == "training_log.csv":
                    print(f"  • {os.path.join(root, f)}")
    else:
        plot_training_curves(args.input, args.output)
