import pandas as pd
import matplotlib.pyplot as plt
import os

# Charger le fichier de log
df = pd.read_csv("runs/training_log.csv")

# Nettoyage / conversion
df["mean100"] = df["mean100"].fillna(0)
df["loss"] = pd.to_numeric(df["loss"], errors="coerce")

# Créer le dossier de sauvegarde si besoin
os.makedirs("reports/plots", exist_ok=True)

# === Plot Rewards ===
plt.figure(figsize=(10,5))
plt.plot(df["frame"], df["ep_return"], label="Reward par épisode", alpha=0.6)
plt.plot(df["frame"], df["mean100"], label="Moyenne mobile (100)", linewidth=2)
plt.xlabel("Frames")
plt.ylabel("Reward")
plt.legend()
plt.title("Courbe de Reward - Atari Breakout (DDQN)")
plt.grid(True)
plt.savefig("reports/plots/rewards_curve.png", dpi=300)
plt.close()  # <-- ferme la figure proprement

# === Plot Epsilon ===
plt.figure(figsize=(10,4))
plt.plot(df["frame"], df["epsilon"], label="Epsilon", color="orange")
plt.xlabel("Frames")
plt.ylabel("ε")
plt.title("Évolution de l'exploration (epsilon)")
plt.grid(True)
plt.savefig("reports/plots/epsilon_curve.png", dpi=300)
plt.close()

# === Plot Loss ===
plt.figure(figsize=(10,4))
plt.plot(df["frame"], df["loss"], label="Loss", color="red", alpha=0.7)
plt.xlabel("Frames")
plt.ylabel("Loss")
plt.title("Courbe de perte (TD Error)")
plt.grid(True)
plt.savefig("reports/plots/loss_curve.png", dpi=300)
plt.close()

print("✅ Graphiques enregistrés dans le dossier reports/plots/")
