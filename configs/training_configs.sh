# Configurations optimisées pour l'entraînement Breakout DDQN

## Configuration Baseline (Original)
# Rapide mais performances limitées
python -m src.train_breakout_ddqn_per \
    --env ALE/Breakout-v5 \
    --total-frames 500000 \
    --seed 1 \
    --save-dir runs_baseline

## Configuration Optimisée (Recommended)
# Balance entre temps et performance
python -m src.train_improved \
    --env ALE/Breakout-v5 \
    --total-frames 3000000 \
    --seed 42 \
    --save-dir runs_optimized \
    --lr 1e-4 \
    --gamma 0.99 \
    --batch-size 32 \
    --buffer-size 500000 \
    --min-replay-size 80000 \
    --target-update-tau 0.001 \
    --epsilon-start 1.0 \
    --epsilon-final 0.01 \
    --epsilon-decay 1500000 \
    --eval-every 50000

## Configuration Aggressive
# Pour maximiser les performances (plus long)
python -m src.train_improved \
    --env ALE/Breakout-v5 \
    --total-frames 10000000 \
    --seed 42 \
    --save-dir runs_aggressive \
    --lr 6.25e-5 \
    --gamma 0.99 \
    --batch-size 32 \
    --buffer-size 1000000 \
    --min-replay-size 100000 \
    --target-update-tau 0.0005 \
    --epsilon-start 1.0 \
    --epsilon-final 0.01 \
    --epsilon-decay 3000000 \
    --eval-every 100000

## Configuration Fast Debug
# Pour tester rapidement
python -m src.train_improved \
    --env ALE/Breakout-v5 \
    --total-frames 100000 \
    --seed 42 \
    --save-dir runs_debug \
    --lr 1e-4 \
    --batch-size 32 \
    --buffer-size 50000 \
    --min-replay-size 10000 \
    --eval-every 10000

## Évaluation des modèles
# Évaluer un modèle entraîné
python -m src.eval \
    --model-path runs_optimized/best.pt \
    --n-episodes 50 \
    --epsilon 0.01 \
    --output results/eval_optimized.csv

## Visualisation
# Regarder l'agent jouer
python -m src.watch_agent \
    --model-path runs_optimized/best.pt \
    --episodes 5 \
    --fps 60
