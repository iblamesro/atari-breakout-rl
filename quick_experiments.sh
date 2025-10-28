#!/bin/bash

# Script pour lancer plusieurs expériences courtes
# Chaque run est sauvegardé dans un dossier unique avec timestamp

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="experiments"

echo "🧪 LANCEMENT D'EXPÉRIENCES MULTIPLES"
echo "===================================="
echo ""

# Créer le dossier d'expériences
mkdir -p $BASE_DIR

# Fonction pour lancer un run
run_experiment() {
    local name=$1
    local frames=$2
    local batch_size=$3
    local lr=$4
    local buffer=$5
    
    local run_dir="${BASE_DIR}/${name}_${TIMESTAMP}"
    
    echo ""
    echo "🚀 Lancement: $name"
    echo "   Frames: $frames"
    echo "   Batch: $batch_size"
    echo "   LR: $lr"
    echo "   Buffer: $buffer"
    echo "   Dossier: $run_dir"
    echo ""
    
    python -m src.train_improved \
        --total-frames $frames \
        --save-dir $run_dir \
        --batch-size $batch_size \
        --lr $lr \
        --buffer-size $buffer \
        --eval-every $(($frames / 2)) \
        --seed $RANDOM
    
    # Sauvegarder un résumé
    echo "Run: $name" > ${run_dir}/info.txt
    echo "Frames: $frames" >> ${run_dir}/info.txt
    echo "Batch size: $batch_size" >> ${run_dir}/info.txt
    echo "Learning rate: $lr" >> ${run_dir}/info.txt
    echo "Buffer size: $buffer" >> ${run_dir}/info.txt
    echo "Timestamp: $TIMESTAMP" >> ${run_dir}/info.txt
    
    echo "✅ $name terminé!"
    echo ""
}

# ============================================
# EXPÉRIENCES RAPIDES (250k frames chacune)
# ============================================

echo "📊 Stratégie: 4 runs de 250k frames (~15-20 min chacun)"
echo ""

# Expérience 1: Baseline rapide
run_experiment "exp1_baseline" 250000 32 1e-4 100000

# Expérience 2: Learning rate plus élevé
run_experiment "exp2_highLR" 250000 32 2.5e-4 100000

# Expérience 3: Batch size plus grand
run_experiment "exp3_bigbatch" 250000 64 2.5e-4 100000

# Expérience 4: Buffer plus grand
run_experiment "exp4_bigbuffer" 250000 64 2.5e-4 200000

echo ""
echo "🎉 TOUTES LES EXPÉRIENCES TERMINÉES!"
echo "===================================="
echo ""
echo "📁 Résultats sauvegardés dans: $BASE_DIR/"
echo ""
echo "📊 Pour comparer les résultats:"
echo "   python compare_experiments.py --dir $BASE_DIR"
echo ""
