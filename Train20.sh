BASE_DIR="/path/to/StructureAsSearch"
SCRIPT="train.py"
# Parameters
NUM_OF_NODES=20
LR=1e-3
EPOCHS=300
N_ITER=60
SHIFT=-1
TAU_VALUES=(2.0 3.0 4.0 5.0)
NOISE_SCALES=(0.005 0.01 0.05 0.1 0.2 0.3)
for TAU in "${TAU_VALUES[@]}"; do
  for NOISE_SCALE in "${NOISE_SCALES[@]}"; do
    # Output file
    OUTPUT_FILE="${BASE_DIR}/Log/TSPSize${NUM_OF_NODES}Iter${N_ITER}Tau${TAU}NS${NOISE_SCALE}SHIFT${SHIFT}_DS.out"
    # SLURM job configuration
    echo "Running with tau=${TAU}, n_iter=${N_ITER}"
    sbatch <<EOF
#!/bin/bash


# ===== Customize your SLURM configuration below =====
# #SBATCH --job-name=...
# #SBATCH --mem=...
# #SBATCH --time=...
# #SBATCH -p ...
# #SBATCH --gres=gpu:...
# #SBATCH -o ...
# ================================================

cd ${BASE_DIR}
python -u ${SCRIPT}  --lr ${LR}  --epochs ${EPOCHS}\
     --n_iter ${N_ITER} --tau ${TAU} --noise_scale ${NOISE_SCALE} --seed 42 \
    --optimizer adam --weight_decay 1e-4 \
    --use_scheduler --warmup_epochs 15 \
    --early_stopping --patience 50 \
    --adaptive_grad_clip \
    --train_data data/tsp_${NUM_OF_NODES}_uniform_train.pt \
    --val_data data/tsp_${NUM_OF_NODES}_uniform_val.pt \
    --visualize --viz_freq 30 --hidden_dim 128 --shift ${SHIFT}
EOF

  done
done

