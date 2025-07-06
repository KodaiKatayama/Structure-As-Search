BASE_DIR="/path/to/StructureAsSearch"
SCRIPT="train.py"
# Parameters
NUM_OF_NODES=100
LR=2e-3
EPOCHS=600
NLAYERS=4
#NOISE_SCALE=0.05
N_ITER=80
SHIFT=-1  # corresponds to coprime k=1 for V^k
TAU_VALUES=(5.0)
NOISE_SCALES=(0.005 0.01 0.02 0.03 0.05 0.1 0.2 0.3)
HIDDIM=512

# Loop through all combinations of tau and n_iter
for TAU in "${TAU_VALUES[@]}"; do
  for NOISE_SCALE in "${NOISE_SCALES[@]}"; do
    # Output file
    OUTPUT_FILE="${BASE_DIR}/Log/TSPSize${NUM_OF_NODES}LAYS${NLAYERS}Iter${N_ITER}Tau${TAU}NS${NOISE_SCALE}SHIFT${SHIFT}_DS.out"
    # SLURM job configuration
    echo "Running with tau=${TAU}, n_iter=${N_ITER}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --qos=low
#SBATCH --job-name=100Deep_shift${SHIFT}_tau${TAU}_iter${N_ITER}_NS_${NOISE_SCALE}
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
#SBATCH -p full
#SBATCH --gres=gpu:1
#SBATCH -o ${OUTPUT_FILE}
cd ${BASE_DIR}
python -u ${SCRIPT}  --lr ${LR}  --epochs ${EPOCHS} --distance_scale 5.0\
     --n_iter ${N_ITER} --tau ${TAU} --noise_scale ${NOISE_SCALE} --seed 42 \
    --optimizer adam --weight_decay 1e-4 \
    --use_scheduler --warmup_epochs 15 \
    --early_stopping --patience 50 \
    --adaptive_grad_clip \
    --visualize --viz_freq 30 \
    --train_data data/tsp_${NUM_OF_NODES}_uniform_train.pt \
    --val_data data/tsp_${NUM_OF_NODES}_uniform_val.pt --hidden_dim ${HIDDIM} --shift ${SHIFT} --n_layers ${NLAYERS} 
EOF

  done
done

