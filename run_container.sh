export WANDB_API_KEY=d5b58767372bab71b489d5f96c2e18504611721b
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

torchrun --nnodes=1 --nproc_per_node=1 train.py \
         --data_path "/gpfs/work5/0/jhstue005/JHS_data/CityScapes" \
         --resize_height 640 \
         --resize_width 1280 \
         --learning_rate 1e-3 \
         --batch_size 6 \
         --weight_decay 2e-4 \
         --epochs 105 \
         --patience 4 \
         --step_size 30 \
         --gamma 0.1