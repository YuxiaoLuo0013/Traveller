#!/bin/bash
set -x 
if [ ! -d "./logs" ]; then
    mkdir .//logs    
fi
lr=1e-4
dataset='Tecent'
seed=2024
python -u ./run_trainer.py \
    --seed $seed \
    --dataset $dataset \
    --TrajGenerator_Translayers 4 \
    --TrajGenerator_heads 8 \
    --eval_epoch 10 \
    --batch_size 512 \
    --num_evaluation 512 \
    --epoch 200 \
    --lr $lr >./logs/${dataset}_traveller_DIT_step2000_ar_1km_24_${lr}_${seed}.log


