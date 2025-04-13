#!/bin/bash
set -x 
if [ ! -d "./logs" ]; then
    mkdir ./logs    
fi
dataset='Shenzhen'
python -u ./train_ar.py \
    --dataset $dataset \
    --lr 1e-4 >./logs/${dataset}_traveller_dit_ar_1km_24_1e-4.log


