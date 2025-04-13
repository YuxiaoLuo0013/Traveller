#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
python -u ./pre_train.py \
    --lr 1e-4 \
    --epochs 10 \
    --batch_size 128 \
    --save_path ./logs \
    --data_path ./datasets/Tecent/Tecent.npy >./logs/bart_pretrain.log 


