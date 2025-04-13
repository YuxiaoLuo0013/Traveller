import torch
import torch.nn as nn
from model import ARModel
from dataloader import load_trajs
import argparse
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Traveller.utils.metrics_movesim import evaluate_travel_pattern


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_ar(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练集和验证集
    dataloader,travel_location_train = load_trajs(dataset=args.dataset,batch_size=args.batch_size,num_workers=args.num_workers,flag='train',length=args.max_length)
    dataloader_eval,travel_location_eval = load_trajs(dataset=args.dataset,batch_size=args.batch_size,num_workers=args.num_workers,flag='eval',length=args.max_length)
    dataloader_test,travel_location_test = load_trajs(dataset=args.dataset,batch_size=args.batch_size,num_workers=args.num_workers,flag='test',length=args.max_length)

    travel_location=max(travel_location_train,travel_location_eval,travel_location_test)
    # 初始化模型
    if args.dataset=='Shenzhen':
        vocab_size=2236
    else:
        vocab_size=900
    model = ARModel(vocab_size=vocab_size,  # 4422(最大位置ID) + 1(mask token) 2224
                      travel_location=travel_location,
                      d_model=512,
                      nhead=8,
                      num_decoder_layers=4,
                      dim_feedforward=2048,
                      max_len=args.max_length).to(device)
                      
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # 添加学习率余弦衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 初始化 early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    best_val_loss = float('inf')
    best_du_jsd=float('inf')
    best_p_jsd=float('inf')
    
    print('Start Training......')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        for batch_idx, (x,home_locations) in enumerate(dataloader):
            x = x.to(device)
            home_locations = home_locations.to(device)
            travel_pattern=x[:,1].to(device)
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                # 前向传播
                output = model(condition=home_locations,tgt=travel_pattern,batch_size=args.batch_size)
                
                # 计算损失
                loss = criterion(output.reshape(-1, output.size(-1)), 
                               travel_pattern.reshape(-1).long())
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Train Loss: {loss.item():.4f}')
        
        avg_train_loss = total_train_loss / len(dataloader)

        # 验证阶段
        model.eval()
        # best_du_jsd=1
        # best_p_jsd=1
        
        with torch.no_grad():
            for (x,home_locations) in dataloader_eval:
                home_locations = home_locations.to(device)
                travel_pattern=x[:,1].to(device)
                generated_travel_pattern=[]
                real_travel_pattern=[]
                with autocast():
                    # 自回归生成
                    output,ys = model(condition=home_locations,tgt=None,batch_size=args.batch_size)
                    # probabilities = torch.softmax(output, dim=-1)
                    ys=ys.long()
                    generated_travel_pattern.append(ys)
                    real_travel_pattern.append(travel_pattern)
            generated_travel_pattern=torch.cat(generated_travel_pattern,dim=0).detach().cpu().numpy()
            real_travel_pattern=torch.cat(real_travel_pattern,dim=0).detach().cpu().numpy()
            du_jsd,  p_jsd = evaluate_travel_pattern(args.dataset,args.max_length,real_travel_pattern,generated_travel_pattern)
            print(f'Epoch: {epoch}, '
                  f'Du JSD: {du_jsd:.4f}, '
                  f'P JSD: {p_jsd:.4f}')
        scheduler.step()

        
        #检查是否需要early stop并保存最佳模型
        early_stopping(du_jsd+p_jsd)
        if (du_jsd+p_jsd) < (best_du_jsd+best_p_jsd):   
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            
            best_du_jsd = du_jsd
            best_p_jsd = p_jsd
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': du_jsd+p_jsd,   
            }, f'{args.save_path}/{args.dataset}_travel_pattern_best_{args.max_length}.pkl')
            print(f'Saved best model with validation loss: {du_jsd+p_jsd:.4f}')
            
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == '__main__':
    args = argparse.Namespace()
    args.batch_size = 512
    args.lr = 1e-5 
    args.dataset='Tecent'
    args.epochs = 50
    args.max_length = 24
    args.num_latents = 64
    args.num_workers = 4
    args.patience = 10  # early stopping 的耐心值
    args.save_path = './autoregressive/checkpoints'  # 模型保存路径
    train_ar(args)
