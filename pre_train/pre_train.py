import torch
import torch.nn as nn
from model import BERTModel
from pre_train_dataloader import TrajectoryDataset
import argparse
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import os


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

def train_bert(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载训练集和验证集
    if args.dataset=='Tecent':
        vocab_size=900
    elif args.dataset=='Shenzhen':
        vocab_size=2236
    data_path_train = f'./datasets/{args.dataset}/{args.dataset}_Train_{args.length}.npy'
    data_path_eval = f'./datasets/{args.dataset}/{args.dataset}_Eval_{args.length}.npy'
    train_loader = DataLoader(TrajectoryDataset(dataset=args.dataset, data_path=data_path_train, train=True, mask_ratio=0.15,mask_token_idx=vocab_size), 
                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(TrajectoryDataset(dataset=args.dataset, data_path=data_path_eval, train=False, mask_ratio=0.15,mask_token_idx=vocab_size), 
                          batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # 初始化模型
    model = BERTModel(vocab_size=vocab_size+1,  # (最大位置ID) + 1(mask token)+1(sos)
                      d_model=512,
                      nhead=8,
                      num_encoder_layers=2,
                      dim_feedforward=2048,
                      mask_token_idx=vocab_size).to(device)
                      
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # 添加学习率余弦衰减
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 初始化 early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    best_val_loss = float('inf')
    
    print('Start Training......')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        for batch_idx, (masked_seq, target_seq) in enumerate(train_loader):
            masked_seq = masked_seq.to(device)
            target_seq = target_seq.to(device)
            
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with autocast():
                # 前向传播
                output = model(masked_seq)
                
                # 计算损失
                loss = criterion(output.reshape(-1, output.size(-1)), 
                               target_seq.reshape(-1))
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Train Loss: {loss.item():.4f}')
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        total_token_acc = 0
        total_seq_acc = 0
        
        with torch.no_grad():
            for masked_seq, target_seq in val_loader:
                masked_seq = masked_seq.to(device)
                target_seq = target_seq.to(device)
                
                with autocast():
                    # 自回归生成
                    output = model(masked_seq)
                    output = output.float()
                    
                    # 计算损失
                    loss = criterion(output.reshape(-1, output.size(-1)), 
                                  target_seq.reshape(-1))
                    
                    # 计算token准确率
                    pred_tokens = output.argmax(dim=-1)
                    correct_tokens = (pred_tokens == target_seq).float()
                    token_acc = correct_tokens.mean().item()
                    
                    # 计算序列准确率
                    seq_acc = (correct_tokens.sum(dim=1) == correct_tokens.size(1)).float().mean().item()
                    
                    total_val_loss += loss.item()
                    total_token_acc += token_acc
                    total_seq_acc += seq_acc
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_token_acc = total_token_acc / len(val_loader)
        avg_seq_acc = total_seq_acc / len(val_loader)
        
        print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Token Acc: {avg_token_acc:.4f}, '
              f'Seq Acc: {avg_seq_acc:.4f}, '
              f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 更新学习率
        scheduler.step()

        
        # 检查是否需要early stop并保存最佳模型
        early_stopping(avg_val_loss)
        if avg_val_loss < best_val_loss:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'{args.save_path}/{args.dataset}_pretrain_best_{args.length}.pt')
            print(f'Saved best model with validation loss: {avg_val_loss:.4f}')
            
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

if __name__ == '__main__':
    args = argparse.Namespace()
    args.batch_size = 512
    args.dataset='Tecent'
    args.lr = 1e-4 
    args.epochs = 100
    args.num_latents = 64
    args.num_workers = 4
    args.length = 24
    args.patience = 3  # early stopping 的耐心值
    args.save_path = './pre_train/checkpoints'  # 模型保存路径
    train_bert(args)
