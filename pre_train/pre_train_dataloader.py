import random
import torch
from torch.utils.data import Dataset
import numpy as np
class TrajectoryDataset(Dataset):
    def __init__(self, dataset, data_path, train=True, mask_ratio=0.15, mask_token_idx=901):
        random.seed(42)
        torch.manual_seed(42)
        


        selected_trajs = np.load(data_path)
    
        # 创建mask
        self.masks = torch.zeros_like(torch.tensor(selected_trajs))
        mask_indices = torch.rand_like(self.masks.float()) < mask_ratio
        self.masks[mask_indices] = 1
        
        # 获取原始序列和masked序列
        self.original_trajs = torch.tensor(selected_trajs).to(torch.long)
        self.masked_trajs = self.original_trajs.clone()
        self.masked_trajs[mask_indices] = mask_token_idx  # 用新的mask token ID
        

        
        if train:
            self.masked_trajs = self.masked_trajs
            self.original_trajs = self.original_trajs 
        else:
            self.masked_trajs = self.masked_trajs 
            self.original_trajs = self.original_trajs

    def __getitem__(self, index):
        return self.masked_trajs[index], self.original_trajs[index]

    def __len__(self):
        return len(self.original_trajs)