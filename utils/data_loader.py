import json
import torch
import random
from torch.utils.data import Dataset  # , Dataloader
import numpy as np
from math import radians, cos, sin, asin, sqrt
class TrajectoryDataset(Dataset):
    def __init__(self, dataset, coder_data):
  
        if dataset == 'Tecent' or dataset == 'Shenzhen':
     
            selected_trajs = coder_data
  
            encoded_trajs = []
            for traj in selected_trajs:
                unique_elements, indices = np.unique(traj, return_index=True)
                sorted_indices = np.argsort(indices)
                encoded_traj = np.zeros_like(traj)
                for new_index, original_index in enumerate(sorted_indices):
                    encoded_traj[traj == unique_elements[original_index]] = new_index
                encoded_trajs.append(encoded_traj)
            encoded_trajs = np.array(encoded_trajs)
            # distances= np.concatenate((np.zeros((is_same.shape[0],1)),is_same.astype(np.int32)),axis=1).astype(np.float32)
            all_distances=torch.tensor(encoded_trajs).unsqueeze(dim=1)
            self.travel_location=torch.max(all_distances)+1

            home_locations = []
            for traj in selected_trajs:
                if len(traj) == 168:
                    # 使用NumPy的切片功能提取每24个点的前6个点
                    first_six_positions = traj.reshape(-1, 24)[:, :6].reshape(-1)
                    # 计算每组前6个点中出现次数最多的点
                    home_location = np.bincount(first_six_positions).argmax()
                    home_locations.append(home_location)
                elif len(traj) == 24:
                    first_six_positions = traj[:6]
                    home_location = np.bincount(first_six_positions).argmax()
                    home_locations.append(home_location)
            home_locations = np.array(home_locations)
            home_locations = torch.tensor(home_locations).long()

            all_data = torch.tensor(selected_trajs).to(torch.float).unsqueeze(dim=1)

        else:
            all_data = []
            for traj in coder_data:
                x = torch.tensor(traj)  #, dtype=torch.long
                all_data.append(x)
            all_data = torch.stack(all_data)
            
        self.data=torch.cat((all_data,all_distances),dim=1)
        self.home_locations=home_locations


    def __getitem__(self, index):
        return self.data[index],self.home_locations[index]

    def __len__(self):
        return len(self.data)

def load_trajs(dataset='TDrive', batch_size=8, num_workers=4,flag='train',length=168):
    if dataset == 'Tecent':
        if flag=='train':
            base_path = './datasets/Tecent'
            file_name = f'Tecent_Train_{length}.npy'
        elif flag=='test':
            base_path = './datasets/Tecent'
            file_name = f'Tecent_Test_{length}.npy'
        elif flag=='eval':
            base_path = './datasets/Tecent'
            file_name = f'Tecent_Eval_{length}.npy'
        file_path = f'{base_path}/{file_name}'
        trajs = np.load(file_path).astype(np.int32)
    elif dataset == 'Shenzhen':
        if flag=='train':
            base_path = './datasets/Shenzhen'
            file_name = f'Shenzhen_Train_{length}.npy'
        elif flag=='test':
            base_path = './datasets/Shenzhen'
            file_name = f'Shenzhen_Test_{length}.npy'
        elif flag=='eval':
            base_path = './datasets/Shenzhen'
            file_name = f'Shenzhen_Eval_{length}.npy'
        file_path = f'{base_path}/{file_name}'
        trajs = np.load(file_path).astype(np.int32)
    else:
        base_path = './datasets'
        file_name = f'{dataset}_Train.json' if train else f'{dataset}_Eval.json'
        file_path = f'{base_path}/{dataset}/{file_name}'
        with open(file_path, 'r') as file:
            trajs = json.loads(file.read())
    
    # trajs = trajs
    # random.shuffle(trajs)
    dataset = TrajectoryDataset(dataset=dataset, coder_data=trajs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    travel_location=dataset.travel_location
    return dataloader,travel_location





if __name__ == '__main__':
    data=np.load('./datasets/Shenzhen/Shenzhen_trajs_20w.npy')

    data_selected=[]
    segment_length=24
    valid_starts = range(0, len(data[0,:])-2*segment_length+1,24)
    for i in range(len(data)):
        for j in range(len(valid_starts)):
            data_selected.append(data[i][ valid_starts[j]: valid_starts[j]+segment_length])
    data_selected=np.array(data_selected)
    # 随机选择50%的数据
    num_samples = len(data_selected)
    selected_indices = random.sample(range(num_samples), int(num_samples * 0.5))
    data_selected = data_selected[selected_indices]

    num_total = len(data_selected)
    indices = list(range(num_total))
    random.shuffle(indices)
    split1 = int(num_total * 0.6)
    split2 = int(num_total * 0.7)
    train_indices = indices[:split1]
    val_indices = indices[split1:split2]
    test_indices = indices[split2:]
    # 保存训练数据
    train_data = data_selected[train_indices]

    np.save('./datasets/Shenzhen/Shenzhen_Train.npy', train_data)
    # 保存验证数据 
    val_data = data_selected[val_indices]
    np.save('./datasets/Shenzhen/Shenzhen_Eval.npy', val_data)
    # 保存测试数据
    test_data = data_selected[test_indices]
    np.save('./datasets/Shenzhen/Shenzhen_Test.npy', test_data)
    # data=np.load('/home/yxluo/research/TrajGDM/datasets/Shenzhen/Shenzhen_trajs_20w.npy')
    # num_total = len(data)
    # indices = list(range(num_total))
    # random.shuffle(indices)
    # split1 = int(num_total * 0.6)
    # split2 = int(num_total * 0.7)
    # train_indices = indices[:split1]
    # val_indices = indices[split1:split2]
    # test_indices = indices[split2:]
    # # 保存训练数据
    # train_data = data[train_indices]
    # np.save('/home/yxluo/research/TrajGDM/datasets/Shenzhen/Shenzhen_Train.npy', train_data)   
    
    # # 保存验证数据 
    # val_data = data[val_indices]
    # np.save('/home/yxluo/research/TrajGDM/datasets/Shenzhen/Shenzhen_Eval.npy', val_data)
    # # 保存测试数据
    # test_data = data[test_indices]
    # np.save('/home/yxluo/research/TrajGDM/datasets/Shenzhen/Shenzhen_Test.npy', test_data)
