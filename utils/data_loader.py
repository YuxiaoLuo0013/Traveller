import json
import torch
import random
from torch.utils.data import Dataset  # , Dataloader
import numpy as np
from math import radians, cos, sin, asin, sqrt
class TrajectoryDataset(Dataset):
    def __init__(self, dataset, coder_data,Sample):
  
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
            temporal_modes=torch.tensor(encoded_trajs).unsqueeze(dim=1)
            self.travel_location=torch.max(temporal_modes)+1

            # Calculate home locations
            if Sample == True:
                # Calculate overall home location distribution
                all_home_locations = []
                for traj in selected_trajs:
                    if len(traj) == 168:
                        first_six_positions = traj.reshape(-1, 24)[:, :6].reshape(-1)
                        home_location = np.bincount(first_six_positions).argmax()
                        all_home_locations.append(home_location)
                    elif len(traj) == 24:
                        first_six_positions = traj[:6]
                        home_location = np.bincount(first_six_positions).argmax()
                        all_home_locations.append(home_location)
                
                # Calculate distribution
                unique_locations, counts = np.unique(all_home_locations, return_counts=True)
                distribution = counts / np.sum(counts)
                
                # Sample from distribution
                home_locations = np.random.choice(unique_locations, size=len(selected_trajs), p=distribution)
            else:
                # Original home location calculation
                home_locations = []
                for traj in selected_trajs:
                    if len(traj) == 168:
                        first_six_positions = traj.reshape(-1, 24)[:, :6].reshape(-1)
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
            
        self.data=torch.cat((all_data,temporal_modes),dim=1)
        self.home_locations=home_locations


    def __getitem__(self, index):
        return self.data[index],self.home_locations[index]

    def __len__(self):
        return len(self.data)

def load_trajs(dataset='TDrive', batch_size=8, num_workers=4,flag='train',seed=42,length=24,sample=Flase):
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

    
    # trajs = trajs
    # random.shuffle(trajs)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = TrajectoryDataset(dataset=dataset, coder_data=trajs, sample=sample)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    travel_location=dataset.travel_location
    return dataloader,travel_location

