import time
import argparse
import numpy as np
import torch
import sys
import os
import json
import random
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tqdm import tqdm
# Now import your modules
from utils.data_loader import load_trajs
from utils.metrics_movesim import evaluate
from models.diffusion import TrajectoryDiffusion
from models.dit import TravDit
from models.auto_encoder_decoder import AutoEncoderDecoder
from autoregressive.model import ARModel


import warnings
warnings.filterwarnings('ignore')
from torch.cuda.amp import autocast, GradScaler
def load_data(args):
    dataloader_train,travel_location_train = load_trajs(dataset=args.dataset,batch_size=args.batch_size,num_workers=args.num_workers,flag='train',length=args.max_length)
    dataloader_eval,travel_location_eval = load_trajs(dataset=args.dataset,batch_size=args.num_evaluation,num_workers=args.num_workers,flag='eval',length=args.max_length)
    dataloader_test,travel_location_test = load_trajs(dataset=args.dataset,batch_size=args.num_evaluation,num_workers=args.num_workers,flag='test',length=args.max_length)
    travel_location=max(travel_location_train,travel_location_eval,travel_location_test)
    return dataloader_train,dataloader_eval,dataloader_test,travel_location


def trainer(args,dataloader_train,dataloader_eval,travel_location):
    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    
    # 设置默认张量类型为 float32
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # travel_location=max(travel_location_train,travel_location_eval,travel_location_test)

    if args.dataset=='Shenzhen':
        num_loc=2236

    elif args.dataset=='Tecent':
        num_loc=900

    else:
        raise NotImplementedError(args.dataset)

    # metrics=TrajectoryMetrics()
    auto_encoder_decoder = AutoEncoderDecoder(vocab_size=num_loc+1,  # (最大位置ID) + 1(mask token)+sos
                      d_model=512,
                      nhead=8,
                      num_encoder_layers=2,
                      dim_feedforward=2048 ).to(device)
    
    pretrained_model = torch.load(f'./pre_train/checkpoints/{args.dataset}_pretrain_best_{args.max_length}.pt', map_location=device)
    auto_encoder_decoder.load_state_dict(pretrained_model['model_state_dict'])
    

    for param in auto_encoder_decoder.parameters():
        param.requires_grad = False
    

    auto_encoder_decoder.eval()

    trajgenerator = TravDit(num_location=num_loc, location_embedding=args.num_hidden,num_head=args.TrajGenerator_heads,input_len=args.max_length,travel_location=travel_location,seed=args.seed).to(device)
    trajgenerator = trajgenerator.type(torch.float32)

    traj_diffusion = TrajectoryDiffusion(model=trajgenerator,encoder_decoder=auto_encoder_decoder,linear_start=0.00085, linear_end=0.0120,full_n_steps=2000,device=device)
    traj_diffusion = traj_diffusion.type(torch.float32)

    optimizer = torch.optim.Adam(trajgenerator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    scaler = GradScaler()
    trajgenerator.train()

    patience = args.patience  # 容忍多少个评估周期没有改善
    patience_counter = 0

    print('Start Training......')

    # relative_improvement_best = float('inf')

    for e in range(args.epoch):
        trajgenerator.train()
        traj_diffusion.train()
        train_loss = []
        trajs_list = []
        t1 = time.time()
        for i,(x,home_locations) in enumerate(dataloader_train):
            x = x.to(device)
            home_locations = home_locations.to(device)
            optimizer.zero_grad()
            
            with autocast():
                loss = traj_diffusion.generation_training(x[:,0],home_locations,x[:,1])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.append(float(loss.cpu().detach()))
            trajs_list.append(x.cpu())
            
        t2 = time.time()
        print('epoch ', e, ' minutes ', round((t2 - t1) / 60), 
              ' training loss ', np.mean(train_loss), 
              ' learning rate ', optimizer.param_groups[0]['lr'],
              flush=True)
        scheduler.step()
        if e != 0 and e % args.eval_epoch == 0:
            print('Evaluating......')
            t1 = time.time()
            with torch.no_grad(), autocast():
                trajgenerator.eval()
                traj_diffusion.eval()
                auto_encoder_decoder.eval()
                trajs_generated = []
                trajs_real = []
                for i, (x,home_locations) in tqdm(enumerate(dataloader_eval), desc='Evaluating', total=len(dataloader_eval)):
                    x = x.to(device).long()
                    home_locations = home_locations.to(device)
                    # 生成轨迹
                    trajs = traj_diffusion.TrajGenerating(num_samples=args.num_evaluation, home_location=home_locations, travel_pattern=x[:, 1])

                    trajs_generated.append(trajs.long())
                    trajs_real.append(x[:, 0, :].long())
                
            # 合并所有生成的和真实的轨迹
                trajs_generated = torch.cat(trajs_generated, dim=0).detach().cpu().numpy()
                trajs_real = torch.cat(trajs_real, dim=0).detach().cpu().numpy()
                
                # 评估指标
                d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, cpc = evaluate(args.dataset,args.max_length,trajs_real, trajs_generated)

                # 初始化历史最佳指标（第一次运行时需要定义）
                if 'best_metrics' not in locals():
                    best_metrics = [1, 1]
                
                # 逐个指标计算相对变化
                metrics = [d_jsd, g_jsd]
                relative_improvements = [
                    (metric - best_metric) / (best_metric + 1e-8)
                    for metric, best_metric in zip(metrics, best_metrics)
                ]
                mean_relative_improvement = -np.mean(relative_improvements)
                
                print('JSD:', d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, 'CPC:', cpc)
                # print('Relative Improvements:', relative_improvements, 'Mean:', mean_relative_improvement)
            
            t2 = time.time()
            print('Evaluating time: ', round((t2 - t1) / 60), ' minutes')
            
            # 保存模型
            trained_models_dir = './TrainedModels/'
            if not os.path.exists(trained_models_dir):
                os.makedirs(trained_models_dir)


            if mean_relative_improvement>0 :
                patience_counter = 0
                best_metrics = metrics  # 更新每个指标的最佳值
                torch.save(
                    traj_diffusion, 
                    os.path.join(trained_models_dir, f'{args.model_name}_{args.dataset}_{args.max_length}.pkl')
                )
                print('Update model parameters!')
            else:
                patience_counter += 1
                print(f'Early stopping counter: {patience_counter}/{patience}')

            if patience_counter >= patience:
                print('Early stopping triggered!')
                break
def test(args,dataloader_test,travel_location):

    if args.dataset=='Shenzhen':
        num_loc=2236        
    elif args.dataset=='Tecent':
        num_loc=900
    else:
        raise NotImplementedError(args.dataset)
    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    # 加载训练好的模型
    ar_model=ARModel(vocab_size=num_loc,  # (最大位置ID) + 1(mask token)+sos
                      travel_location=travel_location,
                      d_model=512,
                      nhead=8,
                      num_decoder_layers=4,
                      dim_feedforward=2048).to(device)
    ar_model.load_state_dict(torch.load('./autoregressive/checkpoints/'+args.dataset+'_travel_pattern_best_'+str(args.max_length)+'.pkl', map_location=device)['model_state_dict'])

    ar_model.eval()


    trained_models_dir = './TrainedModels/'
    traj_diffusion = torch.load(os.path.join(trained_models_dir, 
                                        f'{args.model_name}_{args.dataset}_{args.max_length}.pkl'))
    
    # 设置为评估模式
    traj_diffusion.eval()
    
    print('Start Testing......')
    t1 = time.time()
    
    with torch.no_grad(), autocast():
        trajs_generated = []
        trajs_real = []
        
        # 在测试集上生成轨迹
        for i, (x,home_locations) in tqdm(enumerate(dataloader_test), desc='测试中', total=len(dataloader_test)):
            x = x.to(device).long()
            home_locations = home_locations.to(device)
            output,ys = ar_model(condition=home_locations.to(torch.long),tgt=None,batch_size=args.batch_size)
            travel_pattern=ys.long()

            # 生成轨迹
            trajs = traj_diffusion.TrajGenerating(num_samples=args.num_evaluation, 
                                                home_location=home_locations,
                                                travel_pattern=travel_pattern)
            
            trajs_generated.append(trajs.long())
            trajs_real.append(x[:, 0, :].long())
        
        # 合并所有生成的和真实的轨迹
        trajs_generated = torch.cat(trajs_generated, dim=0).detach().cpu().numpy()
        trajs_real = torch.cat(trajs_real, dim=0).detach().cpu().numpy()
        # 保存生成的轨迹和真实轨迹
        save_dir = './results/trajectories/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        np.save(os.path.join(save_dir, f'{args.dataset}_generated_{args.max_length}_half.npy'), trajs_generated)
        np.save(os.path.join(save_dir, f'{args.dataset}_real_{args.max_length}.npy'), trajs_real)
        print(f'轨迹数据已保存至 {save_dir}')
        # 计算评估指标
        d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, cpc = evaluate(args.dataset,args.max_length,trajs_real, trajs_generated)
        
        print('JSD:', d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, 'CPC:', cpc)
        
    t2 = time.time()
    print('Test time : ', round((t2 - t1) / 60), '  minutes')

    results = {
        'distance_jsd': float(d_jsd),
        'geometry_jsd': float(g_jsd),
        'duration_jsd': float(du_jsd),
        'position_jsd': float(p_jsd),
        'length_jsd': float(l_jsd),
        'frequency_jsd': float(f_jsd),
        'cpc': float(cpc)
    }
    return results
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--model_name', default='Traveller', type=str)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--cuda', default="0", type=str)
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--dataset', default='Tecent', type=str)  
    parser.add_argument('--batch_size', default=512, type=int) 
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_hidden', default=512, type=int)
    parser.add_argument('--TrajGenerator_Translayers', default=4, type=int)
    parser.add_argument('--TrajGenerator_heads', default=8, type=int)
    parser.add_argument('--num_evaluation', default=512, type=int)
    parser.add_argument('--eval_epoch', default=10, type=int)
    parser.add_argument('--max_length', default=24, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--home_condition', action='store_true', default=True)
    parser.add_argument('--distance_condition', action='store_true', default=True)
    args = parser.parse_args()

    dataloader_train,dataloader_eval,dataloader_test,travel_location = load_data(args)
    print('train_len:',len(dataloader_train)*args.batch_size)
    print('eval_len:',len(dataloader_eval)*args.num_evaluation)
    print('test_len:',len(dataloader_test)*args.num_evaluation)
    # trainer(args,dataloader_train,dataloader_eval,travel_location)

    result=test(args,dataloader_test,travel_location)
    print('d_jsd:',result['distance_jsd'])
    print('du_jsd:',result['duration_jsd'])
    print('p_jsd:',result['position_jsd'])
    print('l_jsd:',result['length_jsd'])
    print('f_jsd:',result['frequency_jsd'])
    print('cpc:',result['cpc'])

