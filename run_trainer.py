import time
import argparse
import numpy as np
import torch
import sys
import os
import json
import random
import os
# Add the parent directory of TrajGDM to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tqdm import tqdm
# Now import your modules
from TrajGDM.utils.data_loader import load_trajs
from TrajGDM.utils.metrics import Evaluation_metrics, TrajectoryMetrics
from TrajGDM.utils.metrics_movesim import evaluate
from TrajGDM.models.diffusion import TrajectoryDiffusion
from TrajGDM.models.TrajGeneratorNetwork import TrajGeneratorNetwork_all
from TrajGDM.models.auto_encoder_decoder import AutoEncoderDecoder
from TrajGDM.train_travel_pattern.model import ARModel

from thop import profile
# from TrajGDM.models.diffusion_TrajGDM import TrajectoryDiffusion
# from TrajGDM.models.TrajGeneratorNetwork_TrajGDM import TrajGeneratorNetwork

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
    # 1. 首先设置 CUDA 设备
    torch.cuda.set_device(int(args.cuda))  # 使用 args.cuda 而不是 args.gpu

    # 2. 然后创建设备对象
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    
    # 设置默认张量类型为 float32
    torch.set_default_tensor_type(torch.FloatTensor)
    
    # travel_location=max(travel_location_train,travel_location_eval,travel_location_test)
    if args.dataset=='TDrive':
        num_loc=759
        maxi=27
    elif args.dataset=='Shenzhen':
        num_loc=2236
        maxi=0
    elif args.dataset=='Tecent':
        num_loc=900
        maxi=0
    else:
        raise NotImplementedError(args.dataset)

    # metrics=TrajectoryMetrics()
    auto_encoder_decoder = AutoEncoderDecoder(vocab_size=num_loc+1,  # (最大位置ID) + 1(mask token)+sos
                      d_model=512,
                      nhead=8,
                      num_encoder_layers=2,
                      num_decoder_layers=2,
                      dim_feedforward=2048 ).to(device)
    
    pretrained_model = torch.load(f'./pre_train/checkpoints/{args.dataset}_pretrain_best_{args.max_length}.pt', map_location=device)
    auto_encoder_decoder.load_state_dict(pretrained_model['model_state_dict'])
    
    # 锁住所有参数
    for param in auto_encoder_decoder.parameters():
        param.requires_grad = False
    
    # 设置为评估模式
    auto_encoder_decoder.eval()

    trajgenerator = TrajGeneratorNetwork_all(num_location=num_loc, location_embedding=args.num_hidden,maxi=maxi,num_head=args.TrajGenerator_heads,
                                         lstm_hidden=args.num_hidden, device=device,TrajGenerator_Translayers=args.TrajGenerator_Translayers,TrajGenerator_LSTMlayers=args.TrajGenerator_LSTMlayers,
                                         input_len=args.max_length,travel_location=travel_location,seed=args.seed).to(device)

    trajgenerator = trajgenerator.type(torch.float32)

    # 4. 确保模型确实在正确的设备上
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device specified: {device}")
    print(f"Model device: {next(trajgenerator.parameters()).device}")

    # 在创建 traj_diffusion 模型之前添加


    traj_diffusion = TrajectoryDiffusion(model=trajgenerator,encoder_decoder=auto_encoder_decoder,maxi=maxi,lab=2,linear_start=0.00085, linear_end=0.0120,full_n_steps=2000,home_condition=args.home_condition,distance_condition=args.distance_condition).to(device)

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
                for i, (x,home_locations) in tqdm(enumerate(dataloader_eval), desc='评估中', total=len(dataloader_eval)):
                    if i==0:
                        # 创建模型实例用于profile
                        dummy_model = traj_diffusion
                        # 使用 batch_size=1 来计算单样本推理复杂度
                        dummy_input = (1, home_locations[:1], x[:1, 1])  # batch_size=1
                        macs, params = profile(dummy_model, inputs=dummy_input)
                        print(f"Single sample MACs: {macs / 1e9:.2f} G")
                        print(f"Params: {params / 1e6:.2f} M")
                        print(f"Single sample FLOPs: {macs / 1e9:.2f} G")
                        
                        # 也可以计算实际batch size的复杂度
                        actual_input = (x.shape[0], home_locations, x[:, 1])
                        actual_macs, _ = profile(dummy_model, inputs=actual_input)
                        print(f"Batch {x.shape[0]} MACs: {actual_macs / 1e9:.2f} G")
                    x = x.to(device).long()
                    home_locations = home_locations.to(device)
                    # 生成轨迹
                    trajs = traj_diffusion.TrajGenerating(num_samples=x.shape[0], home_location=home_locations, travel_pattern=x[:, 1])

                    trajs_generated.append(trajs.long())
                    trajs_real.append(x[:, 0, :].long())
                
            # 合并所有生成的和真实的轨迹
                trajs_generated = torch.cat(trajs_generated, dim=0).detach().cpu().numpy()
                trajs_real = torch.cat(trajs_real, dim=0).detach().cpu().numpy()
                
                # 评估指标
                d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, cpc,fow_mape,pop_mape = evaluate(args.dataset,args.max_length,trajs_real, trajs_generated)

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
                
                print('JSD:', d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, 'CPC:', cpc, 'FOW_MAPE:', fow_mape, 'POP_MAPE:', pop_mape)
                # print('Relative Improvements:', relative_improvements, 'Mean:', mean_relative_improvement)
            
            t2 = time.time()
            print('Evaluating time: ', round((t2 - t1) / 60), ' minutes')
            
            # 保存模型
            trained_models_dir = './TrainedModels/'
            if not os.path.exists(trained_models_dir):
                os.makedirs(trained_models_dir)

            # 如果平均相对变化比历史最小值更优，则更新并保存模型
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

            # t2 = time.time()
            # print('Evaluating time: ', round((t2 - t1) / 60), ' minutes')
def test(args,dataloader_test,travel_location):

    if args.dataset=='TDrive':
        num_loc=759
        maxi=27
    elif args.dataset=='Shenzhen':
        num_loc=2236
        maxi=0
    elif args.dataset=='Tecent':
        num_loc=900
        maxi=0
    else:
        raise NotImplementedError(args.dataset)
    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    # 加载训练好的模型
    ar_model=ARModel(vocab_size=num_loc,  # (最大位置ID) + 1(mask token)+sos
                      travel_location=travel_location,
                      d_model=128,
                      nhead=8,
                      num_decoder_layers=4,
                      dim_feedforward=512,max_len=args.max_length).to(device)
    ar_model.load_state_dict(torch.load('./train_travel_pattern/checkpoints/'+args.dataset+'_travel_pattern_best_'+str(args.max_length)+'.pkl', map_location=device,weights_only=False)['model_state_dict'])

    ar_model.eval()

    trained_models_dir = './TrainedModels/'
    traj_diffusion = torch.load(os.path.join(trained_models_dir, 
    f'{args.model_name}_{args.dataset}_{args.max_length}.pkl'), map_location=device, weights_only=False)
    
    # 设置为评估模式
    traj_diffusion.eval()
    traj_diffusion.model_device = device
    traj_diffusion.model.model_device = device
    print(f"Using device: {device}")
    print(f"Model device: {next(traj_diffusion.parameters()).device}")
    print('开始测试......')
    t1 = time.time()
    
    with torch.no_grad(), autocast():
        trajs_generated = []
        trajs_real = []
        
        # 在测试集上生成轨迹
        for i, (x,home_locations) in tqdm(enumerate(dataloader_test), desc='测试中', total=len(dataloader_test)):
            x = x.to(device).long()
            home_locations = home_locations.to(device)
            output,ys = ar_model(condition=home_locations.to(torch.long),cluster=None,tgt=None,batch_size=x.shape[0])
            travel_pattern=ys.long()
            # travel_pattern=x[:,1].long()
            # 生成轨迹
            trajs = traj_diffusion.TrajGenerating(num_samples=x.shape[0], 
                                                home_location=home_locations,
                                                travel_pattern=travel_pattern)
            
            trajs_generated.append(trajs.long())
            trajs_real.append(x[:, 0, :].long())

            # if i==0:
            #     # 计算AR模型的复杂度
            #     macs_ar, params_ar = profile(ar_model, inputs=(home_locations.to(torch.long),None,None,x.shape[0]))
                
            #     # 计算Diffusion模型的复杂度 - 传入模型对象而不是函数
            #     macs_diffusion, params_diffusion = profile(traj_diffusion, inputs=(x.shape[0], home_locations, travel_pattern))
                
            #     print(f"AR Model MACs: {macs_ar / 1e9:.2f} G")
            #     print(f"Diffusion Model MACs: {macs_diffusion / 1e9:.2f} G")
            #     print(f"Total MACs: {(macs_ar+macs_diffusion) / 1e9:.2f} G")
            #     print(f"Total Params: {(params_ar+params_diffusion) / 1e6:.2f} M")
        
        # 合并所有生成的和真实的轨迹
        trajs_generated = torch.cat(trajs_generated, dim=0).detach().cpu().numpy()
        trajs_real = torch.cat(trajs_real, dim=0).detach().cpu().numpy()
        # 保存生成的轨迹和真实轨迹
        save_dir = './results/trajectories/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        np.save(os.path.join(save_dir, f'{args.dataset}_generated_{args.max_length}_hidden_128.npy'), trajs_generated)
        np.save(os.path.join(save_dir, f'{args.dataset}_real_{args.max_length}.npy'), trajs_real)
        print(f'轨迹数据已保存至 {save_dir}')
        # 计算评估指标
        d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, cpc,fow_mape,pop_mape = evaluate(args.dataset,args.max_length,trajs_real, trajs_generated)
        
        print('JSD:', d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd, 'CPC:', cpc, 'FOW_MAPE:', fow_mape, 'POP_MAPE:', pop_mape)
        
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
    # # 创建结果目录
    # results_dir = '/home/yxluo/research/TrajGDM/Results/'
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
        
    # # 保存结果到JSON文件
    # with open(os.path.join(results_dir, f'test_results_{args.model_name}_{args.dataset}.json'), 'w') as f:
    #     json.dump(results, f, indent=4)


        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--model_name', default='Traveller', type=str)
    parser.add_argument('--epoch', default=2, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--cuda', default="7", type=str)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--dataset', default='Shenzhen', type=str)  
    parser.add_argument('--batch_size', default=128, type=int) 
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_hidden', default=512, type=int)
    parser.add_argument('--TrajGenerator_Translayers', default=2, type=int)
    parser.add_argument('--TrajGenerator_heads', default=8, type=int)
    parser.add_argument('--TrajGenerator_LSTMlayers', default=1, type=int)
    parser.add_argument('--num_evaluation', default=512, type=int)
    parser.add_argument('--eval_epoch', default=1, type=int)
    parser.add_argument('--max_length', default=168, type=int)
    parser.add_argument('--repeat', default=1, type=int)
    # parser.add_argument('--home_condition', default=True,type=bool)
    # parser.add_argument('--distance_condition', default=True,type=bool)
    parser.add_argument('--home_condition', action='store_true', default=True)
    parser.add_argument('--distance_condition', action='store_true', default=True)
    args = parser.parse_args()
    # results=[]
    dataloader_train,dataloader_eval,dataloader_test,travel_location = load_data(args)
    print('train_len:',len(dataloader_train)*args.batch_size)
    print('eval_len:',len(dataloader_eval)*args.num_evaluation)
    print('test_len:',len(dataloader_test)*args.num_evaluation)
    trainer(args,dataloader_train,dataloader_eval,travel_location)

    result=test(args,dataloader_test,travel_location)
    print('d_jsd:',result['distance_jsd'])
    print('g_jsd:',result['geometry_jsd'])
    print('du_jsd:',result['duration_jsd'])
    print('p_jsd:',result['position_jsd'])
    print('l_jsd:',result['length_jsd'])
    print('f_jsd:',result['frequency_jsd'])
    print('cpc:',result['cpc'])

