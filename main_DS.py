import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import VLR
from DataLoader_Distributed import load_test_MR2,load_train_MR2
import CLIP_pipeline
import DeepSpeedTrainer

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()                  #创建命令行参数解析器
    parser = deepspeed.add_config_arguments(parser)     #向 parser 添加 DeepSpeed 支持的参数
    args = parser.parse_args()                          #解析命令行输入
    
    # DeepSpeed初始化分布式环境
    deepspeed.init_distributed()
    
    # 设置本地排名
    if 'LOCAL_RANK' in os.environ:                      #LOCAL_RANK 是 DeepSpeed 在分布式训练中自动传入的 GPU 编号
        local_rank = int(os.environ['LOCAL_RANK'])
        is_distributed = True
    else:
        local_rank = -1
        is_distributed = False
    
    # 设置常数
    learning_rate = 2e-5
    num_epochs = 40
    batch_size = 16
    
    # 创建模型
    model = VLR(dim=768)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)        #AdaW支持权重衰减
    
    # 使用支持分布式训练的数据加载器（多卡训练时，每张卡将加载不同子集数据）
    train_loader = load_train_MR2(batch_size, is_distributed, local_rank)
    test_loader = load_test_MR2(batch_size, is_distributed, local_rank)
    
    # DeepSpeed配置
    ds_config = {
        "train_batch_size": batch_size * torch.cuda.device_count() if is_distributed else batch_size,       #实际训练 batch size，等于 batch_size × GPU 数量（若分布式）
        #启用混合精度训练，节省显存，提高速度
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        #启用 ZeRO Stage 2 优化器，将模型参数分布到多个设备上，降低显存占用
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        #Warmup 学习率调度器，提高训练稳定性
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 100
            }
        },
        #梯度累积步数，用于 batch size 太小时累积梯度
        "gradient_accumulation_steps": 1,
        #限制梯度范数，防止 exploding gradients
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }
    
    # 初始化DeepSpeed引擎
    parameters = filter(lambda p: p.requires_grad, model.parameters())      #只传入需要训练的参数（有些 frozen 参数不更新）
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config=ds_config
    )
    
    # 创建修改后的Trainer
    trainer = DeepSpeedTrainer(
        model_engine=model_engine,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        local_rank=local_rank,
        is_distributed=is_distributed
    )
    
    # 训练模型
    trainer.fit(num_epochs)
    
    # 在主进程中保存最终模型
    if local_rank == 0 or local_rank == -1:
        print(f"Best model was at epoch {trainer.best_epoch+1} with accuracy {trainer.best_acc:.4f}")


if __name__ == "__main__":
    main()