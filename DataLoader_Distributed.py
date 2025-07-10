import json
import os
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

def read_jsonl_field_to_list(file_path, field_name):
    results = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if field_name in data:
                        results.append(data[field_name].strip())
                    else:
                        results.append("")
                except json.JSONDecodeError:
                    print(f"Error: Could not parse line in {file_path}")
    except FileNotFoundError:
        print(f"Error: {file_path} does not exist.")
    except IOError:
        print(f"Error: Unable to read {file_path}.")
    
    return results

# 全局变量，可以在主脚本中进行初始化
analy_t_en_train = read_jsonl_field_to_list("./AAR/Adversarial_arguments/train_adversarial_arguments.jsonl", "true_news_view")
analy_f_en_train = read_jsonl_field_to_list("./AAR/Adversarial_arguments/train_adversarial_arguments.jsonl", "false_news_view")
analy_t_en_test = read_jsonl_field_to_list("./AAR/Adversarial_arguments/test_adversarial_arguments.jsonl", "true_news_view")
analy_f_en_test = read_jsonl_field_to_list("./AAR/Adversarial_arguments/test_adversarial_arguments.jsonl", "false_news_view")

class ImageCaptionDataset(Dataset):
    def __init__(self, json_file, img_dir, analy_t, analy_f):
        self.data = self.load_data(json_file)
        self.img_dir = img_dir
        self.analy_1 = analy_t
        self.analy_2 = analy_f
        
    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        key = list(self.data.keys())[idx]
        item = self.data[key]
        
        caption = item['caption']
        img_path = os.path.join(self.img_dir, item['image_path'])
        image = Image.open(img_path).convert('RGB')
        label = item['label']
        
        return caption, image, self.analy_1[idx], self.analy_2[idx], torch.tensor(label)

def custom_collate(batch):
    texts, images, data1, data2, labels = zip(*batch)
    
    images = list(images)
    texts = list(texts)
    data1 = list(data1)
    data2 = list(data2)
    labels = torch.stack(labels, dim=0)
    
    return texts, images, data1, data2, labels

# 修改后的数据加载函数，支持分布式训练
def load_test_MR2(batch_size, is_distributed=False, local_rank=-1):
    all_dataset = ImageCaptionDataset('./data/MR2/en_test.json', './data/MR2', analy_t_en_test, analy_f_en_test)
    
    if is_distributed:
        # 为分布式训练创建采样器
        sampler = DistributedSampler(
            all_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=local_rank,
            shuffle=False  # 测试集通常不需要打乱顺序
        )
        test_loader = DataLoader(
            all_dataset,
            batch_size=batch_size,
            shuffle=False,  # 使用DistributedSampler时必须设为False
            sampler=sampler,
            collate_fn=custom_collate,
            pin_memory=True,
            num_workers=4
        )
    else:
        test_loader = DataLoader(
            all_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate
        )
    
    return test_loader

def load_train_MR2(batch_size, is_distributed=False, local_rank=-1):
    all_dataset = ImageCaptionDataset('./data/MR2/en_train.json', './data/MR2', analy_t_en_train, analy_f_en_train)
    
    if is_distributed:
        # 为分布式训练创建采样器
        sampler = DistributedSampler(
            all_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=local_rank,
            shuffle=True  # 训练集需要打乱顺序
        )
        train_loader = DataLoader(
            all_dataset,
            batch_size=batch_size,
            shuffle=False,  # 使用DistributedSampler时必须设为False
            sampler=sampler,
            collate_fn=custom_collate,
            pin_memory=True,
            num_workers=4
        )
    else:
        train_loader = DataLoader(
            all_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate
        )
    
    return train_loader