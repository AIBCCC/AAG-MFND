import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import PIL

#读取jsonl文件
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
                        results.append("")  # 如果字段不存在，添加空字符串
                except json.JSONDecodeError:
                    print(f"Error: Could not parse line in {file_path}")
    except FileNotFoundError:
        print(f"Error: {file_path} does not exist.")
    except IOError:
        print(f"Error: Unable to read {file_path}.")
    
    return results


analy_t_en_train = read_jsonl_field_to_list("./AAR/Adversarial_arguments/train_adversarial_arguments.jsonl", "true_news_view")
analy_f_en_train = read_jsonl_field_to_list("./AAR/Adversarial_arguments/train_adversarial_arguments.jsonl", "false_news_view")
analy_t_en_test = read_jsonl_field_to_list("./AAR/Adversarial_arguments/test_adversarial_arguments.jsonl", "true_news_view")
analy_f_en_test = read_jsonl_field_to_list("./AAR/Adversarial_arguments/test_adversarial_arguments.jsonl", "false_news_view")

class ImageCaptionDataset(Dataset):
    ##初始化时接收JSON文件路径、图像目录和两种分析数据
    def __init__(self, json_file, img_dir, analy_t, analy_f):

        self.data = self.load_data(json_file)
        self.img_dir = img_dir
        self.analy_1 = analy_t
        self.analy_2 = analy_f

    #加载JSON数据
    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    #返回数据集长度
    def __len__(self):
        return len(self.data)
    
    #根据索引返回数据项，包括标题、图像、两种分析数据和标签
    def __getitem__(self, idx):

        key = list(self.data.keys())[idx]
        item = self.data[key]
        
        caption = item['caption']
        img_path = os.path.join(self.img_dir, item['image_path'])
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, PIL.UnidentifiedImageError):
            print(f"警告：无法打开图像 {img_path}，使用替代图像")

        label = item['label']
        
        return caption, image, self.analy_1[idx], self.analy_2[idx], torch.tensor(label)
    
#自定义数据批处理方法   
def custom_collate(batch):
    texts, images, data1, data2, labels = zip(*batch)
    

    images = list(images)
    texts = list(texts)
    data1 = list(data1)
    data2 = list(data2)

    labels = torch.stack(labels, dim=0)
    
    return  texts, images, data1, data2, labels


# Usage example
def load_test_MR2(batch_size):
    all_dataset = ImageCaptionDataset('./data/MR2/en_test.json', './data/MR2', analy_t_en_test, analy_f_en_test)

    # Create the data loaders
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return test_loader

def load_train_MR2(batch_size):
    all_dataset = ImageCaptionDataset('./data/MR2/en_train.json', './data/MR2',analy_t_en_train,analy_f_en_train)

    # Create the data loaders
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return test_loader

'''
if __name__ == "__main__":
    dataload = load_test_MR2(1)
    for x1,x2,x3,x4,x5 in dataload:
        print(x3)
        break
'''