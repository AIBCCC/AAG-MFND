import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import PIL
from transformers import AutoProcessor, AutoModelForVision2Seq
import tqdm

class ImageCaptionDataset(Dataset):
    ##初始化时接收JSON文件路径、图像目录和两种分析数据
    def __init__(self, json_file, img_dir):
        self.data = self.load_data(json_file)
        self.img_dir = img_dir
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

        return caption, image
    
#自定义数据批处理方法   
def custom_collate(batch):
    texts, images= zip(*batch)
    
    images = list(images)
    texts = list(texts)
    
    return  texts, images


# Usage example
def load_test_MR2(batch_size):
    all_dataset = ImageCaptionDataset('./data/MR2/en_test.json', './data/MR2')

    # Create the data loaders
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return test_loader

def load_train_MR2(batch_size):
    all_dataset = ImageCaptionDataset('./data/MR2/en_train.json', './data/MR2')

    # Create the data loaders
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    return test_loader

# 模型选择（可替换为其他支持图文生成的模型）
MODEL_NAME = "/llava-v1.5-7b"  # 也可用 THUDM/cogvlm-chat-hf
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和处理器
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32).to(DEVICE)
model.eval()

prompt1="[INST] <image> news content: <text> Analyze the given news image and text to determine why this is likely genuine news. Provide specific analyses to support your conclusion that this news item is authentic.[/INST]" 
prompt2="[INST] <image> news content: <text> Analyze the given news image and text to determine why this is likely fake news. Provide specific analyses to support your conclusion that this news item is not authentic.[/INST]"

def generate_analysis(image, text, prompt_template):
    """
    使用模型生成分析结果
    """
    try:
        # 替换提示模板中的文本占位符
        prompt = prompt_template.replace("<text>", text)
        
        # 处理输入
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        
        # 生成响应
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取生成的回答部分（移除原始提示）
        if "[/INST]" in generated_text:
            analysis = generated_text.split("[/INST]")[-1].strip()
        else:
            analysis = generated_text.strip()
            
        return analysis
        
    except Exception as e:
        print(f"生成分析时出错: {str(e)}")
        return f"分析生成失败: {str(e)}"

def analyze_dataset(data_loader, output_file, dataset_name="dataset"):
    """
    分析数据集并保存结果
    """
    results = {}
    
    print(f"开始分析{dataset_name}数据集...")
    
    for batch_idx, (keys, texts, images) in enumerate(tqdm(data_loader, desc=f"处理{dataset_name}")):
        for i, (key, text, image) in enumerate(zip(keys, texts, images)):
            print(f"正在处理项目 {key}...")
            
            # 生成正面论点（支持真实性）
            positive_analysis = generate_analysis(image, text, prompt1)
            
            # 生成反面论点（支持虚假性）
            negative_analysis = generate_analysis(image, text, prompt2)
            
            # 保存结果
            results[key] = {
                "原始文本": text,
                "正面论点_支持真实": positive_analysis,
                "反面论点_支持虚假": negative_analysis,
                "处理状态": "完成"
            }
            
            # 定期保存中间结果，防止程序中断丢失数据
            if (batch_idx * len(keys) + i + 1) % 10 == 0:
                save_results(results, output_file)
                print(f"已处理 {batch_idx * len(keys) + i + 1} 个样本，中间结果已保存")
    
    # 保存最终结果
    save_results(results, output_file)
    print(f"{dataset_name}分析完成，结果已保存到 {output_file}")
    
    return results

def save_results(results, output_file):
    """
    保存结果到JSON文件
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")

def main():
    """
    主函数
    """
    batch_size = 1  # 建议使用小批次以节省内存
    
    # 创建输出目录
    output_dir = "./analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 分析训练集
        print("加载训练数据集...")
        train_loader = load_train_MR2(batch_size)
        train_results = analyze_dataset(
            train_loader, 
            os.path.join(output_dir, "train_analysis_results.json"),
            "训练集"
        )
        
        # 分析测试集
        print("加载测试数据集...")
        test_loader = load_test_MR2(batch_size)
        test_results = analyze_dataset(
            test_loader, 
            os.path.join(output_dir, "test_analysis_results.json"),
            "测试集"
        )
        
        # 生成汇总报告
        summary = {
            "分析完成时间": str(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"),
            "训练集样本数": len(train_results),
            "测试集样本数": len(test_results),
            "使用模型": MODEL_NAME,
            "设备": DEVICE,
            "状态": "分析完成"
        }
        
        save_results(summary, os.path.join(output_dir, "analysis_summary.json"))
        
        print("="*50)
        print("所有分析任务完成！")
        print(f"训练集分析结果: {os.path.join(output_dir, 'train_analysis_results.json')}")
        print(f"测试集分析结果: {os.path.join(output_dir, 'test_analysis_results.json')}")
        print(f"汇总报告: {os.path.join(output_dir, 'analysis_summary.json')}")
        print("="*50)
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        print("请检查数据路径和模型配置")

if __name__ == "__main__":
    main()