import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" #使用HF的镜像网站下载模型

# 设置内存分配器参数，在导入torch之前设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

from PIL import Image #图像加载
import requests  
from transformers import CLIPProcessor, CLIPModel  #模型加载和文本图像预处理
import torch

devices = "cuda:0"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(devices)  #"openai/clip-vit-base-patch32"相对轻量，参数量较少
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

