import torch
import torch.nn as nn
from AAR_clip_model import clip_model,devices

class VisualProjection(nn.Module):
    #投影层，封装线性变换（CLIP的viasual/text_projection)
    def __init__(self,visual_projection):
        super().__init__()
        self.visual_projection=visual_projection

    def forward(self,x):
        # 将输入张量 x 从原始维度 (batch, len, 768) 映射到 目标维度 (batch, len, 768)
        x=self.visual_projection(x)
        return x

class TextProjection(nn.Module):
    def __init__(self, text_projection):
        super().__init__()
        self.text_projection = text_projection

    def forward(self, x):
        """
        将输入张量 x 从 (batch, len, 768) 映射到 (batch, len, 768)
        """
        x = self.text_projection(x)
        return x
    
visual_projection=clip_model.visual_projection
text_projection=clip_model.text_projection

# 创建 VisualProjection 模块并进行测试

Visual_module = VisualProjection(visual_projection).to(devices)
#input_tensor_1 = outputs["vision_model_output"]["last_hidden_state"]
#Visual_output_tensor = Visual_module(input_tensor_1)

Text_module = TextProjection(text_projection).to(devices)
#input_tensor_2 = outputs["text_model_output"]["last_hidden_state"]
#Text_output_tensor = Text_module(input_tensor_2)

#print(Text_output_tensor.size())
#print(Visual_output_tensor.size())   