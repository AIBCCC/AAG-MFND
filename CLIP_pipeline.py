from AAR_clip_model import clip_model,devices,processor
from feature_alignment import Text_module,Visual_module
import requests
from PIL import Image
import torch
from torch.cuda.amp import autocast, GradScaler

# x1：主文本的文本表示
# x2：主图像的视觉表示
# x3：正向推理文本（支持为真）
# x4：反向推理文本（支持为假）

scaler=GradScaler()

def CLIP_pipeline(x1, x2, x3, x4):
    # 使用autocast上下文管理器
    with autocast():

        # 对 x1 进行文本编码（主文本），使用 CLIP 的 tokenizer，最大长度限制为 77（CLIP 默认）
        tmp_inputs_text = processor(text=x1, return_tensors="pt", padding=True, truncation=True, max_length=77).to(devices)

        # 对 x2 进行图像预处理并转 tensor（主图像）
        tmp_inputs_image = processor(images=x2, return_tensors="pt").to(devices)

        # 对 x3（正向推理文本）和 x4（反向推理文本）进行文本预处理
        tmp_inputs_analy1 = processor(text=x3, return_tensors="pt", padding=True, truncation=True, max_length=77).to(devices)
        tmp_inputs_analy2 = processor(text=x4, return_tensors="pt", padding=True, truncation=True, max_length=77).to(devices)

        # 提取主文本的 token-level 表示（last_hidden_state），不做 projection（还原 raw features）
        outputs_1 = clip_model.text_model(**tmp_inputs_text)

        # 提取图像的 token 表示（一般是 patch tokens）
        outputs_2 = clip_model.vision_model(**tmp_inputs_image)

        # 提取文本最后一层隐表示
        outputs_tensor_1 = outputs_1["last_hidden_state"]
        Text_output_tensor = Text_module(outputs_tensor_1)  # 送入 CLIP 提供的 text_projection 层,这两层是 CLIP 模型中专门用于将文本和图像的隐藏表示“映射到同一个共享语义空间”的线性变换层。

        outputs_tensor_2 = outputs_2["last_hidden_state"]
        Visual_output_tensor = Visual_module(outputs_tensor_2)  # 同理，送入 vision projection 层

        # 分别处理两个对抗性分析文本（正/反观点），获取其深层表示
        outputs_3 = clip_model.text_model(**tmp_inputs_analy1)["last_hidden_state"]
        outputs_4 = clip_model.text_model(**tmp_inputs_analy2)["last_hidden_state"]

        outputs_3 = Text_module(outputs_3)  # 映射到与主文本相同的特征空间
        outputs_4 = Text_module(outputs_4)

    # 返回四个张量，分别是：主文本特征、主图像特征、正向推理特征、反向推理特征
    return Text_output_tensor, Visual_output_tensor, outputs_3, outputs_4

# x1 = ["a photo of a cat", "a photo of a dog"]
# x2 = [Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw),
#       Image.open(requests.get("http://images.cocodataset.org/val2017/000000397133.jpg", stream=True).raw)]

# x1,x2 = CLIP_pipeline(x1,x2)

# print(x1.size())
# print(x2.size())