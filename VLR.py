import torch.nn as nn
import torch
from MultiHeadCrossAttention import MultiHeadCrossAttention
import torch.nn.functional as F
from MultiHeadSelfAttention import MultiHeadSelfAttention

#两层全连接神经网络
class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=256, dropout=0.5):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, out_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)       # [batch_size, ..., in_features] → [batch_size, ..., hidden_size]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)         # [batch_size, ..., hidden_size] → [batch_size, ..., out_features]
        return x
    
class VLR(nn.Module):
    def __init__(self, dim=768):        #dim为文本和图像的特征维度
        super(VLR, self).__init__()
        # 对文本和图像分别做自注意力，提取上下文相关的 token 表达
        self.text_self_attention = MultiHeadSelfAttention(model_dim=dim, num_heads=8, dropout=0.5)
        self.image_self_attention = MultiHeadSelfAttention(model_dim=dim, num_heads=8, dropout=0.5)

        # 可训练的融合权重（初始都是 0.5），学习图文对最终判断的相对贡献
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        # 第一次判断模块（融合表示）利用图文融合特征 G 做第一次真假判断（通常作为“初步判断”）
        self.first_judge = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim//2, 2)
        )
        
        # 对抗推理阶段使用的交叉注意力模块
        self.adversarial_cross_attention = MultiHeadCrossAttention(
            model_dim=dim, num_heads=8, dropout=0.5
        )

        # 第二次判断模块（融合 adversarial 推理后的输出）模型在推理整合后再进行一次判断，这一阶段具备“推理解释性”。
        self.second_judge = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dim//2, 2)
        )

    def forward(self, text_features, image_features, adversarial_arguments_1, adversarial_arguments_2):
        # 自注意力建模文本/图像的上下文，然后平均池化成一个向量（句级或图级语义）
        R1 = self.text_self_attention(text_features).mean(dim=1)
        R2 = self.image_self_attention(image_features).mean(dim=1)
        
        # 图文融合：使用可学习权重对两个模态进行线性组合
        G = self.alpha * R1 + self.beta * R2

        # 第一次判断（融合判断）
        z1 = self.first_judge(G)

        # 将正反论点平均池化为句级表示，再加和构建对抗表示（正反都考虑）
        adversarial_arguments = adversarial_arguments_1.mean(dim=1) + adversarial_arguments_2.mean(dim=1)

        # 使用 cross-attention 让“对抗论点”去关注图文融合表示 G
        Lg, _ = self.adversarial_cross_attention(adversarial_arguments, G)  #Lg 表示融合了对抗观点后的新语义表示（有“反思”“辩论”的能力）
        Lg = Lg.mean(dim=1)  # 再次池化为整体表示

        # 第二次判断（带有对抗推理的结果）
        z2 = self.second_judge(Lg)

        # 返回的是两个 softmax 概率（也可直接输出 z1/z2 logits）
        z1_prob = F.softmax(z1, dim=-1)
        z2_prob = F.softmax(z2, dim=-1)

        return z1, z2

# batch_size, len_1, len_2, dim = 4, 10, 15, 768

# x1 = torch.randn(batch_size, len_1, dim)
# x2 = torch.randn(batch_size, len_2, dim)
# x3 = torch.randn(batch_size, 20, dim)
# x4 = torch.randn(batch_size, 25, dim)

# model = VLR()
# output,output_1 = model(x1, x2,x3,x4)

# print("output_1 shape:", output_1.size()) 

# print(output_1)