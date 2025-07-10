#位置前馈网络FFN
#这个模块通常放在注意力层之后，用于增强每个 token 的表达能力。为了增强模型表达能力、引入非线性变换，同时在每个 token 位置上做更深层次的特征抽象。
#升维 → 激活 → 再降维，有助于学到稀疏有效的特征表达
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalWiseFeedForward(nn.Module):
    """
    位置前馈网络(Position-wise Feedforward Network)
    是 Transformer 中每个子层的组成部分之一。
    每个 token 位置上单独应用一个两层的全连接网络。
    """

    def __init__(self, model_dim=768, ffn_dim=2048, dropout=0.5):
        super(PositionalWiseFeedForward, self).__init__()

        # 第一层线性映射（升维）：从模型维度 → FFN 高维空间
        self.w1 = nn.Linear(model_dim, ffn_dim)

        # 第二层线性映射（降维）：从 FFN → 模型维度
        self.w2 = nn.Linear(ffn_dim, model_dim)

        # Dropout 用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 层归一化，增强训练稳定性
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x  # 残差连接，保存原始输入

        # 前馈网络两层：ReLU 激活后再线性映射回原始维度
        x = self.w2(F.relu(self.w1(x)))

        x = self.dropout(x)  # Dropout 正则化
        x += residual        # 残差连接：输出 = 输入 + FFN(x)
        x = self.layer_norm(x)  # LayerNorm 稳定训练
        output = x
        return output

