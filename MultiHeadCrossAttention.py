import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class multimodal_attention(nn.Module):
    """
    多模态点积注意力机制(Dot-product Attention)
    适用于图像与文本之间的跨模态交互特征提取
    """
    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        # 定义 Dropout 用于注意力分数的随机屏蔽，防止过拟合
        self.dropout = nn.Dropout(attention_dropout)
        # Softmax 激活函数，用于归一化注意力权重
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播过程：
        参数:
            q: 查询张量 (batch, len_q, d_k)
            k: 键张量 (batch, len_k, d_k)
            v: 值张量 (batch, len_k, d_v)
            scale: 缩放因子（通常为 1 / sqrt(d_k))
            attn_mask: 可选的 attention mask(如用于屏蔽填充位置)
        返回:
            v_result: 注意力加权后的输出结果 (batch, len_q, d_v)
        """
        # 计算原始注意力权重矩阵：Q x K^T → (batch, len_q, len_k)
        attention = torch.matmul(q, k.transpose(-2, -1))
        # print('attention.shape:{}'.format(attention.shape))

        # 如果有缩放因子（scale），就进行缩放（常用于避免梯度过小或过大）
        if scale:
            attention = attention * scale

        # 如果提供了 attention mask，则对指定位置屏蔽（如用 -inf 填充）
        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)

        # 对 attention 权重做 Softmax 归一化，得到注意力分布
        attention = self.softmax(attention)
        # print('attention.shftmax:{}'.format(attention))

        # 应用 Dropout（用于训练阶段的正则化）
        attention = self.dropout(attention)

        # 将注意力权重乘以 V，得到注意力加权结果
        v_result = torch.matmul(attention, v)
        # print('attn_final.shape:{}'.format(attention.shape))

        # 返回最终融合后的输出
        return v_result
    
class CrossAttention(nn.Module):
    """
    多头交叉注意力机制(Multi-Head Cross Attention)
    通常用于处理两种模态（如图像和文本）之间的交互
    """
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(CrossAttention, self).__init__()

        self.model_dim = model_dim                    # 输入总维度（一般是 transformer 模型维度）
        self.dim_per_head = model_dim // num_heads   # 每个注意力头的维度
        self.num_heads = num_heads

        # 为每个模态输入分别定义 Q/K/V 的线性变换
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        # 点积注意力模块（封装在 multimodal_attention 中）
        self.dot_product_attention = multimodal_attention(dropout)

        # 多头输出后的线性变换
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)

        # Dropout 层，用于正则化防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 残差连接 + LayerNorm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        前向传播
        参数:
            query: 查询向量（来自一种模态，比如文本） shape: (batch, len_q, dim)
            key: 键向量（来自另一种模态，比如图像） shape: (batch, len_k, dim)
            value: 值向量（与 key 同源） shape: (batch, len_k, dim)
            attn_mask: 掩码（可选，用于屏蔽填充部分）
        返回:
            output: 融合后的输出向量,shape: (batch, len_q, dim)
        """
        residual = query  # 保留残差连接的原始输入

        # Q, K, V 做线性映射（多头拼接）
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # 将 Q/K/V 分头处理，转换维度 → (batch, num_heads, seq_len, dim_per_head)
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

         # 缩放因子（避免大维度下注意力数值不稳定）
        scale = self.dim_per_head ** -0.5

        # 计算缩放点积注意力，注意：query 来自一模态，key/value 来自另一模态，实现“交叉”
        attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # 合并多个头（多头拼接）
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        # 输出通过线性层
        output = self.linear_final(attention)

        # Dropout 进行正则化
        output = self.dropout(output)

        # 残差连接 + 层归一化（LayerNorm）
        output = self.layer_norm(residual + output)

        return output
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadCrossAttention, self).__init__()

        self.model_dim = model_dim                         # 模型总维度（默认768）
        self.dim_per_head = model_dim // num_heads         # 每个注意力头的维度
        self.num_heads = num_heads                         # 注意力头数量

        # 内部封装的 CrossAttention 模块，双向都会复用同一个模块
        self.cross_attention = CrossAttention(model_dim, num_heads, dropout)

        # 层归一化，用于稳定训练并支持残差连接
        self.layer_norm = nn.LayerNorm(model_dim)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, attn_mask=None):
        """
        x1: 第一个模态的表示（如文本），形状 (batch, len_1, dim)
        x2: 第二个模态的表示（如图像），形状 (batch, len_2, dim)
        attn_mask: 可选注意力 mask(通常可为 None)

        返回：
            output_1: x1 对 x2 的交叉注意力输出 + 残差
            output_2: x2 对 x1 的交叉注意力输出 + 残差
        """

        # 从 x1 到 x2 做交叉注意力（即：x1 是 Query，x2 是 Key/Value）
        cross_attn_output_1 = self.cross_attention(x1, x2, x2, attn_mask)

        # 从 x2 到 x1 做交叉注意力（即：x2 是 Query，x1 是 Key/Value）
        cross_attn_output_2 = self.cross_attention(x2, x1, x1, attn_mask)

        # 残差连接 + 层归一化
        output_1 = self.layer_norm(x1 + cross_attn_output_1)
        output_2 = self.layer_norm(x2 + cross_attn_output_2)

        return output_1, output_2


# # Example usage
# batch_1, len_1, dim = 2, 10, 768
# batch_2, len_2, dim = 2, 15, 768

# x1 = torch.randn(batch_1, len_1, dim)
# x2 = torch.randn(batch_2, len_2, dim)

# layer = MultiHeadCrossAttention(model_dim=768, num_heads=8, dropout=0.5)
# output_1, output_2 = layer(x1, x2)


# print("output_1 shape:", output_1.size())  # Expected: [batch_1, len_1, 768]
# print("output_2 shape:", output_2.size())  # Expected: [batch_2, len_2, 768]