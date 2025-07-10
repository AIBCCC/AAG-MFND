import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads                      # 注意力头的数量
        self.dim_per_head = model_dim // num_heads     # 每个注意力头的维度（注意要能整除）

        # 定义三个线性层分别用于生成 Query、Key、Value
        #虽然维度不变(用于注意力层的残差连接)，但线性变换后的数据分布已不同,通过独立的线性变换，将输入数据投影到不同的语义空间（Q/K/V），使模型能够动态学习如何关注（Q/K）和关注什么（V）
        #即用于Q=x*W_Q, K=x*W_K, V=x*W_V分别更新权重矩阵W,（权重形状为 [model_dim, model_dim]）
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)

        # 注意力输出的 Dropout
        self.dropout = nn.Dropout(dropout)

        # 注意力得分归一化（沿最后一个维度）
        self.softmax = nn.Softmax(dim=-1)

        # 所有多头合并后，做一次线性映射回原模型维度
        self.linear_out = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 将 Q/K/V 映射到多头形式： (batch, seq_len, num_heads, dim_per_head) → transpose → (batch, num_heads, seq_len, dim_per_head)
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # 计算缩放点积注意力分数：Q x K^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim_per_head ** 0.5)

        # 应用注意力掩码（mask 为 0 的位置填充极小值，使得 softmax 后趋近于0）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 归一化注意力分数
        attn = self.softmax(scores)

        # Dropout 防止过拟合
        attn = self.dropout(attn)

        # 加权 Value 得到上下文表示：注意力权重 * 值向量
        context = torch.matmul(attn, value)

        # 合并多头： (batch, num_heads, seq_len, dim) → (batch, seq_len, num_heads * dim_per_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        # 最后一层线性变换，映射回模型原始维度
        output = self.linear_out(context)

        return output

#双向协同注意力机制（Co-Attention）
class CoAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(CoAttention, self).__init__()

        # 使用两个多头注意力模块，分别进行 x1 对 x2 的注意力 和 x2 对 x1 的注意力
        self.attention1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention2 = MultiHeadAttention(model_dim, num_heads, dropout)

        # 用于融合两个方向的输出（拼接后降维）
        self.linear_out = nn.Linear(2 * model_dim, model_dim)

        # 层归一化，稳定训练
        self.layer_norm = nn.LayerNorm(model_dim)

        # Dropout 正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, mask1=None, mask2=None):
        """
        x1, x2: 两种输入模态（可以是文本和图像，或者两段不同的文本）
        mask1, mask2: 掩码（用于屏蔽无效 token,防止注意力泄露）
        返回: 融合后的特征表示,shape = (batch_size, model_dim)
        """

        # x1 对 x2 做多头注意力：x1 作为 Query，x2 作为 Key 和 Value
        attn_output1 = self.attention1(x1, x2, x2, mask2)

        # x2 对 x1 做多头注意力：x2 作为 Query，x1 作为 Key 和 Value
        attn_output2 = self.attention2(x2, x1, x1, mask1)

        # 对两个注意力输出在序列维度（token 维）上求均值池化（取全局语义表示）
        pooled1 = attn_output1.mean(dim=1)   # shape: (batch, model_dim)
        pooled2 = attn_output2.mean(dim=1)   # shape: (batch, model_dim)

        # 拼接两个方向的输出 → (batch, 2 * model_dim)
        combined = torch.cat([pooled1, pooled2], dim=-1)

        # 融合后的输出通过线性变换降回原始维度
        output = self.dropout(self.linear_out(combined))

        # 层归一化增强训练稳定性
        output = self.layer_norm(output)

        return output

# # Example usage
# batch_size, len_1, len_2, dim = 2, 10, 15, 768

# x1 = torch.randn(batch_size, len_1, dim)
# x2 = torch.randn(batch_size, len_2, dim)

# model = CoAttention(model_dim=dim, num_heads=8, dropout=0.5)
# output = model(x1, x2)

# print("output shape:", output.size())  # Expected: [batch, 768]
