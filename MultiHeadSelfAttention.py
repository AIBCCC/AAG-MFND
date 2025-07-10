import torch.nn as nn
import torch
from MultiHeadCrossAttention import multimodal_attention

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    - 输入是一个模态的特征序列(如文本或图像patch)
    - 输出是同维度的、经过注意力重加权后的序列
    """
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadSelfAttention, self).__init__()

        self.model_dim = model_dim                    # 总的输入输出维度
        self.dim_per_head = model_dim // num_heads   # 每个注意力头的维度
        self.num_heads = num_heads
        
        # Q / K / V 的线性映射：把原始特征映射为多个注意力子空间
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        # 点积注意力模块（复用之前定义的）
        self.dot_product_attention = multimodal_attention(dropout)

        # 把所有注意力头的输出重新拼接后做线性变换
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)  # 残差+归一化
    
    def forward(self, x, attn_mask=None):
        residual = x  # 保留输入用于残差连接

        # Step 1：对输入 x 线性变换，得到 Q/K/V 向量
        query = self.linear_q(x)
        key = self.linear_k(x)
        value = self.linear_v(x)

        # Step 2：将每个向量 reshape 成多头结构 → (batch, num_heads, seq_len, dim_per_head)
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Step 3：缩放点积注意力计算（每个头独立完成）
        scale = (self.dim_per_head) ** -0.5
        attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # Step 4：将所有注意力头的输出拼接起来（恢复原始维度）
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        # Step 5：线性投影回原始维度 + Dropout
        output = self.linear_final(attention)
        output = self.dropout(output)

        # Step 6：残差连接 + 层归一化
        output = self.layer_norm(residual + output)

        return output

# # Example usage
# batch_size = 2
# seq_len = 10
# model_dim = 768

# x = torch.randn(batch_size, seq_len, model_dim)

# self_attention = MultiHeadSelfAttention(model_dim=model_dim, num_heads=8, dropout=0.5)
# output = self_attention(x)

# print("output shape:", output.size())  # Expected: [batch_size, seq_len, model_dim]