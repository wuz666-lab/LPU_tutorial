import torch
import math

import torch.nn as nn
import torch.nn.functional as F

class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE
        self.max_seq_len = max_seq_len
        self.freqs = self._compute_freqs()
    
    def _compute_freqs(self):
        """计算RoPE频率"""
        theta = 10000.0
        dim = self.head_dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        return freqs

    def _apply_rope(self, x, freqs):
        """应用RoPE位置编码"""
        seq_len = x.size(-2)
        # x shape: (batch, heads, seq_len, head_dim)
        
        # 计算位置编码
        t = torch.arange(seq_len, device=x.device).float()
        freqs = freqs[:x.size(-1)//2]
        freqs = torch.outer(t, freqs)
        
        # 分离偶数和奇数维度
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        
        # 应用旋转
        cos_freqs = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
        sin_freqs = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        
        x_rope_even = x_even * cos_freqs - x_odd * sin_freqs
        x_rope_odd = x_even * sin_freqs + x_odd * cos_freqs
        
        # 重新组合
        x_rope = torch.zeros_like(x)
        x_rope[..., ::2] = x_rope_even
        x_rope[..., 1::2] = x_rope_odd
        
        return x_rope

    def _apply_rope_matmul(self, x, freqs):
        """使用矩阵乘法实现RoPE位置编码, 先生成一个旋转矩阵，然后"""
        seq_len = x.size(-2)
        dim = x.size(-1)
        device = x.device
        batch_size, n_heads = x.size(0), x.size(1)

        t = torch.arange(seq_len, device=device).float()
        freqs = freqs[:dim//2]
        freqs = torch.outer(t, freqs)

        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)

        # 构建RoPE旋转矩阵 (seq_len, dim, dim)
        rope_matrix = torch.zeros(seq_len, dim, dim, device=device)

        for i in range(dim//2):
            # 每对维度的2x2旋转矩阵
            rope_matrix[:, 2*i, 2*i] = cos_freqs[:, i]      # cos
            rope_matrix[:, 2*i, 2*i+1] = -sin_freqs[:, i]   # -sin
            rope_matrix[:, 2*i+1, 2*i] = sin_freqs[:, i]    # sin
            rope_matrix[:, 2*i+1, 2*i+1] = cos_freqs[:, i]  # cos

        # 应用矩阵乘法: (batch, heads, seq_len, dim) @ (seq_len, dim, dim)
        # 需要对每个位置分别应用对应的旋转矩阵
        x_rope = torch.matmul(x, rope_matrix)

        return x_rope

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # reshape到多头
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 应用RoPE
        q = self._apply_rope(q, self.freqs)
        k = self._apply_rope(k, self.freqs)
        
        # 计算attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用到values
        out = torch.matmul(attn_weights, v)
        
        # 重新组合heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out, attn_weights

    def compare_rope_methods(self, x):
        """比较两种RoPE实现的结果"""
        print("Comparing RoPE methods...")
        
        # 使用第一种方法
        result1 = self._apply_rope(x, self.freqs)
        
        # 使用第二种方法
        result2 = self._apply_rope_matmul(x, self.freqs)
        
        # 计算差异
        diff = torch.abs(result1 - result2)
        max_diff = torch.max(diff)
        mean_diff = torch.mean(diff)
        
        print(f"Max difference: {max_diff.item():.2e}")
        print(f"Mean difference: {mean_diff.item():.2e}")
        print(f"Results are close: {torch.allclose(result1, result2, atol=1e-6)}")
        
        return result1, result2, diff
# 测试函数
def test_rope_comparison():
    """测试RoPE方法比较"""
    d_model = 64
    n_heads = 8
    batch_size = 2
    seq_len = 10
    
    # 创建模型
    attention = RoPEAttention(d_model, n_heads)
    
    # 创建测试数据
    x = torch.randn(batch_size, n_heads, seq_len, d_model // n_heads)
    
    # 比较两种方法
    result1, result2, diff = attention.compare_rope_methods(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result1.shape}")
    
    return result1, result2, diff
# 运行测试
if __name__ == "__main__":
    test_rope_comparison()
