import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute scaled dot product attention for given query, key, and value tensors."""
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == float('-inf'), float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x: Tensor) -> Tensor:
        """Reshape input tensor to separate heads and head dimensions."""
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute multi-head attention for given query, key, and value tensors."""
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.d_model)

        return self.fc(attn_output)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply position-wise feed-forward to input tensor."""
        return self.fc2(F.relu(self.fc1(x)))

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply a transformer layer to input tensor.

        It consists of a multi-head self-attention layer followed by a position-wise feed-forward layer.
        """
        # Multi-head self-attention
        attn_output = self.mha(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_output))

        # Position-wise feed-forward
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))

        return x