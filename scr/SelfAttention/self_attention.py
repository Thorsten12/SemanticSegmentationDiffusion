from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=12, attn_p=0, proj_p=0):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(in_channels, in_channels)        # Diese Daten werden nach und nach angepasst, starten random und werden dann durch das Training immer besser. 
        self.value = nn.Linear(in_channels, in_channels)        # Es ist wichtig, dass die Dimensionen der Query, Key und Value gleich sind, damit wir die Attention berechnen können.
        self.key = nn.Linear(in_channels, in_channels)          

        self.attn_p = attn_p
        self.proj_p = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p)
        
        x = x.transpose(1,2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj_drop(self.proj_p(x))
        return x
