from torch import nn
from SelfAttention.self_attention import SelfAttention
from SelfAttention.MLP import MLP

class PointTransformerBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 num_heads=4, 
                 mlp_ratio=2,
                 proj_p=0,
                 attn_p=0,
                 mlp_p=0):
        
        super().__init__()
        
        # 1. Self-Attention (Punkte reden mit Punkten, um die Form zu glätten)
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.attn = SelfAttention(in_channels=in_channels,
                                  num_heads=num_heads, 
                                  attn_p=attn_p,
                                  proj_p=proj_p)
        
        
        # 2. MLP (Punkte verarbeiten ihre Position und die Bild-Features isoliert)
        self.norm2 = nn.LayerNorm(in_channels, eps=1e-6)
        self.mlp = MLP(in_channels=in_channels,
                       mlp_ratio=mlp_ratio,
                       mlp_p=mlp_p)
        
    def forward(self, x):
        # x hat hier schon die Form (Batch, n_punkte, in_channels)
        # Und ganz wichtig: 'x' enthält hier bereits die eingesaugten Bild-Features!
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x