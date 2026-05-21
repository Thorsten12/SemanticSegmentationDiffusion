from torch import nn
from SelfAttention.self_attention import SelfAttention
from SelfAttention.MLP import MLP


class ImageTransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4, mlp_ratio=2, proj_p=0, attn_p=0, mlp_p=0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.attn = SelfAttention(in_channels=in_channels,
                                  num_heads=num_heads, 
                                  attn_p=attn_p,
                                  proj_p=proj_p)
        
        self.norm2 = nn.LayerNorm(in_channels, eps=1e-6)
        self.mlp = MLP(in_channels=in_channels,
                       mlp_ratio=mlp_ratio,
                       mlp_p=mlp_p)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
      
        ### 1. Übersetzer: Bild in flache Sequenz umwandeln -> (Batch, H*W, Channels)
        x = x.reshape(batch_size, channels, height*width).permute(0,2,1)
        
        ### 2. Attention anwenden
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        ### 3. Übersetzer: Sequenz wieder zu einem Bild zusammenfalten -> (Batch, Channels, H, W)
        x = x.permute(0,2,1).reshape(batch_size, channels, height, width)
        return x