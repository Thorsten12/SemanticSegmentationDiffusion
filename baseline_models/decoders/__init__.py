"""Decoder registry -- single source of truth mapping --decoder names to
classes. To add a new decoder: implement it as a PyramidDecoder subclass in
its own file (see base.py for the contract), import it below, and add one
line to DECODER_REGISTRY.
"""

from .base import PyramidDecoder
from .unet import UNetDecoder
from .linear_mlp import LinearMLPDecoder
from .semantic_fpn import SemanticFPNDecoder
from .upernet import UPerNetDecoder

from .cascade import CascadeDecoder
from .g_cascade import GCascadeDecoder
from .emcad import EmcadDecoder

from .bat import BATDecoder
from .deepsnake import DeepSnakeDecoder
from .xboundformer import XBoundFormerDecoder
from .boundformer import BoundaryFormerDecoder


DECODER_REGISTRY = {
    # Group A
    "linear_mlp": LinearMLPDecoder,       # Group A #1 -- SegFormer linear/all-MLP floor
    "semantic_fpn": SemanticFPNDecoder,   # Group A #2 -- Panoptic FPN semantic head (CVPR 2019)
    "unet": UNetDecoder,                  # Group A #3 -- U-Net (MICCAI 2015)
    "upernet": UPerNetDecoder,            # Group A #4 -- UPerNet (ECCV 2018)

    # Group B
    "cascade": CascadeDecoder,            # Group B #5 -- CASCADE (WACV 2023)
    "g_cascade": GCascadeDecoder,         # Group B #6 -- G-CASCADE (WACV 2024)
    "emcad": EmcadDecoder,                # Group B #7 -- EMCAD (CVPR 2024)

    # Group C
    "bat": BATDecoder,                     # Group C #8 -- Boundary-Aware Transformer(MICCAI 2021)  
    "deepsnake": DeepSnakeDecoder,         # Group C #9 -- DeepSnake (CVPR 2020)
    "xboundformer": XBoundFormerDecoder,    # Group C #10 -- XBoundFormer (CVPR 2023)
    "boundformer": BoundaryFormerDecoder   # Group C #11 -- BoundaryFormer (CVPR 2022)

    # Group D

}

# Backward-compatible aliases so old --model values keep working unchanged.
DECODER_ALIASES = {
    "unet": "unet",
    "simple_head": "shallow_conv_head",
}


def build_decoder(name, enc_channels, decoder_dim=64, num_classes=1):
    resolved = DECODER_ALIASES.get(name, name)
    if resolved not in DECODER_REGISTRY:
        raise ValueError(
            f"Unknown decoder '{name}' (resolved to '{resolved}'). "
            f"Available: {sorted(DECODER_REGISTRY.keys())} "
            f"(aliases: {sorted(DECODER_ALIASES.keys())})"
        )
    return DECODER_REGISTRY[resolved](enc_channels, decoder_dim=decoder_dim, num_classes=num_classes)
