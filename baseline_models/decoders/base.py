"""Common contract every pyramid decoder in this project follows.

Every decoder receives the SAME thing from the encoder -- a 4-scale feature
pyramid [s4, s8, s16, s32] (finest -> coarsest, strides 4/8/16/32 relative to
the input image) -- and must return per-pixel logits at the target output
size. This lets baseline_seg.py swap decoders via a single --decoder flag
without touching the encoder or the training loop at all.

To add a new decoder:
  1. Subclass PyramidDecoder.
  2. Implement __init__(self, enc_channels, decoder_dim=64, num_classes=1, **kwargs)
     and forward(self, feats) -> logits at feats[0]'s resolution (stride 4).
  3. Register it in DECODER_REGISTRY at the bottom of __init__.py.

IMPORTANT: forward() should return logits at STRIDE 4 (feats[0]'s spatial
size), NOT full image resolution. The final upsample to full image size is
done once, centrally, in BaselineSegmenter.forward() -- this keeps every
decoder implementation simple and guarantees identical upsample handling
(mode="bilinear", align_corners=False) across all of them for a fair
comparison. If a decoder naturally produces full-res logits already (like
the old BaselineUNetSegmenter did), that's also fine: BaselineSegmenter's
final resize is a no-op in that case (shape already matches out_size).
"""

import torch.nn as nn


class PyramidDecoder(nn.Module):
    """Abstract base class. enc_channels = [c4, c8, c16, c32] channel counts
    of the encoder's 4-scale pyramid, finest scale first."""

    def __init__(self, enc_channels, decoder_dim=64, num_classes=1):
        super().__init__()
        assert len(enc_channels) == 4, (
            f"PyramidDecoder expects a 4-scale pyramid (stride 4/8/16/32), "
            f"got {len(enc_channels)} scales ({enc_channels})."
        )
        self.enc_channels = enc_channels
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes

    def forward(self, feats):
        """feats: [s4, s8, s16, s32] tensors, finest first.
        Returns: logits [B, num_classes, H, W] at ANY resolution -- the
        caller (BaselineSegmenter) resizes to the target output size."""
        raise NotImplementedError