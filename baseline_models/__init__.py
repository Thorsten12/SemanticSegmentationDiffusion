"""P2SDiff: boundary-point diffusion for semantic segmentation.

A clean, self-contained baseline that runs the diffusion process on the 2D
coordinates of object-boundary points (instead of per-pixel intensities), then
rasterizes the denoised points into a binary segmentation mask.

Pipeline
--------
1. Mask -> contour -> arc-length uniform sampling of N ordered points -> [-1, 1].
2. Forward process: add Gaussian noise to the *point coordinates* (DDPM, T=1000).
3. Reverse process: a point-Transformer denoiser predicts the clean points (x0),
   conditioned on image features sampled at the current point locations.
4. Rasterize predicted points (fill polygon) -> binary mask; score with Dice/IoU.
"""

__all__ = ["config"]
