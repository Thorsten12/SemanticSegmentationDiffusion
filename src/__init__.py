"""P2SDiff: diffusion on contour points for binary semantic segmentation.

The backbone extracts a feature pyramid; a contour diffusion decoder reads
those features at N boundary points and denoises them into a polygon, which
is rasterized to a mask.
"""

__all__ = ["config"]
