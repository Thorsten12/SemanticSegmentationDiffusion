"""PH2 boundary-point dataset (flat layout).

The PH2 data on disk is *flat*:

    <root>/trainx/IMD002.bmp            # dermoscopy image
    <root>/trainy/IMD002_lesion.bmp     # binary lesion mask

For each sample we return:
    image  : FloatTensor [3, H, W] in [-1, 1]  (conditioning, RGB only)
    points : FloatTensor [N, 2]    in [-1, 1]  (ground-truth boundary, ordered)
    mask   : FloatTensor [1, H, W] in {0, 1}   (clean GT mask, for evaluation)

Points are obtained from the mask via OpenCV contours + arc-length uniform
resampling, then rolled so the topmost point is index 0 (a canonical starting
phase for the ordered sequence).
"""

import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def uniform_sampling(contour: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to `n` points equally spaced by arc length."""
    contour = np.asarray(contour, dtype=np.float32)

    # Close the loop if it isn't already.
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    seg = np.diff(contour, axis=0)                 # segment vectors
    seg_len = np.linalg.norm(seg, axis=1)
    seg_len = np.maximum(seg_len, 1e-8)            # guard against zero-length segments

    s = np.concatenate([[0.0], np.cumsum(seg_len)])  # cumulative arc length
    t = np.linspace(0.0, s[-1], n, endpoint=False)   # target arc positions

    idx = np.searchsorted(s, t, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)

    local_t = (t - s[idx]) / seg_len[idx]
    return contour[idx] + seg[idx] * local_t[:, None]

def curvature_adaptive_sampling(contour: np.ndarray, n: int, alpha: float = 2.0) -> np.ndarray:
    """Resample a closed polygon to `n` points, clustering more points in high-curvature areas.
    
    Args:
        contour (np.ndarray): Input points [M, 2]
        n (int): Target number of points (e.g., 200)
        alpha (float): Controls adaptation strength. 
                       0.0 = completely uniform. 
                       Higher values = more points clustered in sharp bends/indents.
    """
    contour = np.asarray(contour, dtype=np.float32)
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    # 1. Compute segment vectors and their structural lengths
    seg = np.diff(contour, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    seg_len = np.maximum(seg_len, 1e-8)
    
    # Real physical arc length
    s_physical = np.concatenate([[0.0], np.cumsum(seg_len)])

    # 2. Compute curvature (angle changes between consecutive segments)
    # Normalize segments to compute dot products for angles
    seg_norm = seg / seg_len[:, None]
    # Roll to get adjacent segments
    seg_norm_next = np.roll(seg_norm, shift=-1, axis=0)
    
    # Cosine of angles between segment i and segment i+1
    cos_angles = np.sum(seg_norm * seg_norm_next, axis=1)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles) # Bending angle in radians at each vertex

    # 3. Define a density weight per segment
    # Weight = linear length + alpha * local bending intensity
    # We distribute the vertex angle across its adjacent segments
    vertex_curvature = angles
    seg_curvature = 0.5 * (vertex_curvature + np.roll(vertex_curvature, shift=1))
    
    # Dense sampling metric
    seg_weights = seg_len + alpha * seg_curvature
    s_density = np.concatenate([[0.0], np.cumsum(seg_weights)])

    # 4. Generate target positions in the warped density space
    t_density = np.linspace(0.0, s_density[-1], n, endpoint=False)

    # 5. Map warped density positions back to physical arc lengths via interpolation
    t_physical = np.interp(t_density, s_density, s_physical)

    # 6. Standard projection back to 2D coordinates (identical to original pipeline)
    idx = np.searchsorted(s_physical, t_physical, side="right") - 1
    idx = np.clip(idx, 0, len(seg) - 1)

    local_t = (t_physical - s_physical[idx]) / seg_len[idx]
    return contour[idx] + seg[idx] * local_t[:, None]

def _list_pairs(root: str) -> List[dict]:
    """Discover (image, mask) path pairs from the flat trainx/trainy layout."""
    img_dir = os.path.join(root, "trainx")
    mask_dir = os.path.join(root, "trainy")
    if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
        raise FileNotFoundError(
            f"Expected '{img_dir}' and '{mask_dir}'. Point --data_root at the "
            f"folder that contains trainx/ and trainy/."
        )

    pairs = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(".bmp"):
            continue
        stem = os.path.splitext(fname)[0]               # e.g. IMD002
        mask_path = os.path.join(mask_dir, f"{stem}_lesion.bmp")
        img_path = os.path.join(img_dir, fname)
        if os.path.exists(mask_path):
            pairs.append({"id": stem, "img": img_path, "mask": mask_path})
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {root}.")
    return pairs


def make_splits(root: str, n_val: int, n_test: int, seed: int):
    """Deterministic train/val/test split over the discovered pairs."""
    pairs = _list_pairs(root)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test


class PH2ContourDataset(Dataset):
    def __init__(
        self,
        samples: List[dict],
        n_points: int = 200,
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        aug_level: str = "strong",   # "none" | "light" | "strong"
        adaptive_sampling: bool = True
    ):
        self.samples = samples
        self.n_points = n_points
        self.img_size = img_size  # (H, W)
        self.augment = augment and aug_level != "none"
        self.aug_level = aug_level

        self.adaptive_sampling = adaptive_sampling

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    # --- helpers -------------------------------------------------------------

    def _augment(self, img: Image.Image, mask: Image.Image):
        """Synchronous geometric/colour augmentation; returns (img, mask).

        Geometric transforms (flips, affine) are applied identically to image and
        mask so the boundary stays aligned; photometric transforms (jitter, blur)
        touch the image only. Caller validates the result and may retry.
        """
        strong = self.aug_level == "strong"

        # Geometric (shared).
        if random.random() < 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() < 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)

        angle = random.uniform(-30, 30)
        max_tx = int(img.size[0] * 0.1)
        max_ty = int(img.size[1] * 0.1)
        translate = (random.randint(-max_tx, max_tx), random.randint(-max_ty, max_ty))
        scale = random.uniform(0.9, 1.2) if strong else random.uniform(0.95, 1.15)
        shear = random.uniform(-8, 8) if strong else 0
        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear)
        mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear)

        # Photometric (image only).
        if strong:
            img = transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                         saturation=0.2, hue=0.02)(img)
            if random.random() < 0.3:
                img = TF.gaussian_blur(img, kernel_size=5, sigma=random.uniform(0.1, 1.5))
        else:
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        return img, mask

    def _mask_touches_border(self, mask_binary: np.ndarray) -> bool:
        return bool(
            np.any(mask_binary[0, :]) or np.any(mask_binary[-1, :])
            or np.any(mask_binary[:, 0]) or np.any(mask_binary[:, -1])
            or mask_binary.sum() == 0
        )

    def _contour_points(self, mask_binary: np.ndarray) -> np.ndarray:
        """Largest external contour -> N points, rolled to top."""
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contour = max(contours, key=cv2.contourArea).squeeze()
        
        # --- NEU: Dynamische Weiche je nach Konfiguration ---
        if self.adaptive_sampling:
            points = curvature_adaptive_sampling(contour, self.n_points, alpha=2.0)
        else:
            points = uniform_sampling(contour, self.n_points)
            
        top_idx = int(np.argmin(points[:, 1]))      # topmost point -> canonical start
        return np.roll(points, shift=-top_idx, axis=0)

    # --- main ----------------------------------------------------------------

    def _load_raw(self, idx: int):
        """Return (PIL RGB image, PIL L mask) for index `idx`. Override in subclasses."""
        sample = self.samples[idx]
        img = Image.open(sample["img"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")
        return img, mask

    def __getitem__(self, idx: int):
        img, mask = self._load_raw(idx)

        img_used, mask_used = img, mask  # default: no augmentation
        if self.augment:
            # Keep the first augmentation that leaves the whole lesion inside the
            # frame. "light" tries once (else identity, the original recipe);
            # "strong" retries so almost every sample is augmented.
            attempts = 5 if self.aug_level == "strong" else 1
            for _ in range(attempts):
                aug_img, aug_mask = self._augment(img, mask)
                aug_mask_r = TF.resize(
                    aug_mask, self.img_size,
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
                if not self._mask_touches_border((np.array(aug_mask_r) > 0).astype(np.uint8)):
                    img_used, mask_used = aug_img, aug_mask_r
                    break

        # Resize mask (nearest) and build binary map.
        mask_r = TF.resize(
            mask_used, self.img_size,
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        mask_binary = (np.array(mask_r) > 0).astype(np.uint8) * 255

        # Ground-truth boundary points in [-1, 1] (x, y).
        points = self._contour_points(mask_binary).astype(np.float32)
        H, W = self.img_size
        points[:, 0] = points[:, 0] / (W - 1)
        points[:, 1] = points[:, 1] / (H - 1)
        points = points * 2.0 - 1.0

        img_tensor = self.img_transform(img_used)
        points_tensor = torch.from_numpy(points)
        mask_tensor = torch.from_numpy((mask_binary > 0).astype(np.float32)).unsqueeze(0)

        return img_tensor, points_tensor, mask_tensor


class ArrayContourDataset(PH2ContourDataset):
    """Contour dataset backed by in-memory arrays (preprocessed npy).

    images : uint8 array [N, 3, H, W] (channels-first, as stored in the npy files)
    masks  : uint8 array [N, 1, H, W] or [N, H, W]

    Reuses the full augmentation / contour-sampling / normalization pipeline of
    PH2ContourDataset; only the raw-sample loading differs. Using the shared npy
    arrays (in their stored order) lets us reproduce the published index splits
    exactly.
    """
    def __init__(self, images, masks, n_points=200, img_size=(224, 224),
                 augment=False, aug_level="strong", adaptive_sampling=True): 
        # Bypass PH2's file-discovery __init__; set up the base fields directly.
        Dataset.__init__(self)
        self.images = images
        self.masks = masks
        self.n_points = n_points
        self.img_size = img_size
        self.augment = augment and aug_level != "none"
        self.aug_level = aug_level
        self.adaptive_sampling = adaptive_sampling 
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def _load_raw(self, idx):
        img = np.moveaxis(np.asarray(self.images[idx], dtype=np.uint8), 0, -1)  # HWC
        m = np.asarray(self.masks[idx], dtype=np.uint8)
        if m.ndim == 3:                      # [1, H, W] -> [H, W]
            m = m.squeeze(0)
        return Image.fromarray(img).convert("RGB"), Image.fromarray(m).convert("L")
