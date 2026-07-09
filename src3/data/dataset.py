from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2

import numpy as np
from PIL import Image

from .sampling import uniform_sampling, canonicalize_contour

def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Extract the largest external contour from a binary mask as (x, y) points."""
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("No contour found in mask.")

    largest = max(contours, key=cv2.contourArea)
    return largest.squeeze(1).astype(np.float32)  # (N, 2), Reihenfolge (x, y)


class ArrayContourDataset(Dataset):
    def __init__(self, images, masks, n_points: int = 200, img_size=(224, 224), augment=True, aug_level="strong"):
        super().__init__()
        self.images = images
        self.masks = masks
        self.n_points = n_points
        self.img_size = img_size
        self.augment = augment
        self.aug_level = aug_level

        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def _load_raw(self, idx):
        img = np.moveaxis(np.asarray(self.images[idx], dtype=np.uint8), 0, -1)
        m = np.asarray(self.masks[idx], dtype=np.uint8)
        if m.ndim == 3:
            m = m.squeeze(0)
        return Image.fromarray(img).convert("RGB"), Image.fromarray(m).convert("L")

    def __getitem__(self, idx: int):
        img, mask = self._load_raw(idx)

        img_tensor = self.img_transform(img)

        mask_resized = TF.resize(mask, self.img_size, interpolation=transforms.InterpolationMode.NEAREST)
        mask_np = np.array(mask_resized)

        contour = mask_to_contour(mask_np)
        points = uniform_sampling(contour, self.n_points)   # (n_points, 2)
        canonicalize_contour(points)

        # Normalisierung auf [-1, 1], damit die Diffusion mit x_0 in einem sinnvollen Wertebereich arbeitet
        points_norm = points.copy()
        points_norm[:, 0] = (points[:, 0] / self.img_size[1]) * 2 - 1
        points_norm[:, 1] = (points[:, 1] / self.img_size[0]) * 2 - 1

        points_tensor = torch.from_numpy(points_norm).float()  # (n_points, 2)

        mask_tensor = torch.from_numpy(mask_np).long()
        return img_tensor, points_tensor, mask_tensor
    