import random

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


def _mask_touches_border(mask_binary: np.ndarray) -> bool:
    """True wenn die Läsion den Bildrand berührt oder die Maske leer ist.
    Beides macht den extrahierten Kontur unbrauchbar (abgeschnitten/nicht vorhanden)."""
    return bool(
        np.any(mask_binary[0, :]) or np.any(mask_binary[-1, :])
        or np.any(mask_binary[:, 0]) or np.any(mask_binary[:, -1])
        or mask_binary.sum() == 0
    )


class ArrayContourDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        n_points: int = 200,
        img_size=(224, 224),
        augment: bool = True,
        aug_level: str = "strong",
        epoch_multiplier: int = 1,
        max_aug_retries: int = 8,
    ):
        """
        epoch_multiplier: künstlich vervielfachte Länge des Datasets, z.B. 4 für ~4x
            mehr effektive Samples pro Epoche. Jede "virtuelle" Instanz eines
            Originalbilds bekommt eine unabhängig gezogene Augmentierung, da
            Python's random-Modul bei jedem __getitem__-Call neu gezogen wird.
        max_aug_retries: wie oft eine gescheiterte Augmentierung (Maske leer /
            berührt Rand) neu versucht wird, bevor auf das unaugmentierte
            Original zurückgefallen wird.
        """
        super().__init__()
        self.images = images
        self.masks = masks
        self.n_points = n_points
        self.img_size = img_size
        self.augment = augment
        self.aug_level = aug_level
        self.epoch_multiplier = max(1, epoch_multiplier)
        self.max_aug_retries = max_aug_retries

        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.images) * self.epoch_multiplier

    def _load_raw(self, real_idx):
        img = np.moveaxis(np.asarray(self.images[real_idx], dtype=np.uint8), 0, -1)
        m = np.asarray(self.masks[real_idx], dtype=np.uint8)
        if m.ndim == 3:
            m = m.squeeze(0)
        return Image.fromarray(img).convert("RGB"), Image.fromarray(m).convert("L")

    def _augment(self, img: Image.Image, mask: Image.Image):
        is_strong = self.aug_level in ["strong", "elastic"]
        is_elastic = self.aug_level == "elastic"

        if random.random() < 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() < 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)

        # Elastische Deformation nur bei explizitem Level "elastic"
        if is_elastic and random.random() < 0.15:
            seed = random.randint(0, 2**32 - 1)
            img_t = TF.to_tensor(img)
            mask_t = TF.to_tensor(mask)
            torch.manual_seed(seed)
            img_t = transforms.ElasticTransform(alpha=100.0, sigma=10.0)(img_t)
            torch.manual_seed(seed)
            mask_t = transforms.ElasticTransform(
                alpha=100.0, sigma=10.0,
                interpolation=TF.InterpolationMode.NEAREST)(mask_t)
            img = TF.to_pil_image(img_t)
            mask = TF.to_pil_image(mask_t)

        # Basis-Transformationen für "strong" und "elastic" identisch
        angle = random.uniform(-180, 180) if is_strong else random.uniform(-30, 30)
        max_tx = int(img.size[0] * 0.1)
        max_ty = int(img.size[1] * 0.1)
        translate = (random.randint(-max_tx, max_tx), random.randint(-max_ty, max_ty))
        scale = random.uniform(0.75, 1.05) if is_strong else random.uniform(0.9, 1.05)
        shear = random.uniform(-8, 8) if is_strong else 0

        # Reflect-Padding vor der affinen Transform, damit Translation/Rotation/Scale
        # keine schwarzen Ränder erzeugt, sondern der Bildinhalt gespiegelt weiterläuft.
        # Padding-Breite muss groß genug für die maximale Verschiebung + Rotation/Scale sein.
        orig_w, orig_h = img.size
        # reflect-Padding darf laut PIL/torchvision nicht >= Bildgröße sein
        pad_w = min(max_tx + int(orig_w * 0.15), orig_w - 1)
        pad_h = min(max_ty + int(orig_h * 0.15), orig_h - 1)

        img = TF.pad(img, [pad_w, pad_h], padding_mode="reflect")
        mask = TF.pad(mask, [pad_w, pad_h], padding_mode="reflect")

        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear,
                         interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear,
                          interpolation=TF.InterpolationMode.NEAREST)

        # Zurück auf Originalgröße zuschneiden (zentriert), Padding wieder entfernen
        img = TF.center_crop(img, [orig_h, orig_w])
        mask = TF.center_crop(mask, [orig_h, orig_w])

        if is_strong:
            img = transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.2, hue=0.01)(img)
            if random.random() < 0.3:
                img = TF.gaussian_blur(img, kernel_size=5, sigma=random.uniform(0.1, 1.5))
        else:
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

        return img, mask

    def _augment_safe(self, img: Image.Image, mask: Image.Image):
        """Versucht mehrfach zu augmentieren, bis die Maske gültig ist
        (nicht leer, berührt nicht den Rand). Fällt sonst auf das Original zurück."""
        orig_img, orig_mask = img, mask
        for _ in range(self.max_aug_retries):
            aug_img, aug_mask = self._augment(orig_img, orig_mask)
            mask_check = TF.resize(aug_mask, self.img_size,
                                    interpolation=transforms.InterpolationMode.NEAREST)
            mask_bin = (np.array(mask_check) > 0).astype(np.uint8)
            if not _mask_touches_border(mask_bin):
                return aug_img, aug_mask
        return orig_img, orig_mask

    def __getitem__(self, idx: int):
        real_idx = idx % len(self.images)
        img, mask = self._load_raw(real_idx)

        if self.augment:
            img, mask = self._augment_safe(img, mask)

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