import random
import os
import PIL
import PIL.ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL.Image as Image
import torchvision.transforms.functional as TF
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter


# ============================================================
#  HILFSFUNKTIONEN
# ============================================================

def uniform_sampling(contour, n):
    contour = np.asarray(contour, dtype=np.float32)

    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    seg     = np.diff(contour, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s       = np.concatenate([[0], np.cumsum(seg_len)])
    t       = np.linspace(0, s[-1], n, endpoint=False)

    idx     = np.searchsorted(s, t, side='right') - 1
    idx     = np.clip(idx, 0, len(seg) - 1)

    local_t = (t - s[idx]) / seg_len[idx]
    return contour[idx] + seg[idx] * local_t[:, None]


def elastic_deform(image_np, mask_np, alpha=40, sigma=6):
    """
    image_np : (H, W, 3) uint8
    mask_np  : (H, W)    uint8
    """
    shape = mask_np.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha

    x, y     = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords_x = np.clip(x + dx, 0, shape[1] - 1)
    coords_y = np.clip(y + dy, 0, shape[0] - 1)
    indices  = [coords_y.ravel(), coords_x.ravel()]

    mask_out = map_coordinates(mask_np, indices, order=0).reshape(shape)
    img_out  = np.stack([
        map_coordinates(image_np[:, :, c], indices, order=1).reshape(shape)
        for c in range(3)
    ], axis=-1).astype(np.uint8)

    return img_out, mask_out


# ============================================================
#  DATASET
# ============================================================

class ISIC2017ContourDataset(Dataset):
    def __init__(self, images_dir, masks_dir, n_punkte=200, img_size=(256, 256)):
        """
        images_dir : Pfad zu "ISIC-2017_Training_Data"
        masks_dir  : Pfad zu "ISIC-2017_Training_Part1_GroundTruth"
        """
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.n_punkte   = n_punkte
        self.img_size   = img_size  # (H, W)

        # ----------------------------------------------------------
        #  1. Dateipaare sammeln
        # ----------------------------------------------------------
        self.samples = []
        for img_name in os.listdir(images_dir):
            if img_name.endswith('.jpg'):
                base_name = img_name.replace('.jpg', '')
                mask_name = f"{base_name}_segmentation.png"
                img_path  = os.path.join(images_dir, img_name)
                mask_path = os.path.join(masks_dir, mask_name)
                if os.path.exists(mask_path):
                    self.samples.append({'img_path': img_path, 'mask_path': mask_path})

        print(f"ISIC Datensatz: {len(self.samples)} Bilder gefunden.")

        # ----------------------------------------------------------
        #  2. Alles einmalig in RAM laden (uint8, bereits auf img_size resized)
        #     ISIC 2017 (~2000 Bilder, 256×256): ≈ 500 MB
        # ----------------------------------------------------------
        H, W = img_size
        print("Lade Bilder in RAM-Cache …")
        self.cache = {}
        for i, s in enumerate(self.samples):
            img  = Image.open(s['img_path']).convert("RGB")
            mask = Image.open(s['mask_path']).convert("L")
            img  = img.resize((W, H), Image.BILINEAR)
            mask = mask.resize((W, H), Image.NEAREST)
            self.cache[i] = {
                'img':  np.array(img,  dtype=np.uint8),   # (H, W, 3)
                'mask': np.array(mask, dtype=np.uint8),   # (H, W)
            }
        print("Cache fertig.")

        # ----------------------------------------------------------
        #  3. Original-Konturen einmalig vorberechnen (für den Fallback)
        # ----------------------------------------------------------
        print("Berechne Original-Konturen …")
        self.original_data = {}
        skipped = 0
        for i in range(len(self.samples)):
            mask_np  = self.cache[i]['mask']
            mask_bin = (mask_np > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                skipped += 1
                self.original_data[i] = None
                continue
            contour = max(contours, key=cv2.contourArea).squeeze()
            if contour.ndim < 2 or len(contour) < 3:
                skipped += 1
                self.original_data[i] = None
                continue
            pts = uniform_sampling(contour, n_punkte)
            top_idx = np.argmin(pts[:, 1])
            pts = np.roll(pts, -top_idx, axis=0)
            self.original_data[i] = {
                'points':   pts,          # (n_punkte, 2) – Pixelkoordinaten
                'mask_bin': mask_bin,     # (H, W) uint8
            }
        print(f"Konturen fertig. {skipped} Samples übersprungen (leere Maske).")

        # ----------------------------------------------------------
        #  4. Transforms & Augmentation-Objekte EINMALIG anlegen
        # ----------------------------------------------------------
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # ColorJitter einmal instanziieren, nicht pro Sample
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05
        )

    # ----------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Original-Daten aus Cache holen (kein Disk-I/O mehr)
        orig = self.original_data[idx]
        if orig is None:
            # Defektes Sample → nächstes nehmen
            return self.__getitem__((idx + 1) % len(self.samples))

        img  = Image.fromarray(self.cache[idx]['img'])
        mask = Image.fromarray(self.cache[idx]['mask'])

        # ===========================================================
        #  SYNCHRONE AUGMENTATION
        # ===========================================================

        # 1. Flips
        if random.random() < 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # 2. Affine (Rotation + Translation + Zoom)
        angle       = random.uniform(-30, 30)
        H, W        = self.img_size
        max_tx      = int(W * 0.1)
        max_ty      = int(H * 0.1)
        translate   = (random.randint(-max_tx, max_tx), random.randint(-max_ty, max_ty))
        scale       = random.uniform(0.95, 1.15)
        img  = TF.affine(img,  angle=angle, translate=translate, scale=scale, shear=0)
        mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=0)

        # 3. Farb-Jitter (wiederverwendete Instanz)
        img = self.color_jitter(img)

        # 4. Gaussian Blur
        if random.random() < 0.5:
            sigma = random.uniform(0.3, 1.2)
            img   = img.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))

        # 5. Gaussian Noise
        if random.random() < 0.4:
            img_np = np.array(img).astype(np.float32)
            img_np += np.random.randn(*img_np.shape) * random.uniform(3, 12)
            img     = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

        # 6. Elastic Deformation – REDUZIERT (20 % statt 50 %, schwächere Parameter)
        if random.random() < 0.2:
            img_np_el  = np.array(img)
            mask_np_el = np.array(mask.convert("L"))
            img_np_el, mask_np_el = elastic_deform(img_np_el, mask_np_el, alpha=40, sigma=6)
            img  = Image.fromarray(img_np_el)
            mask = Image.fromarray(mask_np_el)

        # 7. Random Crop + Resize
        if random.random() < 0.4:
            W_o, H_o  = img.size
            crop_frac = random.uniform(0.75, 0.95)
            cw = int(W_o * crop_frac)
            ch = int(H_o * crop_frac)
            x0 = random.randint(0, W_o - cw)
            y0 = random.randint(0, H_o - ch)
            img  = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((W_o, H_o), Image.BILINEAR)
            mask = mask.crop((x0, y0, x0 + cw, y0 + ch)).resize((W_o, H_o), Image.NEAREST)

        # 8. Cutout
        if random.random() < 0.3:
            img_np_co            = np.array(img)
            H_co, W_co           = img_np_co.shape[:2]
            cut_h                = random.randint(20, int(H_co * 0.25))
            cut_w                = random.randint(20, int(W_co * 0.25))
            cx                   = random.randint(0, W_co - cut_w)
            cy                   = random.randint(0, H_co - cut_h)
            img_np_co[cy:cy+cut_h, cx:cx+cut_w] = 128
            img                  = Image.fromarray(img_np_co)

        # ===========================================================
        #  KONTUR AUS AUGMENTIERTER MASKE EXTRAHIEREN
        # ===========================================================
        mask_np     = np.array(mask)
        mask_binary = (mask_np > 0).astype(np.uint8) * 255

        touches_edge = (
            np.any(mask_binary[0,  :] > 0) or
            np.any(mask_binary[-1, :] > 0) or
            np.any(mask_binary[:,  0] > 0) or
            np.any(mask_binary[:, -1] > 0) or
            np.sum(mask_binary) == 0
        )

        if touches_edge:
            # Fallback: Original-Daten aus Cache verwenden (kein Disk-I/O!)
            img         = Image.fromarray(self.cache[idx]['img'])
            mask_binary = orig['mask_bin'].copy()
            points      = orig['points'].copy()
        else:
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour     = max(contours, key=cv2.contourArea).squeeze()
            points      = uniform_sampling(contour, self.n_punkte)
            top_idx     = np.argmin(points[:, 1])
            points      = np.roll(points, -top_idx, axis=0)

        img_tensor = self.img_transform(img)

        # ===========================================================
        #  KOORDINATEN NORMALISIEREN [-1, 1]
        # ===========================================================
        H_img, W_img   = self.img_size
        points_norm    = points.astype(np.float32)
        points_norm[:, 0] = points_norm[:, 0] / (W_img - 1) * 2.0 - 1.0
        points_norm[:, 1] = points_norm[:, 1] / (H_img - 1) * 2.0 - 1.0
        points_tensor  = torch.tensor(points_norm, dtype=torch.float32)

        # ===========================================================
        #  MASKE DEGRADIEREN (für U-Net Conditioning)
        # ===========================================================
        degraded_mask = mask_binary.copy()
        if random.random() < 0.8:
            k_size    = random.randint(5, 15)
            kernel    = np.ones((k_size, k_size), np.uint8)
            op_choice = random.choice(['dilate', 'erode', 'close', 'open'])
            iters     = random.randint(1, 2)
            if op_choice == 'dilate':
                degraded_mask = cv2.dilate(degraded_mask, kernel, iterations=iters)
            elif op_choice == 'erode':
                degraded_mask = cv2.erode(degraded_mask, kernel, iterations=iters)
            elif op_choice == 'close':
                degraded_mask = cv2.morphologyEx(degraded_mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
            elif op_choice == 'open':
                degraded_mask = cv2.morphologyEx(degraded_mask, cv2.MORPH_OPEN,  kernel, iterations=iters)

        mask_tensor = torch.tensor(degraded_mask / 255.0, dtype=torch.float32).unsqueeze(0)

        return img_tensor, points_tensor, mask_tensor

    # ----------------------------------------------------------

    @staticmethod
    def verify_dataset(batch_images, batch_points, batch_masks, num_samples=4):
        batch_size, channels, H, W = batch_images.shape
        num_samples = min(batch_size, num_samples)
        fig, axes   = plt.subplots(2, num_samples, figsize=(16, 10))

        if num_samples == 1:
            axes = axes[:, np.newaxis]

        for i in range(num_samples):
            img  = batch_images[i].cpu().numpy().transpose(1, 2, 0)
            img  = np.clip((img * 0.5) + 0.5, 0, 1)
            mask = batch_masks[i].cpu().numpy().squeeze()
            pts  = batch_points[i].cpu().numpy()

            pts_x = (pts[:, 0] + 1.0) / 2.0 * (W - 1)
            pts_y = (pts[:, 1] + 1.0) / 2.0 * (H - 1)

            axes[0, i].imshow(img)
            axes[0, i].plot(pts_x, pts_y, 'r.-', markersize=3, linewidth=1)
            axes[0, i].plot([pts_x[-1], pts_x[0]], [pts_y[-1], pts_y[0]], 'r-', linewidth=1)
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Sample {i+1} (RGB)")

            axes[1, i].imshow(mask, cmap='gray')
            axes[1, i].plot(pts_x, pts_y, 'g.-', markersize=3, linewidth=1)
            axes[1, i].plot([pts_x[-1], pts_x[0]], [pts_y[-1], pts_y[0]], 'g-', linewidth=1)
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Sample {i+1} (Maske)")

        plt.tight_layout()
        plt.show()