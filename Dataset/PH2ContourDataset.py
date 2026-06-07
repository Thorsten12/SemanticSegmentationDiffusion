# This Dataset is being used for finetuning

import random
import os
import PIL
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import PIL.Image as Image   
import torchvision.transforms.functional as TF
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def uniform_sampling(contour, n):
    contour = np.asarray(contour, dtype=np.float32)

    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])

    # Segmentvektoren + Längen
    seg = np.diff(contour, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)

    # kumulative Bogenlänge
    s = np.concatenate([[0], np.cumsum(seg_len)])

    # Zielpositionen
    t = np.linspace(0, s[-1], n, endpoint=False)

    # Segmentindex
    idx = np.searchsorted(s, t, side='right') - 1
    idx = np.clip(idx, 0, len(seg)-1)

    # lokale Interpolation
    local_t = (t - s[idx]) / seg_len[idx]

    return contour[idx] + seg[idx] * local_t[:, None]

# Ganz oben in der Datei ergänzen
from scipy.ndimage import map_coordinates, gaussian_filter

def elastic_deform(image_np, mask_np, alpha=80, sigma=10):
    """
    Beide Arrays müssen numpy sein: image_np (H,W,3), mask_np (H,W)
    """
    shape = mask_np.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords_x = np.clip(x + dx, 0, shape[1]-1)
    coords_y = np.clip(y + dy, 0, shape[0]-1)
    indices = [coords_y.ravel(), coords_x.ravel()]

    # Maske mit nearest-neighbor (kein Blending der Binärwerte!)
    mask_out = map_coordinates(mask_np, indices, order=0).reshape(shape)

    # Bild mit bilinearer Interpolation
    img_out = np.stack([
        map_coordinates(image_np[:,:,c], indices, order=1).reshape(shape)
        for c in range(3)
    ], axis=-1).astype(np.uint8)

    return img_out, mask_out

class PH2ContourDataset(Dataset):
    def __init__(self, base_path, n_punkte=200, img_size=(256, 256)):
        """
        base_path: Der Pfad zum Ordner "PH2 Dataset images" 
                   (z.B. "C:/Users/.../PH2Dataset/PH2 Dataset images")
        """
        self.base_path = base_path
        self.n_punkte = n_punkte
        self.img_size = img_size
        self.samples = []

        # 1. Die exakte Ordnerstruktur des PH2-Datasets auslesen
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            
            # Überprüfen, ob es ein Ordner ist (z.B. "IMD002")
            if os.path.isdir(folder_path):
                # Die Pfade exakt nach der PH2-Namenskonvention zusammenbauen
                img_path = os.path.join(folder_path, f"{folder_name}_Dermoscopic_Image", f"{folder_name}.bmp")
                mask_path = os.path.join(folder_path, f"{folder_name}_lesion", f"{folder_name}_lesion.bmp")
                
                # Nur hinzufügen, wenn beide Dateien auch wirklich existieren
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.samples.append({
                        'img_path': img_path,
                        'mask_path': mask_path
                    })

        print(f"Datensatz geladen: {len(self.samples)} Bilder und Masken gefunden.")

        # 2. PyTorch Transformationen für das Konditionierungs-Bild
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- A. BILD VERARBEITEN ---
        img = Image.open(sample['img_path']).convert("RGB")
        mask = Image.open(sample['mask_path']).convert("L") 

        # --- SYNCHRONE AUGMENTATION ---
        
        # 1. Horizontaler Flip (Links/Rechts)
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            
        # 2. Vertikaler Flip (Oben/Unten)
        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # 3. Rotation & Zoom (Sicherere Werte!)
        angle = random.uniform(-30, 30) # Vorher -45 bis 45 (oft zu extrem für die Ecken)
        
        # Wir verschieben nur max 10% des Bildes (vorher feste Pixel, was bei kleinen Bildern tödlich ist)
        max_trans_x = int(img.size[0] * 0.1)
        max_trans_y = int(img.size[1] * 0.1)
        translate = (random.randint(-max_trans_x, max_trans_x), random.randint(-max_trans_y, max_trans_y))
        
        # Wir zoomen tendenziell etwas REIN (Scale > 1.0), damit Ränder aus dem Weg gehen.
        # Rauszoomen (Scale < 1.0) bringt oft schwarze Balken an den Rand, auf denen OpenCV hängen bleibt.
        scale = random.uniform(0.95, 1.15) 

        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=0)
        mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=0)

        # 4. Farb-Jitter 
        # ---- NEU: Hue + Saturation Jitter ----
        img = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.3,   # NEU — wichtig für Hauttöne
            hue=0.05          # NEU — kleine Farbverschiebung
        )(img)

        # ---- NEU: Gaussian Blur (50% Chance) ----
        if random.random() < 0.5:
            sigma = random.uniform(0.3, 1.2)
            img = img.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))

        # ---- NEU: Gaussian Noise (40% Chance) ----
        if random.random() < 0.4:
            img_np_noise = np.array(img).astype(np.float32)
            noise = np.random.randn(*img_np_noise.shape) * random.uniform(3, 12)
            img = Image.fromarray(np.clip(img_np_noise + noise, 0, 255).astype(np.uint8))

        # ---- NEU: Elastic Deformation (50% Chance) ----
        # WICHTIG: Vor dem Boundary-Check einfügen — der Guard fängt Fehler ab
        if random.random() < 0.5:
            img_np_el = np.array(img)
            mask_np_el = np.array(mask.convert("L"))
            img_np_el, mask_np_el = elastic_deform(img_np_el, mask_np_el, alpha=60, sigma=8)
            img = Image.fromarray(img_np_el)
            mask = Image.fromarray(mask_np_el)

        # ---- NEU: Random Crop + Resize (40% Chance) ----
        if random.random() < 0.4:
            W_orig, H_orig = img.size
            crop_frac = random.uniform(0.75, 0.95)
            cw = int(W_orig * crop_frac)
            ch = int(H_orig * crop_frac)
            x0 = random.randint(0, W_orig - cw)
            y0 = random.randint(0, H_orig - ch)
            img  = img.crop((x0, y0, x0+cw, y0+ch))
            mask = mask.crop((x0, y0, x0+cw, y0+ch))
            img  = img.resize((W_orig, H_orig), Image.BILINEAR)
            mask = mask.resize((W_orig, H_orig), Image.NEAREST)

        # ---- NEU: Cutout (30% Chance) ----
        # Blockiert einen zufälligen Bereich im BILD (nicht in der Maske)
        if random.random() < 0.3:
            img_np_co = np.array(img)
            H_co, W_co = img_np_co.shape[:2]
            cut_h = random.randint(20, int(H_co * 0.25))
            cut_w = random.randint(20, int(W_co * 0.25))
            cx = random.randint(0, W_co - cut_w)
            cy = random.randint(0, H_co - cut_h)
            img_np_co[cy:cy+cut_h, cx:cx+cut_w] = 128  # Grau statt Schwarz → weniger Artefakte
            img = Image.fromarray(img_np_co)

    

        # --- RESIZE FIX ---
        mask = TF.resize(mask, self.img_size, interpolation=transforms.InterpolationMode.NEAREST)
        img_tensor = self.img_transform(img) 

        mask_np = np.array(mask)
        mask_binary = (mask_np > 0).astype(np.uint8) * 255
        
        # NEU: Wir prüfen den äußersten Pixel-Rand (oben, unten, links, rechts)
        touches_top = np.any(mask_binary[0, :] > 0)
        touches_bottom = np.any(mask_binary[-1, :] > 0)
        touches_left = np.any(mask_binary[:, 0] > 0)
        touches_right = np.any(mask_binary[:, -1] > 0)
        
        # Wenn die Maske den Rand berührt ODER komplett verschwunden ist:
        if touches_top or touches_bottom or touches_left or touches_right or np.sum(mask_binary) == 0:
            # Fallback: Die Augmentation war zu aggressiv! 
            # Wir laden das Originalbild sicherheitshalber OHNE Augmentation neu.
            img = Image.open(sample['img_path']).convert("RGB")
            mask = Image.open(sample['mask_path']).convert("L")
            mask = TF.resize(mask, self.img_size, interpolation=transforms.InterpolationMode.NEAREST)
            img_tensor = self.img_transform(img)
            mask_np = np.array(mask)
            mask_binary = (mask_np > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        contour = max(contours, key=cv2.contourArea).squeeze()
        points = uniform_sampling(contour, self.n_punkte) 

        oberster_punkt_idx = np.argmin(points[:, 1])

        points = np.roll(points, shift=-oberster_punkt_idx, axis=0)

        # --- C. KOORDINATEN NORMALISIEREN [-1, 1] ---
        # ACHTUNG: PyTorch img_size ist (Höhe, Breite). 
        # OpenCV Punkte sind aber (X=Breite, Y=Höhe).
        # --- C. KOORDINATEN NORMALISIEREN [-1, 1] ---
        H, W = self.img_size 
        
        points_norm = points.astype(np.float32)
        points_norm[:, 0] = points_norm[:, 0] / (W - 1)
        points_norm[:, 1] = points_norm[:, 1] / (H - 1)
        points_norm = points_norm * 2.0 - 1.0

        points_tensor = torch.tensor(points_norm, dtype=torch.float32)

        # --- D. MASKE STRUKTURELL VERRAUSCHEN ---
        # WICHTIG: Die Punkte wurden bereits aus der perfekten mask_binary berechnet!
        # Jetzt degradieren wir die Maske für das U-Net.
        
        degraded_mask = mask_binary.copy()
        
        # In ca. 80 % der Fälle die Maske strukturell verschlechtern
        if random.random() < 0.8:
            # Zufällige Kernel-Größe (z.B. 5x5 bis 15x15 Pixel) für unterschiedlich starke Verzerrung
            k_size = random.randint(5, 15)
            kernel = np.ones((k_size, k_size), np.uint8)
            
            # Zufällige morphologische Operation wählen
            op_choice = random.choice(['dilate', 'erode', 'close', 'open'])
            
            if op_choice == 'dilate':
                # Lässt die Maske wachsen (Kante liegt zu weit außen)
                degraded_mask = cv2.dilate(degraded_mask, kernel, iterations=random.randint(1, 2))
            elif op_choice == 'erode':
                # Lässt die Maske schrumpfen (Kante liegt zu weit innen)
                degraded_mask = cv2.erode(degraded_mask, kernel, iterations=random.randint(1, 2))
            elif op_choice == 'close':
                # Schließt Löcher, macht den Rand oft klobiger und ungenauer
                degraded_mask = cv2.morphologyEx(degraded_mask, cv2.MORPH_CLOSE, kernel, iterations=random.randint(1, 2))
            elif op_choice == 'open':
                # Reißt kleine Strukturen weg, rundet ab
                degraded_mask = cv2.morphologyEx(degraded_mask, cv2.MORPH_OPEN, kernel, iterations=random.randint(1, 2))

        # Den Tensor jetzt aus der VERRAUSCHTEN Maske erstellen!
        mask_tensor = torch.tensor(degraded_mask / 255.0, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        
        return img_tensor, points_tensor, mask_tensor
    
    

    @staticmethod
    def verify_dataset(batch_images, batch_points, batch_masks, num_samples=4):
        """
        Plottet Bilder, Masken und Punkte aus dem PyTorch DataLoader zur Überprüfung.
        batch_images: Tensor der Form (B, 3, H, W) im Bereich [-1, 1]
        batch_points: Tensor der Form (B, n_punkte, 2) im Bereich [-1, 1]
        batch_masks: Tensor der Form (B, 1, H, W) im Bereich [0, 1]
        """
        # Sicherstellen, dass wir nicht mehr Samples plotten, als im Batch sind
        batch_size, channels, H, W = batch_images.shape
        num_samples = min(batch_size, num_samples)
        # 2 Zeilen (Bild oben, Maske unten)
        fig, axes = plt.subplots(2, num_samples, figsize=(16, 10))
        
        # Fallback, falls nur 1 Bild geplottet wird, damit das axes-Array 2D bleibt
        if num_samples == 1:
            axes = axes[:, np.newaxis] 
        for i in range(num_samples):
            # --- 1. BILD DENORMALISIEREN ---
            img = batch_images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * 0.5) + 0.5 
            img = np.clip(img, 0, 1)
            
            # --- 2. MASKE ENTPACKEN ---
            # Shape von (1, H, W) auf (H, W) reduzieren für Matplotlib
            mask = batch_masks[i].cpu().numpy().squeeze()
            # --- 3. PUNKTE DENORMALISIEREN ---
            pts = batch_points[i].cpu().numpy()
            pts_x = (pts[:, 0] + 1.0) / 2.0 * (W - 1)
            pts_y = (pts[:, 1] + 1.0) / 2.0 * (H - 1)
            # --- 4. PLOTTEN (ZEILE 1: BILD) ---
            axes[0, i].imshow(img)
            axes[0, i].plot(pts_x, pts_y, 'r.-', markersize=3, linewidth=1)
            axes[0, i].plot([pts_x[-1], pts_x[0]], [pts_y[-1], pts_y[0]], 'r-', linewidth=1)
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Sample {i+1} (RGB)")
            # --- 5. PLOTTEN (ZEILE 2: MASKE) ---
            axes[1, i].imshow(mask, cmap='gray')
            # Grüne Punkte zur besseren Sichtbarkeit auf schwarz/weiß
            axes[1, i].plot(pts_x, pts_y, 'g.-', markersize=3, linewidth=1)
            axes[1, i].plot([pts_x[-1], pts_x[0]], [pts_y[-1], pts_y[0]], 'g-', linewidth=1)
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Sample {i+1} (Maske)")
        plt.tight_layout()
        plt.show()