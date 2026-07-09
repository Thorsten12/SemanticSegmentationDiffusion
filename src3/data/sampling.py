import numpy as np
import matplotlib.pyplot as plt

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

import os

def canonicalize_contour(contour: np.ndarray, debug: bool = False,
                          debug_path: str = "./output", debug_tag: str = "sample"):
    """Erzwingt feste Umlaufrichtung + reproduzierbaren Startpunkt via PCA."""

    pts = contour.astype(np.float64)

    x, y = pts[:, 0], pts[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    signed_area = 0.5 * np.sum(x * y_next - x_next * y)

    if signed_area < 0:
        pts = pts[::-1]

    # Zentroid
    x, y = pts[:, 0], pts[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_next - x_next * y
    A = 0.5 * np.sum(cross)
    A = A if abs(A) > 1e-8 else 1e-8
    cx = np.sum((x + x_next) * cross) / (6 * A)
    cy = np.sum((y + y_next) * cross) / (6 * A)
    centroid = np.array([cx, cy])

    # PCA
    centered = pts - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]

    proj = centered @ main_axis

    skew = np.mean(proj ** 3)
    if skew < 0:
        main_axis = -main_axis
        proj = -proj

    start_idx = int(np.argmax(proj))
    pts_rolled = np.roll(pts, -start_idx, axis=0)

    if debug:
        if debug_path is None:
            debug_path = "/mnt/user-data/outputs/contour_debug"
        os.makedirs(debug_path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(pts_rolled[:, 0], pts_rolled[:, 1], '-', color='lightgray', linewidth=1, zorder=1)
        sc = ax.scatter(pts_rolled[:, 0], pts_rolled[:, 1],
                         c=np.arange(len(pts_rolled)), cmap='viridis', s=15, zorder=2)
        fig.colorbar(sc, ax=ax, label='Punktindex')
        ax.scatter(*centroid, color='red', marker='x', s=100, label='Zentroid', zorder=3)
        scale = np.max(np.abs(proj))
        p1 = centroid - main_axis * scale
        p2 = centroid + main_axis * scale
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color='orange', label='Hauptachse')
        ax.scatter(pts_rolled[0, 0], pts_rolled[0, 1], color='lime', s=150,
                   edgecolor='black', linewidth=1.5, label='Punkt 0', zorder=4)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.legend()
        ax.set_title(f"start_idx={start_idx}, signed_area vorher={signed_area:.1f}")

        fig.savefig(os.path.join(debug_path, f"{debug_tag}.png"), dpi=120, bbox_inches='tight')
        plt.close(fig)   # <-- wichtig: schließt die Figure, kein Leak mehr

    return pts_rolled.astype(np.float32)



def adaptive_sampling():
    ...