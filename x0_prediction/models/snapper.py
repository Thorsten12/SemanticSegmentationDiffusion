"""Post-DDIM normal-profile boundary snapper (V3/V5.2-style, standalone).

Design-Ziel: der Diffusions-Denoiser trifft die grobe Form gut, aber lokale
Details (exakte Kante) driften. Dieser Snapper ist bewusst NICHT in den
Diffusions-Forward-Pass eingezogen (kein additiver low_t-Gate wie V5.2) --
er läuft stattdessen einmalig NACH dem fertigen DDIM-Sampling, genau wie der
V3-`NormalBoundaryRefiner`.

Kernidee (identisch zu V5.2's SparseExactBoundaryCorrector, aber ohne den
dichten Boundary-Head -- hier wird direkt auf den rohen Backbone-Maps
gesampelt, um die Architektur schlank zu halten):

  1. Für jeden der N Konturpunkte wird entlang der lokalen Normalen ein
     kurzes 1-D-Profil aus den rohen Feature-Maps gelesen (mehrere Encoder-
     Skalen, optional plus rohes RGB).
  2. Ein kleiner Transformer/Conv-Block lässt die N Punkte entlang des
     Rings kommunizieren (zyklische Nachbarschaft, kein Positions-ID).
  3. Zwei Köpfe sagen (a) einen signierten Normal-Offset und (b) eine
     Konfidenz vorher. Die Korrektur wird nur `confidence^power`-gewichtet
     angewendet -- niedrige Konfidenz -> praktisch keine Korrektur, damit
     der Snapper nicht versucht, große (globale) Fehler zu "reparieren",
     für die er nicht gebaut ist.

Training: eigener, separater Loss (`teacher_loss`), analog zu V3 und
V5.2's Teacher-Pfad. Der Snapper wird NICHT auf echten On-Policy-DDIM-
Zwischenständen trainiert (die sind früh im Training oft weit von GT
entfernt -- das würde den Confidence-Kopf zu "immer unsicher" kollabieren
lassen), sondern auf künstlich normal-perturbierten GT-Konturen
(`perturb_along_normals`). Das gibt dem Snapper von Anfang an reichlich
realistische "fast richtig, nur noch Feinkorrektur nötig"-Beispiele,
unabhängig davon, wie gut der Diffusionspfad gerade ist.

`teacher_loss` kombiniert weiterhin die Offset-/Confidence-Regression
(`loss_offset`, `loss_conf`) mit einem NEUEN, optionalen BoundaryIoU-Term
(`loss_biou`): dafür wird aus `offset_pred`/`conf_logit` exakt dieselbe
korrigierte Kontur rekonstruiert, die `forward()` bei der Inferenz erzeugen
würde (gleiches Confidence-Gate), und gegen die GT-Maske rasterisiert. Das
gibt dem Snapper ein zusätzliches, geometrie-bewusstes Signal auf Maskenebene
-- unabhängig vom Diffusions- und Proposal-Loss, und ohne dass der Snapper
auf echten Diffusions-Rollouts trainiert werden muss (weiterhin nur die
künstlich perturbierten Teacher-Konturen).

Öffentliche API:
    snapper = BoundarySnapper(scale_channels=encoder.feature_channels, n_points=100)
    corrected = snapper(points, maps, image=None, hard=True)     # Inferenz
    loss, parts = snapper.teacher_loss(gt_points, maps, image=None)  # Training
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.rasterize import soft_boundary_iou_loss


# --------------------------------------------------------------------------
# Geometrie-Hilfsfunktionen (identisch zum V3/V5.2-Muster)
# --------------------------------------------------------------------------

def contour_frame(points: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tangente + Normale für eine geschlossene, geordnete Kontur [B,N,2]."""
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    tangent = next_p - prev_p
    tangent = tangent / torch.norm(tangent, dim=-1, keepdim=True).clamp(min=eps)
    normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
    return tangent, normal


def curvature_scalar(points: torch.Tensor) -> torch.Tensor:
    """Zweite-Differenz-Krümmungsmaß [B,N,1]."""
    prev_p = torch.roll(points, shifts=1, dims=1)
    next_p = torch.roll(points, shifts=-1, dims=1)
    return torch.norm(prev_p - 2.0 * points + next_p, dim=-1, keepdim=True)


def perturb_along_normals(
    points: torch.Tensor,
    max_offset: float = 0.08,
    smooth_passes: int = 2,
) -> torch.Tensor:
    """Erzeugt künstlich leicht-falsche Konturen aus GT für das Teacher-Training.

    Zufälliger Offset pro Vertex entlang der Normalen, geglättet über
    `smooth_passes` (gewichtetes Mitteln mit den Ringnachbarn), damit die
    Störung wie eine plausible, kontinuierliche Ungenauigkeit aussieht statt
    Vertex-für-Vertex-Rauschen.
    """
    _, normal = contour_frame(points)
    displacement = torch.empty(
        points.shape[0], points.shape[1], 1,
        device=points.device, dtype=points.dtype,
    ).uniform_(-float(max_offset), float(max_offset))
    for _ in range(max(int(smooth_passes), 0)):
        displacement = (
            torch.roll(displacement, 1, 1) + 2.0 * displacement
            + torch.roll(displacement, -1, 1)
        ) / 4.0
    return (points + displacement * normal).clamp(-1.0, 1.0)


def sample_at_points(feat_map: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """Bilineares Sampling einer [B,C,H,W]-Karte an [B,N,K,2]-Punkten (Range [-1,1]).

    Gibt [B,N,K,C] zurück. Nutzt F.grid_sample intern (grid_sample erwartet
    [B,H_out,W_out,2] -- wir flachen N*K in die H_out-Dimension).
    """
    b, n, k, _ = locations.shape
    grid = locations.view(b, n * k, 1, 2)
    sampled = F.grid_sample(feat_map, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)  # [B,C,N*K,1]
    sampled = sampled.squeeze(-1).transpose(1, 2)                       # [B,N*K,C]
    return sampled.view(b, n, k, -1)


# --------------------------------------------------------------------------
# Ring-Kommunikationsblock (identisch zum V5.2-Muster, hier eigenständig)
# --------------------------------------------------------------------------

class _CyclicRelativeBlock(nn.Module):
    """Kleiner ring-bewusster Kommunikationsblock über die N Konturpunkte.

    Fixer (nicht gelernter) Attention-Bias nach zyklischem Abstand -- kein
    gelerntes Positions-Embedding pro Vertex-Index, damit das Modul
    unabhängig von einer festen Vertex-Nummerierung bleibt. Nur ein milder
    Prior (nahe Nachbarn leicht bevorzugt), keine harte Maskierung.
    """

    def __init__(self, dim: int, num_heads: int, n_points: int, bias_strength: float = 0.12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, padding_mode="circular")
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.0)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        idx = torch.arange(n_points, dtype=torch.float32)
        d = (idx[:, None] - idx[None, :]).abs()
        d = torch.minimum(d, float(n_points) - d)
        d = d / max(float(n_points) / 2.0, 1.0)
        bias = float(bias_strength) * (1.0 - 2.0 * d)
        self.register_buffer("relative_bias", bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)
        x = x + F.gelu(y)
        y = self.norm2(x)
        y, _ = self.attn(y, y, y, attn_mask=self.relative_bias.to(dtype=y.dtype), need_weights=False)
        x = x + y
        x = x + self.ff(self.norm3(x))
        return x


# --------------------------------------------------------------------------
# Der Snapper selbst
# --------------------------------------------------------------------------

class BoundarySnapper(nn.Module):
    """Confidence-gated Normal-Offset-Snapper, Post-DDIM-Nachbearbeitung.

    Args:
        scale_channels: Kanalzahlen der rohen Backbone-Maps, genau wie
            `encoder.feature_channels` im Denoiser (`ContourDenoiser`).
        n_points: Anzahl Konturpunkte (muss zu deinem Diffusionsmodell passen).
        levels: wie viele der (feinsten) `scale_channels`-Level gesampelt werden.
        n_samples: Anzahl Profil-Samples entlang der Normalen (ungerade erzwungen,
            damit ein Sample exakt beim Vertex selbst liegt).
        radius: max. Suchradius entlang der Normalen (in denselben normalisierten
            Koordinaten wie deine Punkte, also [-1,1]-Bildraum).
        confidence_power: Schärfe des Confidence-Gates (`sigmoid(conf)^power`).
        use_rgb: zusätzlich rohes RGB an denselben Stellen sampeln (sparse,
            kein dichter RGB-Decoder).
    """

    def __init__(
        self,
        scale_channels: Sequence[int],
        n_points: int = 100,
        levels: int = 2,
        n_samples: int = 11,
        radius: float = 0.10,
        profile_dim: int = 20,
        hidden_dim: int = 64,
        ring_bands: int = 4,
        num_heads: int = 4,
        relative_bias_strength: float = 0.12,
        confidence_power: float = 2.0,
        use_rgb: bool = True,
    ):
        super().__init__()
        self.scale_channels = list(scale_channels)
        self.levels = max(1, min(int(levels), len(self.scale_channels)))
        self.n_points = int(n_points)
        self.n_samples = max(5, int(n_samples))
        if self.n_samples % 2 == 0:
            self.n_samples += 1
        self.radius = float(radius)
        self.profile_dim = int(profile_dim)
        self.hidden_dim = int(hidden_dim)
        self.confidence_power = max(float(confidence_power), 1.0)
        self.use_rgb = bool(use_rgb)

        self.sample_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, self.profile_dim), nn.GELU(),
                nn.Linear(self.profile_dim, self.profile_dim),
            )
            for c in self.scale_channels[: self.levels]
        ])
        self.rgb_proj = nn.Sequential(
            nn.Linear(3, self.profile_dim), nn.GELU(),
            nn.Linear(self.profile_dim, self.profile_dim),
        ) if self.use_rgb else None

        self.profile_mlp = nn.Sequential(
            nn.Linear(self.n_samples * self.profile_dim, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Fixes Ring-Positions-Feature (sin/cos über mehrere Harmonische),
        # kein gelerntes Embedding pro Index.
        import math
        phi = 2.0 * math.pi * torch.arange(self.n_points, dtype=torch.float32) / self.n_points
        ring = []
        for k in range(1, max(int(ring_bands), 1) + 1):
            ring.extend([(k * phi).sin(), (k * phi).cos()])
        ring = torch.stack(ring, dim=-1).unsqueeze(0)
        self.register_buffer("ring_features", ring, persistent=True)
        self.ring_proj = nn.Sequential(
            nn.Linear(ring.shape[-1], self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Geometrie-Features pro Vertex: xy + Tangente + Normale + Krümmung + Radialabstand
        self.geom_mlp = nn.Sequential(
            nn.Linear(2 + 2 + 2 + 1 + 1, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.ring_block = _CyclicRelativeBlock(
            self.hidden_dim, num_heads=num_heads, n_points=self.n_points,
            bias_strength=relative_bias_strength,
        )
        self.offset_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        # Start als Identität: der Snapper korrigiert bei Init noch nichts.
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        nn.init.zeros_(self.conf_head[-1].weight)
        nn.init.zeros_(self.conf_head[-1].bias)

        offsets = torch.linspace(-self.radius, self.radius, self.n_samples)
        self.register_buffer("profile_offsets", offsets, persistent=True)
        self.last_stats: Dict[str, torch.Tensor] = {}

    # -- gemeinsamer interner Vorwärtspfad: Profil lesen -> Offset+Conf ------

    def _predict(self, points: torch.Tensor, maps: Sequence[torch.Tensor],
                 image: Optional[torch.Tensor] = None):
        if points.shape[1] != self.n_points:
            raise ValueError(f"expected {self.n_points} points, got {points.shape[1]}")
        tangent, normal = contour_frame(points)
        offsets = self.profile_offsets.to(device=points.device, dtype=points.dtype)
        loc = points[:, :, None, :] + normal[:, :, None, :] * offsets.view(1, 1, -1, 1)
        loc = loc.clamp(-1.0, 1.0)

        profile = None
        for level in range(self.levels):
            raw = sample_at_points(maps[level], loc)              # [B,N,K,C]
            emb = self.sample_proj[level](raw)
            profile = emb if profile is None else profile + emb
        profile = profile / float(self.levels)
        if self.rgb_proj is not None and image is not None:
            rgb = sample_at_points(image, loc)
            profile = profile + self.rgb_proj(rgb)
        token = self.profile_mlp(profile.flatten(-2))

        center = points.mean(dim=1, keepdim=True)
        radial = torch.norm(points - center, dim=-1, keepdim=True)
        scale = radial.mean(dim=1, keepdim=True).clamp(min=1e-4)
        radial_norm = radial / scale
        curv = curvature_scalar(points)
        geom = torch.cat([points, tangent, normal, curv, radial_norm], dim=-1)
        token = token + self.geom_mlp(geom)
        token = token + self.ring_proj(self.ring_features.to(dtype=token.dtype)).expand(points.shape[0], -1, -1)
        token = self.ring_block(token)

        offset = self.radius * torch.tanh(self.offset_head(token).squeeze(-1))
        conf_logit = self.conf_head(token).squeeze(-1)
        return offset, conf_logit, normal

    # -- gemeinsame Rekonstruktion der korrigierten Kontur aus (offset, conf) -

    def _apply_gate(self, points: torch.Tensor, offset: torch.Tensor,
                     conf_logit: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        """Identisches Confidence-Gating wie in `forward()`, aber als eigene
        Methode, damit `teacher_loss` dieselbe rekonstruierte Kontur für den
        neuen BoundaryIoU-Term nutzen kann, ohne die Inferenz-API zu berühren."""
        conf_prob = torch.sigmoid(conf_logit)
        gate = conf_prob.pow(self.confidence_power)
        correction = gate * offset
        return (points + normal * correction.unsqueeze(-1)).clamp(-1.0, 1.0)

    # -- öffentliche Inferenz-API --------------------------------------------

    def forward(self, points: torch.Tensor, maps: Sequence[torch.Tensor],
                image: Optional[torch.Tensor] = None, hard: bool = True) -> torch.Tensor:
        """Snapt eine fertige Kontur (z.B. DDIM-Sampling-Output). Gibt korrigierte
        Punkte zurück. `hard=True` ist der übliche Inferenz-Fall (volle
        Gate-Anwendung); `hard=False` existiert nur der Vollständigkeit halber."""
        offset, conf_logit, normal = self._predict(points, maps, image=image)
        corrected = self._apply_gate(points, offset, conf_logit, normal)

        with torch.no_grad():
            conf_prob = torch.sigmoid(conf_logit)
            gate = conf_prob.pow(self.confidence_power)
            correction = gate * offset
            self.last_stats = {
                "snap_offset_abs": offset.detach().abs().mean(),
                "snap_conf_mean": conf_prob.detach().mean(),
                "snap_gate_mean": gate.detach().mean(),
                "snap_correction_abs": correction.detach().abs().mean(),
            }
        return corrected

    # -- Teacher-Loss für separates Snapper-Training -------------------------

    def teacher_loss(
        self,
        gt_points: torch.Tensor,
        maps: Sequence[torch.Tensor],
        image: Optional[torch.Tensor] = None,
        perturb_max_offset: float = 0.08,
        perturb_smooth_passes: int = 2,
        confidence_radius: float = 0.060,
        tangent_tolerance: float = 0.040,
        masks: Optional[torch.Tensor] = None,
        lambda_biou: float = 0.0,
        biou_size: int = 64,
    ):
        """Separater Teacher-Loss (V3/V5.2-Stil): der Snapper trainiert NICHT auf
        echten Diffusions-Zwischenständen, sondern auf künstlich normal-
        perturbierten GT-Konturen. Das gibt dem Confidence-Kopf von Anfang an
        genug reale "nah an GT"-Positivbeispiele, unabhängig vom Trainingsstand
        des Diffusionsmodells.

        NEU: optional ein zusätzlicher, zeitunabhängiger BoundaryIoU-Term
        (`lambda_biou > 0`, braucht `masks`). Dafür wird aus `offset_pred` /
        `conf_logit` -- via `_apply_gate`, exakt dasselbe Gating wie in
        `forward()` -- die vollständige korrigierte Kontur rekonstruiert und
        gegen die GT-Maske rasterisiert. Bewertet also direkt, wie gut die
        vom Snapper *tatsächlich ausgegebene* Kontur (inkl. Confidence-Gate)
        geometrisch zur GT-Maske passt -- zusätzlich zur reinen Offset-/
        Confidence-Regression, die nur die einzelnen Vorhersagen prüft.
        Bleibt bei `lambda_biou == 0` (Default) exakt das alte Verhalten.

        Gibt (total_loss, parts-dict) zurück -- total_loss additiv ins normale
        Trainings-Backward einhängen, z.B.:
            snap_loss, snap_parts = snapper.teacher_loss(points, cond_maps_raw)
            loss = diffusion_loss + lambda_snap * snap_loss
        """
        teacher_points = perturb_along_normals(
            gt_points.detach(), max_offset=perturb_max_offset,
            smooth_passes=perturb_smooth_passes,
        )
        offset_pred, conf_logit, normal = self._predict(teacher_points, maps, image=image)

        with torch.no_grad():
            d = torch.cdist(teacher_points.float(), gt_points.float(), p=2)
            min_dist, idx = d.min(dim=2)
            b_idx = torch.arange(teacher_points.shape[0], device=teacher_points.device)[:, None]
            nearest = gt_points[b_idx, idx]
            delta = nearest - teacher_points
            tangent = torch.stack([normal[..., 1], -normal[..., 0]], dim=-1)
            signed_n = (delta * normal).sum(dim=-1)
            signed_t = (delta * tangent).sum(dim=-1).abs()
            target_conf = (
                (min_dist <= float(confidence_radius))
                & (signed_t <= float(tangent_tolerance))
                & (signed_n.abs() <= self.radius)
            ).to(teacher_points.dtype)
            target_offset = signed_n.clamp(-self.radius, self.radius)

        pos = target_conf.sum().clamp(min=1.0)
        neg = (target_conf.numel() - target_conf.sum()).clamp(min=1.0)
        pos_weight = (neg / pos).clamp(1.0, 12.0)
        loss_conf = F.binary_cross_entropy_with_logits(conf_logit, target_conf, pos_weight=pos_weight)

        w = target_conf
        if float(w.sum().item()) > 0:
            loss_offset = F.smooth_l1_loss(
                offset_pred / max(self.radius, 1e-6),
                target_offset / max(self.radius, 1e-6),
                reduction="none",
            )
            loss_offset = (loss_offset * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss_offset = offset_pred.new_zeros(())

        if masks is not None and lambda_biou > 0:
            corrected = self._apply_gate(teacher_points, offset_pred, conf_logit, normal)
            loss_biou = soft_boundary_iou_loss(corrected, masks, size=biou_size)
        else:
            loss_biou = offset_pred.new_zeros(())

        total = loss_offset + loss_conf + lambda_biou * loss_biou
        parts = {
            "snap_teacher_loss_offset": loss_offset.detach(),
            "snap_teacher_loss_conf": loss_conf.detach(),
            "snap_teacher_conf_rate": target_conf.detach().mean(),
            "snap_teacher_target_dist": min_dist.detach().mean(),
            "snap_teacher_loss_biou": loss_biou.detach(),
        }
        return total, parts