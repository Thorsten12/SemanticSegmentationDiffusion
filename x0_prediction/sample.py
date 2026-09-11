"""Evaluate / sample from a trained P2SDiff model.

As a library: `evaluate(...)` runs DDIM sampling over a loader, rasterizes the
predicted points and returns mean Dice / IoU (used by training for validation).

As a CLI:
    python -m src.sample --ckpt src/runs/baseline/best.pth --split test
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .diffusion import GaussianDiffusion
from .data import build_contour_dataset, split_counts
from .models import ContourDenoiser, build_conditioner, BoundarySnapper, ContourProposalHead, apply_proposal_target
from .utils import dice_score, iou_score, points_to_mask, save_prediction_grid


# ---------------------------------------------------------------------------
# NEW: GIF rendering of the denoising trajectory (--gif / --gif_steps).
# ---------------------------------------------------------------------------

def _render_denoising_gif(image_chw, gt_points, intermediates, gif_path,
                           img_size=None, fps=4):
    """Render one sample's denoising trajectory to an animated GIF.

    image_chw: [C,H,W] tensor (CPU), already in the model's normalized/display
        range -- displayed as-is (min-max stretched) as the background.
    gt_points: [N,2] tensor in [-1,1], the ground-truth contour (drawn once,
        static, in every frame for reference).
    intermediates: list of [N,2] tensors in [-1,1], one per collected DDIM
        step, in sampling order (noisy/coarse -> clean). This is exactly
        `ddim_sample`'s new second return value, already sliced down to a
        single sample.
    gif_path: output path, e.g. ".../val_epoch12_sample0.gif".
    img_size: pixel size to render each frame at (defaults to the image's
        own H). Only affects display resolution, not the underlying points.

    Uses matplotlib (Agg backend, no display needed) to draw each frame,
    then imageio to assemble the GIF. Both are already project dependencies
    (matplotlib via train.py's plotting; imageio is added as a new,
    lightweight dependency -- `pip install imageio` if not already present).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    img = image_chw.numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    h = img_size or img.shape[0]

    gt_xy = gt_points.numpy()

    frames = []
    n_steps = len(intermediates)
    for step_idx, pts in enumerate(intermediates):
        pts_xy = pts.numpy()

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(img, extent=(-1, 1, 1, -1))  # map pixel axes to [-1,1] coord space
        ax.plot(np.append(gt_xy[:, 0], gt_xy[0, 0]),
                np.append(gt_xy[:, 1], gt_xy[0, 1]),
                "-", color="lime", linewidth=1.5, alpha=0.8, label="GT")
        ax.plot(np.append(pts_xy[:, 0], pts_xy[0, 0]),
                np.append(pts_xy[:, 1], pts_xy[0, 1]),
                "-o", color="red", linewidth=1.5, markersize=2, label="pred")
        ax.set_xlim(-1, 1); ax.set_ylim(1, -1)
        ax.axis("off")
        ax.set_title(f"step {step_idx + 1}/{n_steps}", fontsize=9)
        if step_idx == 0:
            ax.legend(loc="lower right", fontsize=7, framealpha=0.6)

        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        frames.append(frame)

    # Hold the final (clean) frame a bit longer so the GIF doesn't just
    # snap back to the start when it loops.
    frames = frames + [frames[-1]] * max(0, int(fps))

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)


@torch.no_grad()
def evaluate(encoder, denoiser, diffusion, loader, cfg, device, snapper=None,
             proposal_head=None, viz_path=None,
             snap_mode="post", snap_t_threshold=None, snap_every=1,
             gif=False, gif_steps=10, gif_path=None, gif_n_samples=2):
    """Sample boundaries, rasterize, and return (mean Dice, mean IoU).

    `snapper`: optional BoundarySnapper.
    `snap_mode` controls WHEN/HOW it's applied (V6):
      - "post"   (default, legacy V3/V5.2-standalone behavior): applied once
                 after DDIM sampling finishes, as a pure post-processing step.
                 Matches the old (pre-V6) behavior exactly when snapper is given.
      - "loop"   : applied INSIDE `diffusion.ddim_sample`, gated to steps with
                 `t <= snap_t_threshold` (see diffusion.py's ddim_sample
                 docstring) -- matches the V5.2 README's "low-t only, inside
                 the diffusion process" description. No post-hoc call is made
                 in this mode.
      - "both"   : in-loop snapping during sampling AND an additional post-hoc
                 pass on the final output. Mostly useful for ablating whether
                 a final cleanup pass on top of in-loop correction helps.
      - "none"   : snapper is ignored even if provided (equivalent to passing
                 snapper=None, kept as an explicit option for CLI ablations).
    `snap_t_threshold`: absolute diffusion timestep (not a fraction) -- required
        (non-None) for "loop"/"both". Callers (train.py, sample.py's CLI)
        convert the user-facing `--snap_t_threshold_frac` (fraction of
        `cfg.timesteps`) to this absolute value before calling `evaluate`.
    `snap_every`: forwarded to ddim_sample for "loop"/"both" -- see there.

    `proposal_head`: optional ContourProposalHead (V5/V6). If given, computed
    once per batch from the raw encoder pyramid and combined with the
    denoiser's per-step x0 prediction according to `cfg.proposal_target`
    inside `diffusion.ddim_sample`. Pass None (with cfg.proposal_target ==
    "absolute") to evaluate without any proposal, e.g. for the
    ellipse+absolute / plain baseline ablation. Supports multi-shape unions
    (`proposal_head.n_shapes > 1`) transparently -- `proposal_head(raw)`
    always returns a single [B, n_points, 2] contour regardless of n_shapes.

    --- NEW: denoising-trajectory GIF (V7) ---------------------------------
    gif: if True, renders an animated GIF of the DDIM denoising trajectory
        (noisy -> clean contour) for the first `gif_n_samples` items of the
        FIRST batch only (matching how `viz_path`'s PNG grid is cached from
        just the first batch, so this costs no extra encoder/denoiser passes
        beyond the ones evaluate() already runs for scoring).
    gif_steps: number of frames to sample across the DDIM trajectory
        (default 10). Internally converted to `collect_every =
        max(1, ddim_steps // gif_steps)` and passed to `ddim_sample`; the
        actual frame count is `ceil(ddim_steps / collect_every)` (+1 for the
        guaranteed final step), so it may not be exactly `gif_steps` for
        DDIM step counts that don't divide evenly -- this is intentional
        (better to slightly over/under-shoot than silently skip the last,
        cleanest step).
    gif_path: base output path WITHOUT extension, e.g.
        ".../val_epoch12". One file per rendered sample is written as
        "{gif_path}_sample{i}.gif". If None while `gif=True`, defaults to
        the same directory/stem as `viz_path` (with ".png" stripped).
    gif_n_samples: how many samples (from the first batch) to render GIFs
        for. Small by default since each is its own file and rendering
        cost scales with it.
    """
    encoder.eval(); denoiser.eval()
    if snapper is not None:
        snapper.eval()
    if proposal_head is not None:
        proposal_head.eval()

    use_loop_snap = snapper is not None and snap_mode in ("loop", "both")
    use_post_snap = snapper is not None and snap_mode in ("post", "both")
    if use_loop_snap and snap_t_threshold is None:
        raise ValueError("snap_mode='loop'/'both' requires snap_t_threshold to be set")

    if gif and gif_path is None and viz_path is not None:
        gif_path = os.path.splitext(viz_path)[0]

    collect_every = 0
    if gif:
        collect_every = max(1, cfg.ddim_steps // max(1, int(gif_steps)))

    dices, ious = [], []
    viz_cache = None
    gif_rendered = False

    for images, gt_points, gt_masks in loader:
        images = images.to(device)
        raw = encoder.extract(images)                  # backbone runs once
        proposal = proposal_head(raw) if proposal_head is not None else None
        cond_fn = lambda t_b: encoder.fuse(raw, t_b)   # time-conditioned per step
        shape = (images.shape[0], cfg.n_points, 2)

        # NEW: only request intermediates on the very first batch (the one
        # we'll actually render), to avoid needlessly holding onto the full
        # trajectory tensor list for every batch in the loader.
        want_gif_this_batch = gif and not gif_rendered
        sample_out = diffusion.ddim_sample(
            denoiser, cond_fn, shape,
            proposal=proposal, proposal_target=cfg.proposal_target,
            ddim_steps=cfg.ddim_steps, guidance_scale=cfg.guidance_scale, clamp=1.0,
            snapper=snapper if use_loop_snap else None,
            snap_maps=raw if use_loop_snap else None,
            snap_image=images if use_loop_snap else None,
            snap_t_threshold=snap_t_threshold if use_loop_snap else None,
            snap_every=snap_every,
            collect_every=collect_every if want_gif_this_batch else 0,
        )
        if want_gif_this_batch:
            pred_points, intermediates = sample_out
        else:
            pred_points = sample_out
            intermediates = None

        if use_post_snap:
            pred_points = snapper(pred_points, raw, image=images, hard=True)
            if want_gif_this_batch and intermediates is not None:
                intermediates = intermediates + [pred_points.detach().clone()]

        pred_masks_np, batch_scores = [], []
        for i in range(images.shape[0]):
            pred_mask = points_to_mask(pred_points[i], cfg.img_size)
            gt_mask = gt_masks[i].squeeze().cpu().numpy().astype(np.uint8)
            d = dice_score(pred_mask, gt_mask)
            j = iou_score(pred_mask, gt_mask)
            dices.append(d); ious.append(j)
            pred_masks_np.append(pred_mask)
            batch_scores.append({"dice": d, "iou": j})

        if viz_path is not None and viz_cache is None:
            prop_cpu = proposal.cpu() if proposal is not None else None
            viz_cache = (images.cpu(), gt_points, pred_points.cpu(),
                         gt_masks, pred_masks_np, batch_scores, prop_cpu)

        if want_gif_this_batch and intermediates is not None:
            n_render = min(int(gif_n_samples), images.shape[0])
            images_cpu = images.cpu()
            for i in range(n_render):
                sample_intermediates = [step[i].cpu() for step in intermediates]
                out_path = f"{gif_path}_sample{i}.gif"
                _render_denoising_gif(
                    images_cpu[i], gt_points[i].cpu(),
                    sample_intermediates, out_path,
                )
            gif_rendered = True

    if viz_path is not None and viz_cache is not None:
        imgs, gtp, pp, gtm, pm, sc, prop = viz_cache
        save_prediction_grid(imgs, gtp, pp, gtm, pm, viz_path,
                             max_samples=min(4, imgs.shape[0]), scores=sc,
                             proposal_points=prop)

    return float(np.mean(dices)), float(np.mean(ious))


def _resolve_residual_target(cfg) -> str:
    """Same rule as train.py's _resolve_residual_target -- kept in sync so a
    checkpoint's denoiser is reconstructed with the SAME residual semantics
    it was trained with."""
    proposal_target = getattr(cfg, "proposal_target", "absolute")
    if proposal_target == "residual":
        return "zero"
    return "input" if getattr(cfg, "predict_residual", True) else "none"


def _resolve_sampler_type(cfg) -> str:
    """Same backward-compat rule as ContourDenoiser.__init__: older saved
    configs only have `use_deformable_sampling` (bool), not `sampler_type`
    (str). Resolve it the same way ContourDenoiser itself would, so
    load_checkpoint constructs the SAME sampler class the checkpoint was
    actually trained with, regardless of which generation of config it came
    from.

    - Newer checkpoints: `sampler_type` in {"input","deformable","attention"}
      is already present and authoritative.
    - Older checkpoints (pre-`sampler_type`): only `use_deformable_sampling`
      exists. True -> "deformable", False/absent -> "input". These never
      had "attention" as an option, so no ambiguity there.
    """
    sampler_type = getattr(cfg, "sampler_type", "input")
    use_deform = getattr(cfg, "use_deformable_sampling", False)
    if sampler_type == "input" and use_deform:
        return "deformable"
    if sampler_type not in ("input", "deformable", "attention"):
        raise ValueError(f"unknown sampler_type in checkpoint config: {sampler_type!r}")
    return sampler_type


def load_checkpoint(ckpt_path, cfg, device):
    """Returns (encoder, denoiser, snapper, proposal_head)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer the config stored in the checkpoint so the architecture matches.
    saved = ckpt.get("config")
    if saved:
        for k in ("encoder", "backbone", "cond_channels", "stem_dim", "n_points", "hidden_dim",
                  "n_transformer_layers", "n_heads", "pretrained", "freeze_backbone",
                  "in_channels", "unet_start_dim", "unet_dim_mults", "unet_groupnorm_groups",
                  "coord_fourier_bands", "pos_grid_bands", "img_size", "npy_size",
                  "pvt_variant", "pvt_pretrained_path",
                  "snap_n_samples", "snap_radius", "snap_levels", "snap_profile_dim",
                  "snap_hidden_dim", "snap_ring_bands", "snap_num_heads",
                  "snap_relative_bias", "snap_confidence_power", "snap_use_rgb",
                  "proposal_target", "proposal_harmonics", "proposal_hidden_dim",
                  "proposal_num_heads",
                  # V6: multi-shape / shape-type proposal knobs.
                  "proposal_type", "proposal_n_shapes", "proposal_union_raster_size",
                  # V6: in-loop snapping knobs (needed at eval/sample time,
                  # not part of the model's own state_dict).
                  "snap_mode", "snap_t_threshold_frac", "snap_every",
                  # needed to reconstruct the denoiser's forward-pass wiring
                  # (residual_target) identically to how it was trained.
                  "predict_residual", "pos_scale", "attn_mask", "timesteps",
                  "guidance_time_scale",
                  # NEW: multi-scale sampling strategy ablation (see
                  # models/denoiser.py: MultiScalePointSampler /
                  # DeformableScalePointSampler / AttentionScalePointSampler).
                  # `sampler_type` is the current, authoritative field;
                  # `use_deformable_sampling` is kept for older checkpoints
                  # that predate `sampler_type` (see _resolve_sampler_type
                  # above -- same backward-compat rule ContourDenoiser
                  # applies internally). Checkpoints missing ANY of these
                  # keys simply skip them here and cfg keeps its dataclass
                  # default (sampler_type="input",
                  # use_deformable_sampling=False) -- old checkpoints load
                  # exactly as before, no migration needed.
                  "sampler_type", "use_deformable_sampling",
                  "deform_n_samples", "deform_min_scale_res", "deform_radius_cells",
                  "attn_sampler_window", "attn_sampler_heads",
                  "top_k_scales"):
            if k in saved:
                setattr(cfg, k, saved[k])
        # Checkpoints saved before the high-res stem existed have no stem weights.
        if "stem_dim" not in saved:
            cfg.stem_dim = 0

    # Don't re-download ImageNet weights at load time; the checkpoint already has them.
    cfg.pretrained = False
    encoder = build_conditioner(cfg).to(device)

    residual_target = _resolve_residual_target(cfg)
    sampler_type = _resolve_sampler_type(cfg)
    denoiser = ContourDenoiser(
        pos_scale=cfg.pos_scale,
        n_points=cfg.n_points, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.n_transformer_layers, num_heads=cfg.n_heads,
        attn_window=cfg.attn_mask,
        scale_channels=encoder.feature_channels, proj_dim=cfg.cond_channels,
        coord_fourier_bands=cfg.coord_fourier_bands,
        timesteps=cfg.timesteps,
        predict_residual=cfg.predict_residual,
        residual_target=residual_target,
        guidance_time_scale=cfg.guidance_time_scale,
        # NEW: multi-scale sampling strategy ablation. `sampler_type` is
        # passed explicitly as the already-resolved value (see
        # _resolve_sampler_type above) so this is correct for BOTH old
        # checkpoints (only had use_deformable_sampling) and new ones
        # (have sampler_type directly, incl. "attention"). getattr(...)
        # defaults match Config's dataclass defaults, so this is safe even
        # if `cfg` was ever constructed some other way that skipped them
        # (belt-and-suspenders; with the whitelist above, a real Config
        # will always have these set after loading a checkpoint).
        sampler_type=sampler_type,
        use_deformable_sampling=getattr(cfg, "use_deformable_sampling", False),
        deform_n_samples=getattr(cfg, "deform_n_samples", 4),
        deform_min_scale_res=getattr(cfg, "deform_min_scale_res", 28),
        deform_radius_cells=getattr(cfg, "deform_radius_cells", 2.5),
        attn_sampler_window=getattr(cfg, "attn_sampler_window", 5),
        attn_sampler_heads=getattr(cfg, "attn_sampler_heads", 4),
        top_k_scales=getattr(cfg, "top_k_scales", 3),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    denoiser.load_state_dict(ckpt["denoiser"])

    proposal_head = None
    if getattr(cfg, "proposal_target", "absolute") != "absolute":
        coarsest_channels = encoder.feature_channels[-1]
        proposal_head = ContourProposalHead(
            coarsest_channels=coarsest_channels,
            n_points=cfg.n_points,
            hidden_dim=cfg.proposal_hidden_dim,
            num_heads=cfg.proposal_num_heads,
            harmonics=cfg.proposal_harmonics,
            proposal_type=getattr(cfg, "proposal_type", "fourier"),
            n_shapes=getattr(cfg, "proposal_n_shapes", 1),
            union_raster_size=getattr(cfg, "proposal_union_raster_size", 64),
        ).to(device)
        if "proposal_head" in ckpt:
            proposal_head.load_state_dict(ckpt["proposal_head"])
        else:
            print(f"  warn: no 'proposal_head' in {ckpt_path} -- using freshly-initialized proposal head")

    snapper = BoundarySnapper(
        scale_channels=encoder.feature_channels,
        n_points=cfg.n_points,
        levels=cfg.snap_levels,
        n_samples=cfg.snap_n_samples,
        radius=cfg.snap_radius,
        profile_dim=cfg.snap_profile_dim,
        hidden_dim=cfg.snap_hidden_dim,
        ring_bands=cfg.snap_ring_bands,
        num_heads=cfg.snap_num_heads,
        relative_bias_strength=cfg.snap_relative_bias,
        confidence_power=cfg.snap_confidence_power,
        use_rgb=cfg.snap_use_rgb,
    ).to(device)
    if "snapper" in ckpt:
        snapper.load_state_dict(ckpt["snapper"])
    else:
        print(f"  warn: no 'snapper' in {ckpt_path} -- using freshly-initialized (identity) snapper")

    return encoder, denoiser, snapper, proposal_head


def main():
    parser = argparse.ArgumentParser(description="Evaluate / sample P2SDiff")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", choices=["ph2", "isic2017", "isic2018", "ham10000"])
    parser.add_argument("--skin_root", type=str)
    parser.add_argument("--encoder", choices=["convnext", "pvt", "unet"])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--viz", type=str, default="prediction_grid.png")
    parser.add_argument("--no_snap", action="store_true",
                        help="Evaluate without the boundary snapper at all (overrides --snap_mode).")
    parser.add_argument("--snap_mode", choices=["post", "loop", "both", "none"], default=None,
                        help="How to apply the snapper. Defaults to the checkpoint's saved "
                             "config value, or 'post' if not present (legacy behavior).")
    parser.add_argument("--snap_t_threshold_frac", type=float, default=None,
                        help="In-loop snapping (--snap_mode loop/both) fires in the last X%% "
                             "of timesteps (fraction in [0,1]). Required for "
                             "--snap_mode loop/both unless present in the checkpoint config.")
    parser.add_argument("--snap_every", type=int, default=None)
    # ----- denoising-trajectory GIF -----
    parser.add_argument("--gif", action="store_true",
                        help="Additionally render an animated GIF of the DDIM denoising "
                             "trajectory for a few samples from the first batch of --split.")
    parser.add_argument("--gif_steps", type=int, default=10,
                        help="Approximate number of frames across the DDIM trajectory "
                             "(default 10). Converted internally to a stride over "
                             "--ddim_steps; see evaluate()'s docstring for the exact rule.")
    parser.add_argument("--gif_n_samples", type=int, default=2,
                        help="How many samples from the first batch to render GIFs for.")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    split = "vl" if args.split == "val" else "te"
    ds = build_contour_dataset(cfg.skin_root, cfg.dataset, split, cfg.n_points,
                               cfg.img_size, augment=False, npy_size=cfg.npy_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    encoder, denoiser, snapper, proposal_head = load_checkpoint(args.ckpt, cfg, device)
    diffusion = GaussianDiffusion(cfg.timesteps, cfg.beta_start, cfg.beta_end, device=device)

    snap_mode = args.snap_mode or getattr(cfg, "snap_mode", "post")
    snap_t_threshold_frac = args.snap_t_threshold_frac if args.snap_t_threshold_frac is not None \
        else getattr(cfg, "snap_t_threshold_frac", 0.15)
    snap_t_threshold = int(round(snap_t_threshold_frac * cfg.timesteps))
    snap_every = args.snap_every if args.snap_every is not None else getattr(cfg, "snap_every", 1)

    active_snapper = None if args.no_snap else snapper
    dice, iou = evaluate(encoder, denoiser, diffusion, loader, cfg, device,
                         snapper=active_snapper, proposal_head=proposal_head, viz_path=args.viz,
                         snap_mode=snap_mode, snap_t_threshold=snap_t_threshold, snap_every=snap_every,
                         gif=args.gif, gif_steps=args.gif_steps, gif_n_samples=args.gif_n_samples)
    tag = "no-snap" if args.no_snap else f"snap={snap_mode}"
    sampler_tag = getattr(cfg, "sampler_type", "input")
    print(f"[{args.split}] ({tag}, sampler={sampler_tag}) Dice {dice:.4f} | IoU {iou:.4f} | viz -> {args.viz}")
    if args.gif:
        base = os.path.splitext(args.viz)[0]
        print(f"GIF(s) -> {base}_sample*.gif")


if __name__ == "__main__":
    main()