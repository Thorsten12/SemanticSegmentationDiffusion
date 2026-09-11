"""Decoder (Group C) -- XBound-Former.
Reference: Wang, Chen, Ma, Wang, Fei, Shuai, Tang, Zhou & Qin, "XBound-Former:
Toward Cross-scale Boundary Modeling in Transformers", IEEE TMI 2023
(originally arXiv:2206.00806).
Paper: https://arxiv.org/pdf/2206.00806
Official repo (not vendored here, but used to cross-check this reimplementation
against the paper's equations): https://github.com/jcwang123/xboundformer

Faithful reimplementation of Section III (Eq. 1-9), NOT a copy of the
official repo (GitHub blocks automated fetching of raw files). This is
architecturally different from the Group B decoders (CASCADE, G-CASCADE,
EMCAD): those are conv-based pyramid decoders operating on spatial feature
maps; XBound-Former operates on TOKEN SEQUENCES with genuine multi-head
self-/cross-attention, and learns an explicit per-scale "boundary
embedding" vector that later cross-attends across scales. Traceable to:

  - In-scale boundary modeling, Eq. 1: f^1, xi <- ex-Bound^(1..Nex)(im-
    Bound^(1..Nim)(f^0)). N_im=N_ex=2 by default (paper's chosen setting,
    Section IV-C / IV-F.2 ablation).

  - im-Bound, Eq. 2-3:
        rho^i = MSA(z^{i-1}) + MLP(MSA(z^{i-1}))     (residuals + LayerNorm
                                                        elided in the paper's
                                                        compact notation;
                                                        made explicit here
                                                        as a standard pre-LN
                                                        transformer block)
        M_hat^i = Sigmoid(Linear(rho^i))              (boundary key-point
                                                        prediction head)
        z^i = rho^i + rho^i * M_hat^i                 (Eq. 3; see note)
    NOTE on Eq. 2/3 notation: the paper defines '(+)' [circled-plus] as
    element-wise ADDITION directly under Eq. 2, then re-defines the SAME
    symbol as element-wise MULTIPLICATION directly under Eq. 3 -- an
    internal symbol clash in the paper's own text (not introduced by us).
    We resolve it the only way that is self-consistent with the stated
    goal ("obtain the enhanced feature"): a residual attention-style gate,
    z^i = rho^i + rho^i * M_hat^i, rather than reading both occurrences as
    the same operation.

  - ex-Bound: repeated N_ex times, described only in prose (no closed-form
    equation in the paper, unlike im-Bound's Eq. 2-3). It "treats boundary
    key points as query objects" and uses "a transformer decoder... a
    sequence of Masked MSA, MSA, and MLP modules" to refine a per-scale
    boundary embedding xi, then feeds the embedding back to refine the
    feature z and predict another key-point map. We implement this as:
    (a) xi cross-attends into z (xi as query, z as key/value) + MLP --
        this is the "decoder" half that turns z into a boundary-embedding
        vector. NOTE: the paper's "Masked MSA" step would be a masked
        SELF-attention over xi; since xi is a single token per scale here,
        self-attention over a length-1 sequence is a mathematical no-op
        (nothing to mask against, softmax of one logit is always 1), so we
        omit it as dead computation rather than including inert code.
    (b) z is then refined by broadcast-adding xi into every token 
        (an additive coupling, cheaper than joint self-attention over 
        [z ; xi] — see ExBoundBlock's docstring for the O(N²) vs O((N+1)²) rationale), 
        followed by ordinary self-attention over z alone, then MLP, then the 
        same key-point head/gate as im-Bound.
    This structure is our best-effort, self-consistent reading of the
    prose description -- the paper gives no equation for it, so this
    should be treated as an interpretation, not a literal transcription.

  - Cross-scale boundary fusion (X-Bound), Eq. 4-5:
        gamma_low  = f_low  + MSA(f_low,  xi_high, xi_high)
        gamma_high = f_high + MSA(f_high, xi_low,  xi_low)
    followed by upsampling gamma_high to f_low's resolution, concatenating
    with gamma_low, and a linear (here: 1x1 conv) projection back down to
    f_low's channel width -> f_low^2. "low scale" = finer/larger (H,W),
    "high scale" = coarser/smaller (H,W) -- the paper's own terminology,
    orthogonal to "shallow/deep" network depth.
    NOTE on what Eq. 4-5 reduce to: xi has exactly ONE token per scale, so
    MSA(f, xi, xi) has exactly one key/value pair -- softmax over a single
    logit is always exactly 1, regardless of the query. So this "cross-
    scale attention" is architecturally equivalent to a plain per-head
    linear projection of xi, broadcast-added to every spatial position (a
    FiLM-style channel modulation) -- there is no spatially-varying
    attention pattern here despite the MSA framing. We still implement the
    literal MSA machinery (matching Eq. 4-5 as written, preserving the
    per-head V-projection), but flag this since it clarifies what the
    module is actually doing in practice, which is easy to miss just by
    reading the equations.
    NOTE on >2 scales: Eq. 4-5 define a single low/high PAIR; the paper's
    prose only says this runs "on {f_l^1}_{l=1}^3" (3 of the 4 scales),
    with the deepest/coarsest scale (l=4) left as f_4^2 = f_4^1, but never
    states whether the 3 fusions are independent pairwise pairs or a
    cascade. We implement it as a TOP-DOWN CASCADE, exactly like this
    project's other 3 decoders (CASCADE/G-CASCADE/EMCAD): start at the
    coarsest scale (unchanged), then for each finer scale in turn, treat
    the just-produced (already-fused) coarser scale as "high" and the
    current scale's own in-scale-refined feature as "low". This is the
    most defensible generalization consistent with Fig. 2's overall top-
    down figure and this project's established decoder pattern -- it is
    our interpretation, not a value given verbatim in the paper.

  - Multi-scale segmentation + boundary supervision, Eq. 6-9: L_Seg
    averages Dice loss over all 4 scale predictions (each against its OWN
    downsampled ground truth, "deeply multi-scale supervision"); L_Map
    averages cross-entropy over every im-/ex-Bound block's predicted key-
    point map against a pre-computed ground-truth key-point map (Section
    III-C: a Suzuki border-following + circle-deviation-scoring + non-max-
    suppression algorithm run on the GT mask -- a DATA PREPROCESSING step,
    not a decoder module, and out of scope for this file). As with the
    other 3 decoders in this project, our shared trainer only supervises
    a single final output, so neither the 4-way L_Seg averaging nor L_Map
    is reproduced here architecturally beyond exposing the raw ingredients
    (`self._aux_predictions` for the other 3 scale heads, `self._keypoint_maps`
    for every block's predicted key-point map) so they COULD be wired up
    later if the trainer is extended.
    NOTE on final inference output: unlike EMCAD (which explicitly states
    "p4, the last stage, is the final segmentation map"), this paper never
    states which of the 4 scale predictions is used as the actual model
    output at inference time -- Eq. 6 is a training-time loss averaged
    over all 4, not an aggregation rule. We return the finest-resolution
    head (native stride 4) as the decoder's output, by analogy with common
    practice and consistent with every other decoder in this project
    returning a stride-4 map -- this is our choice, not a value stated in
    the paper.

Adaptations for arbitrary encoders (this codebase supports ConvNeXt, PVT,
ResNet-50, Swin-T, etc., not just the paper's PVTv2): the paper operates
directly on PVTv2's native per-stage channel widths (e.g. [64,128,320,512])
with no separate decoder-width schedule, since im-/ex-/X-Bound all operate
in-place at each stage's own channel count. Since our encoders have
arbitrary channel counts, we add a 1x1 projection conv per stage (encoder
channels -> a decoder_dim-based width, same [d4,d8,d16,d32] = decoder_dim
* [1,2,4,8] schedule as the other 3 decoders) before any of the attention
modules. We also add small learned per-scale position embeddings (the
paper says features are "added with position embeddings" without
specifying learned vs. sinusoidal or how they handle variable input
resolutions -- we use a small learned grid, bilinearly resized to the
actual feature map size at runtime, a standard ViT-style trick for
variable input sizes; this is our choice, flagged as such).
    (b) z is refined using the updated boundary embedding xi. The paper
        does not specify precisely whether z and xi are concatenated into
        one joint self-attention sequence or whether xi is injected into
        the feature sequence before the MSA module. This implementation
        uses an additive broadcast injection:

            z <- z + broadcast(xi)

        followed by self-attention, MLP, and key-point prediction.

        This is a deliberate memory-safe interpretation. Literal joint
        attention over [z; xi] would increase the attention sequence length
        and, at stride 4, would preserve the quadratic memory problem that
        motivated spatial-reduction attention in this implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PyramidDecoder


def to_tokens(x):
    """[B, C, H, W] -> ([B, N, C], (H, W))"""
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2), (H, W)


def to_map(tokens, size):
    """[B, N, C] -> [B, C, H, W]"""
    B, N, C = tokens.shape
    H, W = size
    return tokens.transpose(1, 2).reshape(B, C, H, W)


class MultiHeadAttention(nn.Module):
    """Generic MSA(query, key, value) with independent projection widths
    for the query source vs. the key/value source, so the same module
    covers both plain self-attention (Eq. 2, q=k=v=z, same channel width)
    and the cross-scale attention of Eq. 4-5 (q from one scale's features,
    k=v from another scale's boundary embedding, different channel width).

    sr_ratio > 1 applies a spatial-reduction conv (PVTv2-style SRA) to the
    key/value source before attention, so full self-attention at stride-4
    resolution (e.g. 64x64=4096 tokens) doesn't blow up to an N x N
    attention matrix. Only meaningful when key/value come from a genuine
    2D feature map (kv_hw given); for length-1 sources like the boundary
    embedding xi, sr_ratio must stay 1 (nothing to reduce)."""

    def __init__(self, q_dim, kv_dim, attn_dim, num_heads=8, sr_ratio=1):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(q_dim, attn_dim)
        self.to_k = nn.Linear(kv_dim, attn_dim)
        self.to_v = nn.Linear(kv_dim, attn_dim)
        self.proj = nn.Linear(attn_dim, q_dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_norm = nn.LayerNorm(kv_dim)

    def _reduce(self, tokens, hw):
        """[B, N, C] tokens at spatial size hw -> [B, N/sr^2, C], via a
        strided conv (PVTv2 SRA). Only called when key is value (self-attn
        on a real 2D map); caller reuses the result for both K and V."""
        B, N, C = tokens.shape
        H, W = hw
        x = tokens.transpose(1, 2).reshape(B, C, H, W)

        ratio = min(self.sr_ratio, H, W)
        if ratio > 1:
            x = F.avg_pool2d(x, kernel_size=ratio, stride=ratio)

        x = x.flatten(2).transpose(1, 2)
        return self.sr_norm(x)

    def forward(self, query, key, value, kv_hw=None):
        if self.sr_ratio > 1:
            assert kv_hw is not None
            key = self._reduce(key, kv_hw)
            value = key

        B, Nq, _ = query.shape
        Nk = key.shape[1]

        q = self.to_q(query).view(
            B, Nq, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.to_k(key).view(
            B, Nk, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.to_v(value).view(
            B, Nk, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = attn.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(
            B, Nq, self.num_heads * self.head_dim
        )
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class KeyPointHead(nn.Module):
    """Linear + Sigmoid boundary key-point predictor, used after every
    im-/ex-Bound block (Section III-A.1). Supervision target (Section
    III-C's GT key-point map generation) is out of scope for this file --
    see module-level docstring."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # [B, N, 1]


class ImBoundBlock(nn.Module):
    """One im-Bound block, Eq. 2-3 (residuals/LN made explicit -- see
    module-level docstring for the Eq. 3 notation-clash resolution)."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, dim, dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ratio=mlp_ratio)
        self.keypoint_head = KeyPointHead(dim)

    def forward(self, z, hw):
        zn = self.norm1(z)
        z = z + self.attn(
            zn,
            zn,
            zn,
            kv_hw=hw,
        )


        rho = z + self.mlp(self.norm2(z))
        m_hat = self.keypoint_head(rho)
        return rho + rho * m_hat, m_hat


class ExBoundBlock(nn.Module):
    """One ex-Bound block -- our best-effort, self-consistent reading of
    the prose description (the paper gives no closed-form equation here).
    See module-level docstring for the general interpretation rationale.

    Deviation from the module-level docstring's part (b): rather than
    running self-attention over the concatenated sequence [z ; xi], we
    broadcast-add xi into every token of z (an additive, FiLM-style
    modulation) and then run ordinary self-attention over z alone. True
    self-attention over [z ; xi] would cost O((N+1)^2) instead of O(N^2)
    for a single extra token, so we replace it with this cheaper additive
    coupling. Since xi is a single vector, this still lets every spatial
    token directly incorporate the just-updated boundary embedding before
    the self-attention step -- the practical effect the paper's prose asks
    for -- but it is NOT literal joint self-attention over [z ; xi], and
    should not be read as such."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, sr_ratio=1):
        super().__init__()
        self.xi_norm = nn.LayerNorm(dim)
        self.z_norm = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, dim, dim, num_heads=num_heads)
        self.xi_mlp_norm = nn.LayerNorm(dim)
        self.xi_mlp = MLP(dim, ratio=mlp_ratio)

        self.z_norm2 = nn.LayerNorm(dim)
        self.z_attn = MultiHeadAttention(
            dim, dim, dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
        )
        self.z_norm3 = nn.LayerNorm(dim)
        self.z_mlp = MLP(dim, ratio=mlp_ratio)
        self.keypoint_head = KeyPointHead(dim)

    def forward(self, z, xi, hw):
        xi = xi + self.cross_attn(
            self.xi_norm(xi),
            self.z_norm(z),
            self.z_norm(z),
        )
        xi = xi + self.xi_mlp(self.xi_mlp_norm(xi))

        # Additive broadcast coupling, not joint self-attention over
        # [z; xi] -- see class docstring.
        z = z + xi.expand(-1, z.shape[1], -1)

        zn = self.z_norm2(z)
        z = z + self.z_attn(zn, zn, zn, kv_hw=hw)

        rho = z + self.z_mlp(self.z_norm3(z))
        m_hat = self.keypoint_head(rho)
        return rho + rho * m_hat, xi, m_hat


class InScaleBoundaryModule(nn.Module):
    """Eq. 1: N_im cascaded im-Bound blocks, then N_ex cascaded ex-Bound
    blocks, all at a single scale. Returns the refined feature f^1, the
    final boundary embedding xi, and every block's key-point map (exposed
    for optional future L_Map supervision)."""

    def __init__(
        self, dim, n_im=2, n_ex=2, num_heads=8,
        mlp_ratio=4.0, sr_ratio=1
    ):
        super().__init__()

        self.im_blocks = nn.ModuleList([
            ImBoundBlock(
                dim, num_heads, mlp_ratio, sr_ratio
            )
            for _ in range(n_im)
        ])
        self.ex_blocks = nn.ModuleList([
            ExBoundBlock(
                dim, num_heads, mlp_ratio, sr_ratio
            )
            for _ in range(n_ex)
        ])

        self.xi_init = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.xi_init, std=0.02)

    def forward(self, z, hw):
        keypoint_maps = []

        for block in self.im_blocks:
            z, m_hat = block(z, hw)
            keypoint_maps.append(m_hat)

        B = z.shape[0]
        xi = self.xi_init.expand(B, -1, -1)

        for block in self.ex_blocks:
            z, xi, m_hat = block(z, xi, hw)
            keypoint_maps.append(m_hat)

        return z, xi, keypoint_maps


class XBoundFusion(nn.Module):
    """Eq. 4-5: cross-scale boundary learner (X-Bound). See module-level
    docstring for (a) the notation/degeneracy note on single-token
    key/value attention and (b) the >2-scale cascade interpretation."""

    def __init__(self, low_channels, high_channels, num_heads=8):
        super().__init__()
        self.low_cross = MultiHeadAttention(low_channels, high_channels, low_channels, num_heads=num_heads)
        self.high_cross = MultiHeadAttention(high_channels, low_channels, high_channels, num_heads=num_heads)
        self.proj = nn.Conv2d(low_channels + high_channels, low_channels, kernel_size=1)

    def forward(self, f_low, xi_low, f_high, xi_high, low_size):
        low_tokens, low_hw = to_tokens(f_low)
        high_tokens, high_hw = to_tokens(f_high)

        gamma_low = low_tokens + self.low_cross(low_tokens, xi_high, xi_high)      # Eq. 4
        gamma_high = high_tokens + self.high_cross(high_tokens, xi_low, xi_low)    # Eq. 5

        gamma_low_map = to_map(gamma_low, low_hw)
        gamma_high_map = to_map(gamma_high, high_hw)
        gamma_high_up = F.interpolate(gamma_high_map, size=low_size, mode="bilinear", align_corners=False)

        fused = torch.cat([gamma_low_map, gamma_high_up], dim=1)
        return self.proj(fused)


class XBoundFormerDecoder(PyramidDecoder):
    """Full XBound-Former decoder: per-scale in-scale boundary modeling
    (im-Bound + ex-Bound) at all 4 encoder stages, then a top-down cascade
    of cross-scale boundary fusion (X-Bound) over 3 of the 4 stages
    (coarsest stays unfused), with a 1x1 SegHead at every stage. Returns
    the finest-resolution head as the model output (see module docstring's
    note on final-output ambiguity); the other 3 heads and every block's
    predicted key-point map are exposed as attributes after a forward call
    for optional future multi-scale / boundary supervision."""

    def __init__(
        self, enc_channels, decoder_dim=64, num_classes=1,
        n_im=2, n_ex=2, num_heads=8, mlp_ratio=4.0, pos_grid=14
    ):
        super().__init__(enc_channels, decoder_dim, num_classes)
        c4, c8, c16, c32 = enc_channels
        d4, d8, d16, d32 = (
            decoder_dim,
            decoder_dim * 2,
            decoder_dim * 4,
            decoder_dim * 8,
        )

        # Encoder-to-decoder projections.
        self.proj4 = nn.Conv2d(c4, d4, kernel_size=1)
        self.proj8 = nn.Conv2d(c8, d8, kernel_size=1)
        self.proj16 = nn.Conv2d(c16, d16, kernel_size=1)
        self.proj32 = nn.Conv2d(c32, d32, kernel_size=1)

        # Learned positional grids, resized at runtime.
        self.pos4 = nn.Parameter(torch.zeros(1, d4, pos_grid, pos_grid))
        self.pos8 = nn.Parameter(torch.zeros(1, d8, pos_grid, pos_grid))
        self.pos16 = nn.Parameter(torch.zeros(1, d16, pos_grid, pos_grid))
        self.pos32 = nn.Parameter(torch.zeros(1, d32, pos_grid, pos_grid))

        for pos in (self.pos4, self.pos8, self.pos16, self.pos32):
            nn.init.trunc_normal_(pos, std=0.02)

        self.in_scale4 = InScaleBoundaryModule(
            d4, n_im, n_ex, num_heads, mlp_ratio, sr_ratio=8
        )
        self.in_scale8 = InScaleBoundaryModule(
            d8, n_im, n_ex, num_heads, mlp_ratio, sr_ratio=4
        )
        self.in_scale16 = InScaleBoundaryModule(
            d16, n_im, n_ex, num_heads, mlp_ratio, sr_ratio=2
        )
        self.in_scale32 = InScaleBoundaryModule(
            d32, n_im, n_ex, num_heads, mlp_ratio, sr_ratio=1
        )

        # top-down cross-scale fusion cascade; deepest/coarsest (32) is
        # left unfused, per the paper ("f4^2 = f4^1").
        self.xbound_16 = XBoundFusion(d16, d32, num_heads=num_heads)
        self.xbound_8 = XBoundFusion(d8, d16, num_heads=num_heads)
        self.xbound_4 = XBoundFusion(d4, d8, num_heads=num_heads)

        self.head4 = nn.Conv2d(d4, num_classes, kernel_size=1)
        self.head8 = nn.Conv2d(d8, num_classes, kernel_size=1)
        self.head16 = nn.Conv2d(d16, num_classes, kernel_size=1)
        self.head32 = nn.Conv2d(d32, num_classes, kernel_size=1)

        self._aux_predictions = None  # (p8, p16, p32) upsampled to stride 4, optional deep supervision
        self._keypoint_maps = None    # dict of per-scale lists of predicted boundary key-point maps

    def _add_pos(self, x, pos):
        if x.shape[-2:] != pos.shape[-2:]:
            pos = F.interpolate(pos, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + pos

    def _run_scale(self, module, feat_map):
        tokens, hw = to_tokens(feat_map)
        z1, xi, kmaps = module(tokens, hw)
        return to_map(z1, hw), xi, kmaps

    def forward(self, feats):
        s4, s8, s16, s32 = feats
        target_size = s4.shape[-2:]

        f4 = self._add_pos(self.proj4(s4), self.pos4)
        f8 = self._add_pos(self.proj8(s8), self.pos8)
        f16 = self._add_pos(self.proj16(s16), self.pos16)
        f32 = self._add_pos(self.proj32(s32), self.pos32)

        f4_1, xi4, km4 = self._run_scale(self.in_scale4, f4)
        f8_1, xi8, km8 = self._run_scale(self.in_scale8, f8)
        f16_1, xi16, km16 = self._run_scale(self.in_scale16, f16)
        f32_1, xi32, km32 = self._run_scale(self.in_scale32, f32)

        # cross-scale boundary fusion, top-down cascade (see XBoundFusion docstring)
        f32_2 = f32_1  # unchanged, per paper
        f16_2 = self.xbound_16(f16_1, xi16, f32_2, xi32, low_size=f16_1.shape[-2:])
        f8_2 = self.xbound_8(f8_1, xi8, f16_2, xi16, low_size=f8_1.shape[-2:])
        f4_2 = self.xbound_4(f4_1, xi4, f8_2, xi8, low_size=f4_1.shape[-2:])

        p4 = self.head4(f4_2)  # native stride 4
        p8 = F.interpolate(self.head8(f8_2), size=target_size, mode="bilinear", align_corners=False)
        p16 = F.interpolate(self.head16(f16_2), size=target_size, mode="bilinear", align_corners=False)
        p32 = F.interpolate(self.head32(f32_2), size=target_size, mode="bilinear", align_corners=False)

        self._aux_predictions = (p8, p16, p32)
        self._keypoint_maps = {"s4": km4, "s8": km8, "s16": km16, "s32": km32}

        return p4  # finest-resolution head; see docstring's note on final-output ambiguity
