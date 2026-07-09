"""
This code provides Gaussian diffusion process on N point coordinates (x,y) in 2D space.

The denoise predicts noise, ... . Training minimizes an MSE between the predicted noise and the true noise.

Later we will add more here.
"""


import torch
import torch.nn.functional as F

from ..utils import calc_boundary_att, soft_dice_loss, calc_curvature


class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu",
                 use_uncertainty_weighting=True):
        self.timesteps = timesteps
        self.use_uncertainty_weighting = use_uncertainty_weighting  # <-- Flag: Uncertainty Weighting an/aus

        betas = torch.linspace(start=beta_start, end=beta_end, steps=timesteps)
        alphas = 1.0 - betas

        self.betas = betas.to(device)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # log_sigma Parameter nur anlegen, wenn Uncertainty Weighting aktiv ist.
        if self.use_uncertainty_weighting:
            self.log_sigma_uniformity = torch.nn.Parameter(torch.zeros(1, device=device))
            self.log_sigma_boundary   = torch.nn.Parameter(torch.zeros(1, device=device))
            self.log_sigma_dice       = torch.nn.Parameter(torch.zeros(1, device=device))
        else:
            self.log_sigma_uniformity = None
            self.log_sigma_boundary   = None
            self.log_sigma_dice       = None


    @torch.no_grad()
    def q_sample(self, x_0, timestep, noise=None):                                              # forward pass
        if noise is None:
            noise = torch.randn_like(input=x_0)
            noise = torch.clamp(noise, -3.0, 3.0)
    
        device = x_0.device
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(device)[timestep]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[timestep]
    
        # (B,) -> (B, 1, 1, ..., 1), passend zur Anzahl Dims von x_0
        shape = [x_0.shape[0]] + [1] * (x_0.dim() - 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(shape)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(shape)
    
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t

    @torch.no_grad()
    def p_sample(self, denoiser, x_t, t, cond):                                                 # backwards pass
        device = x_t.device
        b_size = x_t.shape[0]
    
        betas_t = self.betas.to(device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t]
        alphas_t = 1.0 - betas_t
    
        shape = [b_size] + [1] * (x_t.dim() - 1)
        betas_t = betas_t.view(shape)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(shape)
        alphas_t = alphas_t.view(shape)
    
        predicted_noise = denoiser(x=x_t, t=t, cond=cond)
    
        model_mean = (1.0 / torch.sqrt(alphas_t)) * (
            x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise
        )
    
        if (t == 0).all():
            return torch.clamp(model_mean, -1.2, 1.2) # sonst bleiben die möglicherweise am rand stecken einfach
    
        noise = torch.randn_like(x_t)
    
        sigma_t = torch.sqrt(betas_t)
        x_t_minus_1 = model_mean + sigma_t * noise
    
        return torch.clamp(x_t_minus_1, -1.0, 1.0)

    @torch.no_grad()
    def sample_ddpm(self, denoiser, encoder, images, shape, device):
        feats = encoder.extract(images)

        x_t = torch.randn(shape, device=device)
        x_t = torch.clamp(x_t, -3.0, 3.0)
        b_size = shape[0]

        for i in reversed(range(self.timesteps)):
            t = torch.full((b_size,), i, device=device, dtype=torch.long)
            cond = encoder.fuse(feats, t)
            x_t = self.p_sample(denoiser, x_t, t, cond)

        return x_t

    @torch.no_grad()
    def sample(self, denoiser, encoder, images, shape, device, steps=50, cfg_scale=3.0):
        """
        DDIM Sampler mit Classifier-Free Guidance (CFG).
        cfg_scale: Stärke der Bildführung (1.0 = kein CFG-Effekt, >1.0 = verstärkt).
        """
        feats = encoder.extract(images)

        x_t = torch.randn(shape, device=device)
        x_t = torch.clamp(x_t, -3.0, 3.0)
        b_size = shape[0]

        times = torch.linspace(-1, self.timesteps - 1, steps + 1, dtype=torch.long, device=device)

        for i in reversed(range(1, len(times))):
            t_idx = times[i]
            t_prev_idx = times[i - 1]
            t = torch.full((b_size,), t_idx, device=device, dtype=torch.long)

            cond = encoder.fuse(feats, t)

            if cfg_scale > 1.0:
                # --- CFG BATCHING ---
                # Alles auf 2B verdoppeln: erste Hälfte = conditioned, zweite Hälfte = unconditioned
                x_t_double = torch.cat([x_t, x_t], dim=0)
                t_double = torch.cat([t, t], dim=0)
                cond_double = [torch.cat([c, c], dim=0) for c in cond]

                # Maske: erste b_size Zeilen behalten cond (False = nicht droppen),
                # zweite b_size Zeilen bekommen cond genullt (True = droppen)
                drop_mask = torch.cat([
                    torch.zeros(b_size, dtype=torch.bool, device=device),
                    torch.ones(b_size, dtype=torch.bool, device=device),
                ], dim=0)

                pred_noise_both = denoiser(x_t_double, t_double, cond_double, drop_mask=drop_mask)

                pred_noise_cond, pred_noise_uncond = torch.chunk(pred_noise_both, 2, dim=0)

                # CFG Extrapolations-Formel
                pred_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                # Ohne CFG: normaler Single-Batch-Pfad, cond immer aktiv
                drop_mask = torch.zeros(b_size, dtype=torch.bool, device=device)
                pred_noise = denoiser(x_t, t, cond, drop_mask=drop_mask)

            # Standard DDIM-Mathematik (unverändert)
            alpha_bar_t = self.alphas_cumprod[t_idx].view(-1, 1, 1)
            if t_prev_idx >= 0:
                alpha_bar_prev = self.alphas_cumprod[t_prev_idx].view(-1, 1, 1)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            pred_x_0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            pred_x_0 = torch.clamp(pred_x_0, -3.0, 3.0)

            direction_pointing_to_xt = torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
            x_t = torch.sqrt(alpha_bar_prev) * pred_x_0 + direction_pointing_to_xt

        return x_t


    def get_x0_from_noise(self, predicted_noise, t, x_t):
        """Rechnet aus dem vorhergesagten Rauschen die zugehörige x0-Schätzung zurück."""
        device = x_t.device
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(device)[t]

        shape = [x_t.shape[0]] + [1] * (x_t.dim() - 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(shape)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(shape)

        return (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t.clamp(min=1e-8)


    import torch
    import torch.nn.functional as F


    def training_loss(self, predicted_noise, true_noise, x_t, x_0, t,
                   masks=None,
                   lamda_mse=1.0,
                   lambda_uniformity_max=0.1,
                   lamda_boundary_max=1.0,
                   lamda_dice_max=1.0,
                   boundary_gate_ratio=0.4):
        """
        Uncertainty Weighting (Kendall et al.) wird NUR auf die drei Hilfs-Losses
        angewendet (uniformity, boundary, dice). loss_mse bleibt bewusst fix
        gewichtet als Anker, damit das System nicht trivial alle sigma hochtreibt
        um alle Losses zu ignorieren.

        Über self.use_uncertainty_weighting (Konstruktor-Flag) lässt sich die
        gesamte Uncertainty-Weighting-Mechanik abschalten: dann werden die
        Hilfs-Losses einfach direkt mit ihren lambda_max-Gewichten skaliert,
        ohne exp(-2*log_sigma)-Skalierung und ohne +log_sigma-Regularisierung.

        Boundary und Dice werden PRO SAMPLE geschaltet: nur Samples mit
        t_ratio < boundary_gate_ratio tragen bei. Der teure Forward-Pass
        (calc_boundary_att / soft_dice_loss) wird nur auf der aktiven Teilmenge
        ausgeführt, nicht auf dem vollen Batch mit anschließender Maskierung.
        """
        t_ratio = t.float() / self.timesteps  # [B]
    
        # 1. MSE Loss (Kern-Trainingssignal, bleibt unangetastet / kein sigma)
        loss_mse = F.mse_loss(predicted_noise, true_noise)
    
        # 2. Predicted x0
        predicted_x0 = self.get_x0_from_noise(predicted_noise, t, x_t)
    
        # 3. Uniformity Loss (aktiv gegen Ende der Diffusion, t_ratio klein)
        #    -- curvature-gewichtet: an scharfen GT-Ecken wird die Uniformity-Strafe
        #       abgeschwächt (exp(-2*curvature) -> klein), an glatten Segmenten voll
        #       wirksam. Nutzt dieselbe normalisierte Curvature wie calc_boundary_att
        #       (spatial_att, [0,1] pro Sample), aber gegensätzliches Vorzeichen der
        #       Wirkung: dort verstärkt sie, hier schwächt sie ab; kein Zeit-Gate und
        #       kein Sockel-Term hier, da die Zeit-Steuerung separat über
        #       weight_uniformity läuft.
        weight_uniformity = lambda_uniformity_max * (1.0 - t_ratio)
        uniformity_active = weight_uniformity.max() > 0
        if uniformity_active:
            nxt = torch.roll(predicted_x0, shifts=-1, dims=1)
            dists = torch.norm(predicted_x0 - nxt, dim=-1)  # [B, N] -- Kantenlängen
    
            gt_curvature = calc_curvature(x_0).squeeze(-1)   # [B, N], normalisiert [0,1]
            uniformity_weights = torch.exp(-2.0 * gt_curvature)  # [B, N]
    
            mean_dist = dists.mean(dim=1, keepdim=True)                      # [B, 1]
            squared_diffs = (dists - mean_dist) ** 2                         # [B, N]
            loss_uniformity = (squared_diffs * uniformity_weights).mean()
    
            scaled_uniformity = weight_uniformity.mean() * loss_uniformity

            if self.use_uncertainty_weighting:
                precision_uniformity = torch.exp(-2 * self.log_sigma_uniformity)
                term_uniformity = (precision_uniformity * scaled_uniformity
                                    + self.log_sigma_uniformity).squeeze()
            else:
                term_uniformity = scaled_uniformity.squeeze()
        else:
            loss_uniformity = torch.zeros_like(loss_mse)
            term_uniformity = torch.zeros_like(loss_mse)
    
        # -- Per-Sample-Gate für Boundary und Dice, gemeinsame Maske --
        boundary_mask = t_ratio < boundary_gate_ratio  # [B], bool
        n_active = int(boundary_mask.sum().item())
    
        # 4. Boundary Attention Loss (jetzt hart gegatet: nur t_ratio < 0.4)
        if n_active > 0:
            x_0_active = x_0[boundary_mask]                       # [B_active, N, 2]
            t_active = t[boundary_mask]                           # [B_active]
            pred_noise_active = predicted_noise[boundary_mask]    # [B_active, ...]
            true_noise_active = true_noise[boundary_mask]         # [B_active, ...]
    
            boundary_att = calc_boundary_att(x_0_active, t_active, T=self.timesteps, gamma=1.5)
            loss_boundary = (boundary_att * ((pred_noise_active - true_noise_active) ** 2)).mean()

            if self.use_uncertainty_weighting:
                precision_boundary = torch.exp(-2 * self.log_sigma_boundary)
                term_boundary = (precision_boundary * (lamda_boundary_max * loss_boundary)
                                 + self.log_sigma_boundary).squeeze()
            else:
                term_boundary = (lamda_boundary_max * loss_boundary).squeeze()
        else:
            loss_boundary = torch.zeros_like(loss_mse)
            term_boundary = torch.zeros_like(loss_mse)
    
        # 5. Soft-Dice Loss (gleiche Maske wie boundary, da gleicher Threshold)
        dice_active = masks is not None and n_active > 0
        if dice_active:
            predicted_x0_active = predicted_x0[boundary_mask]     # [B_active, N, 2]
            masks_active = masks[boundary_mask]                   # [B_active, H, W]
    
            loss_dice = soft_dice_loss(predicted_x0_active, masks_active)

            if self.use_uncertainty_weighting:
                precision_dice = torch.exp(-2 * self.log_sigma_dice)
                term_dice = (precision_dice * (lamda_dice_max * loss_dice)
                             + self.log_sigma_dice).squeeze()
            else:
                term_dice = (lamda_dice_max * loss_dice).squeeze()
        else:
            loss_dice = torch.zeros_like(loss_mse)
            term_dice = torch.zeros_like(loss_mse)
    
        # 6. Gesamtsumme
        total = (lamda_mse * loss_mse
                 + term_uniformity
                 + term_boundary
                 + term_dice)
    
        return total, {
            "loss_mse": loss_mse.detach(),
            "loss_uniformity": loss_uniformity.detach(),
            "loss_boundary": loss_boundary.detach(),
            "loss_dice": loss_dice.detach(),
            "n_active_boundary_dice": n_active,
            "sigma_uniformity": (torch.exp(self.log_sigma_uniformity).detach()
                                  if self.use_uncertainty_weighting else None),
            "sigma_boundary": (torch.exp(self.log_sigma_boundary).detach()
                                if self.use_uncertainty_weighting else None),
            "sigma_dice": (torch.exp(self.log_sigma_dice).detach()
                            if self.use_uncertainty_weighting else None),
        }

if __name__ == "__main__":
    
    diffusion = GaussianDiffusion(timesteps=1000, beta_start=1e-4, beta_end=0.02)
    x_0 = torch.tensor([[0,1],[2,3],[4,5],[6,7],[8,9]], dtype=torch.float32)  # Example input coordinates
    timestep = int(input("Enter timestep: ")) or 100  # Example timestep
    noise = diffusion.q_sample(x_0, timestep)
    print(noise)