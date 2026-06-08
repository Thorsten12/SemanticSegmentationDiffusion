"""Exponential moving average of model parameters."""

import copy

import torch


class EMA:
    """Maintains a shadow copy of one or more modules' parameters.

    Usage:
        ema = EMA([unet, denoiser], decay=0.999)
        ...                         # after each optimizer.step():
        ema.update([unet, denoiser])
        ema_unet, ema_denoiser = ema.modules
    """

    def __init__(self, modules, decay=0.999, warmup=True):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.modules = [copy.deepcopy(m) for m in modules]
        for m in self.modules:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    def _current_decay(self):
        """Ramp the decay up early so the EMA isn't dominated by random init.

        decay_t = min(decay, (1 + step) / (10 + step)) -> ~0.9 by step ~90,
        converging to `decay` thereafter (standard EMA warmup).
        """
        if not self.warmup:
            return self.decay
        return min(self.decay, (1 + self.step) / (10 + self.step))

    @torch.no_grad()
    def update(self, modules):
        decay = self._current_decay()
        self.step += 1
        for ema_m, m in zip(self.modules, modules):
            for ema_p, p in zip(ema_m.parameters(), m.parameters()):
                ema_p.mul_(decay).add_(p, alpha=1.0 - decay)
            # Keep buffers (e.g. norm stats) in sync.
            for ema_b, b in zip(ema_m.buffers(), m.buffers()):
                ema_b.copy_(b)

    def state_dicts(self):
        return [m.state_dict() for m in self.modules]
