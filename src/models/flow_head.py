import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Encodes scalar t ∈ [0,1] into a rich embedding.
    Same idea as positional encodings in transformers.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * (8.0 / half)
        )
        emb = t[:, None] * freqs[None, :]          # (B, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)


class VectorFieldNetwork(nn.Module):
    """
    Predicts the flow matching vector field:
        v_θ(x_t, t, c) ≈ x_1 - x_0

    Inputs:
        x_t:  noisy action at time t,  shape (B, action_dim)
        t:    time in [0,1],            shape (B,)
        cond: VLM conditioning emb,     shape (B, conditioning_dim)

    Output:
        v:    predicted vector field,   shape (B, action_dim)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # time embedding
        self.time_emb = SinusoidalTimeEmbedding(cfg.hidden_dim)

        # project conditioning down to hidden_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.conditioning_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        # input projection: action + time + cond → hidden
        self.input_proj = nn.Linear(cfg.action_dim + cfg.hidden_dim, cfg.hidden_dim)

        # residual MLP trunk
        self.layers = nn.ModuleList([
            ResidualBlock(cfg.hidden_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # output projection → action space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # zero-init output — start by predicting zero vector field
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        x_t: torch.Tensor,   # (B, action_dim)
        t: torch.Tensor,      # (B,)
        cond: torch.Tensor,   # (B, conditioning_dim)
    ) -> torch.Tensor:

        t_emb = self.time_emb(t)                          # (B, hidden_dim)
        c_emb = self.cond_proj(cond)                      # (B, hidden_dim)

        # fuse time and conditioning multiplicatively
        gate = t_emb * c_emb                              # (B, hidden_dim)

        x = torch.cat([x_t, gate], dim=-1)               # (B, action_dim + hidden_dim)
        x = self.input_proj(x)                            # (B, hidden_dim)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)                        # (B, action_dim)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FlowMatchingHead(nn.Module):
    """
    Full flow matching module:
    - Training: CFM loss
    - Inference: Euler ODE solver → action sample
    """

    def __init__(self, cfg, ema_decay: float = 0.999):
        super().__init__()
        self.cfg = cfg
        self.ema_decay = ema_decay

        self.net = VectorFieldNetwork(cfg)

        # EMA model for stable inference
        self.ema_net = copy.deepcopy(self.net)
        for p in self.ema_net.parameters():
            p.requires_grad_(False)

    def cfm_loss(
        self,
        x_1: torch.Tensor,   # (B, action_dim) — clean demo actions
        cond: torch.Tensor,  # (B, conditioning_dim) — VLM embeddings
    ) -> torch.Tensor:
        B = x_1.shape[0]
        device = x_1.device

        # sample noise and time
        x_0 = torch.randn_like(x_1)
        t = torch.rand(B, device=device)

        # straight-line interpolation
        t_exp = t[:, None]                            # (B, 1)
        x_t = (1 - t_exp) * x_0 + t_exp * x_1          # (B, action_dim)

        # target vector field
        target = x_1 - x_0                             # (B, action_dim)

        # predict
        pred = self.net(x_t, t, cond)                  # (B, action_dim)

        return F.mse_loss(pred, target)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,   # (B, conditioning_dim)
        n_steps: int | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Euler ODE integration: noise → action.
        At each step: x_{t+dt} = x_t + dt * v_θ(x_t, t, c)
        """
        n_steps = n_steps or self.cfg.n_euler_steps
        net = self.ema_net if use_ema else self.net
        device = cond.device
        B = cond.shape[0]

        x = torch.randn(B, self.cfg.action_dim, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = net(x, t, cond)
            x = x + dt * v

        return x                                      # (B, action_dim)

    @torch.no_grad()
    def update_ema(self):
        decay = self.ema_decay

        for ema_p, p in zip(self.ema_net.parameters(), self.net.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)