from __future__ import annotations

import torch
from torch import nn


class SimpleCNNBaseline(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.net(x).flatten(1)
        return self.classifier(x)


class HLFE(nn.Module):
    def __init__(self, d_model: int = 128, split_bin: int = 129, freq_bins: int = 257) -> None:
        super().__init__()
        self.split_bin = split_bin
        self.freq_bins = freq_bins
        self.low_proj = nn.Sequential(nn.Linear(split_bin, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.high_proj = nn.Sequential(nn.Linear(freq_bins - split_bin, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.hl_logits = nn.Parameter(torch.tensor([0.3, 0.0], dtype=torch.float32))
        self.register_buffer("hl_logits_init", self.hl_logits.detach().clone())
        self.hl_temperature = 0.7
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_low = x[:, :, : self.split_bin]
        x_high = x[:, :, self.split_bin :]
        y_low = self.low_proj(x_low)
        y_high = self.high_proj(x_high)
        weights = self.get_hl_weights()
        y = weights[0] * y_low + weights[1] * y_high
        return self.out_proj(y)

    def get_hl_weights(self) -> torch.Tensor:
        return torch.softmax(self.hl_logits / self.hl_temperature, dim=0)

    def get_hl_logits(self) -> torch.Tensor:
        return self.hl_logits

    def get_hl_logits_init(self) -> torch.Tensor:
        return self.hl_logits_init

    def get_hl_logits_grad_norm(self) -> float | None:
        if self.hl_logits.grad is None:
            return None
        return float(self.hl_logits.grad.detach().norm().cpu().item())

    def hl_weights_value(self) -> tuple[float, float]:
        weights = self.get_hl_weights().detach().cpu()
        return float(weights[0].item()), float(weights[1].item())


class _FallbackSequenceEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = max(d_model // 2, 1)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.proj = nn.Linear(hidden * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.proj(self.norm(y))


class MambaEmotionModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        freq_bins: int = 257,
        d_model: int = 128,
        use_hlfe: bool = False,
        num_layers: int = 2,
        dropout: float = 0.1,
        force_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.use_hlfe = use_hlfe
        self.uses_mamba_ssm = False
        self.model_backend = "bigru_fallback"
        if use_hlfe:
            self.input = HLFE(d_model=d_model, split_bin=129, freq_bins=freq_bins)
        else:
            self.input = nn.Sequential(nn.Linear(freq_bins, d_model), nn.GELU(), nn.LayerNorm(d_model))

        if force_backend not in (None, "real_mamba_ssm", "bigru_fallback"):
            raise ValueError(f"Unknown Mamba backend: {force_backend}")

        if force_backend != "bigru_fallback":
            try:
                from mamba_ssm import Mamba

                self.encoder = nn.Sequential(
                    *[Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)]
                )
                self.uses_mamba_ssm = True
                self.model_backend = "real_mamba_ssm"
            except Exception:
                if force_backend == "real_mamba_ssm":
                    raise
                self.encoder = _FallbackSequenceEncoder(d_model=d_model, num_layers=num_layers, dropout=dropout)
        else:
            self.encoder = _FallbackSequenceEncoder(d_model=d_model, num_layers=num_layers, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(self.dropout(x))

    def get_hl_weights(self) -> torch.Tensor | None:
        if self.use_hlfe and hasattr(self.input, "get_hl_weights"):
            return self.input.get_hl_weights()
        return None

    def get_hl_logits(self) -> torch.Tensor | None:
        if self.use_hlfe and hasattr(self.input, "get_hl_logits"):
            return self.input.get_hl_logits()
        return None

    def get_hl_logits_init(self) -> torch.Tensor | None:
        if self.use_hlfe and hasattr(self.input, "get_hl_logits_init"):
            return self.input.get_hl_logits_init()
        return None

    def get_hl_logits_grad_norm(self) -> float | None:
        if self.use_hlfe and hasattr(self.input, "get_hl_logits_grad_norm"):
            return self.input.get_hl_logits_grad_norm()
        return None

    def get_hl_logits_delta_norm(self) -> float | None:
        logits = self.get_hl_logits()
        logits_init = self.get_hl_logits_init()
        if logits is None or logits_init is None:
            return None
        return float(torch.norm(logits.detach().cpu() - logits_init.detach().cpu()).item())

    def hlfe_weights_value(self) -> tuple[float, float] | None:
        if self.use_hlfe and hasattr(self.input, "hl_weights_value"):
            return self.input.hl_weights_value()
        return None

    def hlfe_alpha_value(self) -> float | None:
        weights = self.hlfe_weights_value()
        if weights is not None:
            return weights[0]
        return None
