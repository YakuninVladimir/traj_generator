from __future__ import annotations

import torch.nn as nn


def zero_last_linear(module: nn.Module) -> None:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


def mlp(
    sizes: list[int],
    activation: type[nn.Module] = nn.SiLU,
    norm: bool = False,
    dropout: float = 0.0,
    zero_last: bool = False,
) -> nn.Module:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        in_dim = sizes[i]
        out_dim = sizes[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    net = nn.Sequential(*layers)
    if zero_last:
        zero_last_linear(net)
    return net

