"""Heatmap visualisations of token- and layer-level relevance."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _clean_tokens(tokens: list[str]) -> list[str]:
    return [t.replace("\u0120", " ").replace("\u010a", "\\n") for t in tokens]


def _to_numpy(x) -> np.ndarray:
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def save_token_heatmap(tokens, relevance, out_path: Path, title: str = "") -> None:
    tokens = _clean_tokens(tokens)
    r = _to_numpy(relevance)
    vmax = float(np.max(np.abs(r))) + 1e-9

    fig, ax = plt.subplots(figsize=(max(10.0, len(tokens) * 0.2), 2.0))
    ax.imshow(r[None, :], cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_layer_heatmap(tokens, layer_relevance, out_path: Path, title: str = "") -> None:
    tokens = _clean_tokens(tokens)
    r = _to_numpy(layer_relevance)
    # per-layer (per-row) normalization so each layer's pattern is visible
    # — same convention as lxt's official latent-feature-attribution recipe.
    row_max = np.maximum(np.abs(r).max(axis=1, keepdims=True), 1e-9)
    r_norm = r / row_max

    fig, ax = plt.subplots(
        figsize=(max(10.0, len(tokens) * 0.2), max(4.0, r.shape[0] * 0.25))
    )
    ax.imshow(r_norm, cmap="bwr", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=6)
    ax.set_yticks(range(r.shape[0]))
    ax.set_yticklabels([f"L{i}" for i in range(r.shape[0])], fontsize=7)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Transformer layers")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
