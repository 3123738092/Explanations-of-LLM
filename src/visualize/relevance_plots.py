"""Plot token, layer, attention-head, and parameter relevance figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _clean_tokens(tokens: list[str]) -> list[str]:
    return [t.replace("\u0120", " ").replace("\u010a", "\\n") for t in tokens]


def _to_numpy(x) -> np.ndarray:
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _minmax01(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


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


def save_attention_head_heatmap(
    head_relevance,
    out_path: Path,
    title: str = "Attention-head parameter relevance",
) -> None:
    """Visualise component-level absolute parameter relevance as [heads, layers]."""
    r = _minmax01(_to_numpy(head_relevance).astype(float))
    n_layers, n_heads = r.shape

    fig, ax = plt.subplots(
        figsize=(max(8.0, n_layers * 0.55), max(4.0, n_heads * 0.35))
    )
    im = ax.imshow(r.T, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"H{i}" for i in range(n_heads)], fontsize=8)
    ax.set_xlabel("Transformer layers")
    ax.set_ylabel("Attention heads")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Min-max normalised |parameter relevance|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_layer_parameter_trend(
    layer_component_abs,
    component_names,
    out_path: Path,
    title: str = "Layer-wise parameter relevance",
) -> None:
    """Heatmap [Layers x Modules] with column-wise min-max + raw values in cells.

    Each module column is independently min-max scaled to [0, 1] so colour
    reveals *within-module* peak layers without one module drowning the others.
    Raw absolute sums are printed in each cell for cross-module magnitude.
    """
    if isinstance(layer_component_abs, tuple):
        r = _to_numpy(layer_component_abs[0]).astype(float)
    else:
        r = _to_numpy(layer_component_abs).astype(float)

    L, C = r.shape
    col_min = r.min(axis=0, keepdims=True)
    col_max = np.maximum(r.max(axis=0, keepdims=True), 1e-12)
    norm = (r - col_min) / (col_max - col_min)

    fig, ax = plt.subplots(figsize=(max(4.5, C * 1.3), max(4.0, L * 0.30)))
    im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(C))
    ax.set_xticklabels(component_names)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"L{i}" for i in range(L)], fontsize=8)
    ax.set_xlabel("Module")
    ax.set_ylabel("Transformer layer")

    for i in range(L):
        for j in range(C):
            ax.text(j, i, f"{r[i, j]:.0f}", ha="center", va="center",
                    color="white" if norm[i, j] < 0.5 else "black", fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("|relevance| sum, column-wise min-max")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_layer_param_line(
    layer_component_abs,
    component_names,
    out_path: Path,
    title: str = "Layer-wise parameter relevance (line chart)",
) -> None:
    """Per-component self-normalised line chart.

    Each component curve is independently divided by its own max, so all three
    span [0, 1] and their layer-wise *shapes* are directly comparable.
    """
    r = _to_numpy(layer_component_abs).astype(float)
    L, C = r.shape
    layers = np.arange(L)
    colors = {"attention": "#3366cc", "mlp": "#dc3912", "layernorm": "#109618"}

    fig, ax = plt.subplots(figsize=(max(8.0, L * 0.6), 4.5))
    for idx, name in enumerate(component_names):
        comp_max = float(np.max(r[:, idx]))
        if comp_max < 1e-12:
            continue
        ax.plot(layers, r[:, idx] / comp_max, marker="o", linewidth=2.0,
                label=name, color=colors.get(name))
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers], fontsize=8)
    ax.set_xlabel("Transformer layers")
    ax.set_ylabel("Per-component self-normalised |relevance|")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=min(3, C))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
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
