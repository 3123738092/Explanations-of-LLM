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
    title: str = "Layer-wise parameter relevance trend",
) -> None:
    """Dual-panel: total sum (top) and per-parameter mean (bottom)."""
    layers = None

    # Unpack tuple (sum, mean) or fall back to single-array sum
    if isinstance(layer_component_abs, tuple):
        r_sum = _to_numpy(layer_component_abs[0]).astype(float)
        r_mean = _to_numpy(layer_component_abs[1]).astype(float)
    else:
        r_sum = _to_numpy(layer_component_abs).astype(float)
        r_mean = None

    layers = np.arange(r_sum.shape[0])
    colors = {
        "attention": "#3366cc",
        "mlp": "#dc3912",
        "layernorm": "#109618",
    }

    n_panels = 1 if r_mean is None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(8.0, r_sum.shape[0] * 0.6), 4.5 * n_panels))

    if n_panels == 1:
        axes = [axes]

    # Panel 1: total sum (original)
    norm_sum = r_sum / max(float(np.max(r_sum)), 1e-12)
    ax0 = axes[0]
    for idx, name in enumerate(component_names):
        ax0.plot(layers, norm_sum[:, idx], marker="o", linewidth=2.0,
                 label=name, color=colors.get(name))
    ax0.set_ylabel("Normalised total |relevance|")
    ax0.set_ylim(bottom=0.0)
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(frameon=False, ncol=min(3, len(component_names)))
    ax0.set_title(title + "  (total sum)")

    # Panel 2: per-parameter mean (per-component normalization)
    if r_mean is not None:
        ax1 = axes[1]
        for idx, name in enumerate(component_names):
            comp_max = float(np.max(r_mean[:, idx]))
            if comp_max < 1e-12:
                continue
            ax1.plot(layers, r_mean[:, idx] / comp_max, marker="s", linewidth=2.0,
                     label=name, color=colors.get(name))
        ax1.set_xticks(layers)
        ax1.set_xticklabels([f"L{i}" for i in layers], fontsize=8)
        ax1.set_xlabel("Transformer layers")
        ax1.set_ylabel("Per-param |relevance|  (norm. per component)")
        ax1.set_ylim(bottom=0.0)
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(frameon=False, ncol=min(3, len(component_names)))
        ax1.set_title("Per-parameter mean — each component self-normalised")
    else:
        axes[0].set_xticks(layers)
        axes[0].set_xticklabels([f"L{i}" for i in layers], fontsize=8)
        axes[0].set_xlabel("Transformer layers")

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
