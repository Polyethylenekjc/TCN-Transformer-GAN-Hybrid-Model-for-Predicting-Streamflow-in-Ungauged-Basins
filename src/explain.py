"""Explainability analysis with SHAP and LIME."""

from pathlib import Path as _Path
import sys as _sys

# Ensure project root is on sys.path when running as a script.
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Use Songti (SimSun) for Chinese labels; fall back to default if unavailable.
plt.rcParams["font.sans-serif"] = ["SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.model import StreamflowPredictionModel
from src.dataset import StreamflowDataset
from src.utils.config_loader import ConfigLoader


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("Explain")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _split_indices(n: int, split: str) -> List[int]:
    if n <= 0:
        return []
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    if train_end < 1:
        train_end = 1
    if val_end <= train_end:
        val_end = min(n, train_end + 1)
    if val_end >= n:
        val_end = max(train_end + 1, n - 1)

    indices = list(range(n))
    if split == "train":
        return indices[:train_end]
    if split == "val":
        return indices[train_end:val_end]
    return indices[val_end:]


def _select_indices(indices: List[int], num_samples: int) -> List[int]:
    if num_samples <= 0 or len(indices) <= num_samples:
        return indices
    positions = np.linspace(0, len(indices) - 1, num_samples).astype(int)
    return [indices[i] for i in positions]


def _load_images(dataset: StreamflowDataset, indices: List[int]) -> Tuple[torch.Tensor, List[str]]:
    images = []
    dates = []
    for idx in indices:
        sample = dataset[idx]
        images.append(sample["images"])
        dates.append(sample["date"])
    return torch.stack(images, dim=0), dates


class MeanOutputModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return pred.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)


def _patch_slices(height: int, width: int, patch_size: int) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
    slices = []
    y_positions = list(range(0, height, patch_size))
    x_positions = list(range(0, width, patch_size))
    for y0 in y_positions:
        y1 = min(y0 + patch_size, height)
        for x0 in x_positions:
            x1 = min(x0 + patch_size, width)
            slices.append((y0, y1, x0, x1))
    return slices, len(y_positions), len(x_positions)


def run_shap(
    model: torch.nn.Module,
    background: torch.Tensor,
    explain_images: torch.Tensor,
    out_dir: Path,
    batch_size: int,
    logger: logging.Logger,
    nsamples: int = 50,
) -> np.ndarray:
    try:
        import shap
    except Exception as exc:
        raise RuntimeError("SHAP is not installed. Please install shap.") from exc

    # Always run SHAP on CPU to avoid GPU OOM (GradientExplainer creates
    # nsamples * len(background) intermediate tensors with gradients).
    logger.info("Moving model & data to CPU for SHAP (avoids GPU OOM)...")
    cpu_device = torch.device("cpu")
    model_cpu = model.cpu()
    mean_model = MeanOutputModel(model_cpu)
    mean_model.eval()

    background_cpu = background.cpu()
    explain_cpu = explain_images.cpu()

    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Use a small subset of background to limit memory
    max_bg = min(5, background_cpu.shape[0])
    bg_subset = background_cpu[:max_bg]

    logger.info(f"Building SHAP GradientExplainer (background={max_bg}, nsamples={nsamples})...")
    explainer = shap.GradientExplainer(mean_model, bg_subset)

    logger.info(f"Computing SHAP values for {explain_cpu.shape[0]} samples (one at a time)...")
    shap_values_all = []
    for i in range(explain_cpu.shape[0]):
        sample = explain_cpu[i : i + 1]
        sv = explainer.shap_values(sample, nsamples=nsamples)
        if isinstance(sv, list):
            sv = sv[0]
        shap_values_all.append(sv)
        if (i + 1) % 10 == 0:
            logger.info(f"  SHAP progress: {i + 1}/{explain_cpu.shape[0]}")

    shap_values = np.concatenate(shap_values_all, axis=0)

    # Move model back to original device
    orig_device = explain_images.device
    model.to(orig_device)

    mean_abs = np.mean(np.abs(shap_values), axis=(0, 3, 4))
    per_sample = np.mean(np.abs(shap_values), axis=(3, 4))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shap_per_sample_time_channel.npy", per_sample)
    np.save(out_dir / "shap_global_time_channel.npy", mean_abs)

    # Save CSV summaries
    time_scores = mean_abs.mean(axis=1)
    channel_scores = mean_abs.mean(axis=0)

    np.savetxt(out_dir / "shap_time.csv", time_scores, delimiter=",", header="mean_abs", comments="")
    np.savetxt(out_dir / "shap_channel.csv", channel_scores, delimiter=",", header="mean_abs", comments="")
    np.savetxt(out_dir / "shap_time_channel.csv", mean_abs, delimiter=",", header="c0,c1,c2,c3,c4,c5,c6,c7,c8,c9", comments="")

    # Plots
    plt.figure(figsize=(8, 4))
    plt.plot(time_scores, marker="o")
    plt.title("SHAP Mean | Time")
    plt.xlabel("Time step")
    plt.ylabel("Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(channel_scores)), channel_scores)
    plt.title("SHAP Mean | Channel")
    plt.xlabel("Channel")
    plt.ylabel("Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_channel.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.imshow(mean_abs, aspect="auto", cmap="viridis")
    plt.colorbar(label="Mean |SHAP|")
    plt.title("SHAP Mean | Time x Channel")
    plt.xlabel("Channel")
    plt.ylabel("Time step")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_time_channel.png", dpi=160)
    plt.close()

    return shap_values


def run_lime(
    model: torch.nn.Module,
    background: torch.Tensor,
    explain_images: torch.Tensor,
    patch_size: int,
    lime_samples: int,
    region: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    focus_rows: Tuple[int, int],
    focus_cols: Tuple[int, int],
    out_dir: Path,
    logger: logging.Logger,
    infer_batch_size: int = 2,
) -> np.ndarray:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except Exception as exc:
        raise RuntimeError("LIME is not installed. Please install lime.") from exc

    model.eval()
    with torch.no_grad():
        baseline = background.mean(dim=0)

    t_steps, channels, height, width = explain_images.shape[1:]
    slices, n_rows, n_cols = _patch_slices(height, width, patch_size)
    num_patches = len(slices)
    num_features = t_steps * num_patches

    feature_names = []
    for t in range(t_steps):
        for p in range(num_patches):
            feature_names.append(f"t{t}_p{p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "lime_per_sample_weights.npy"
    mean_path = out_dir / "lime_feature_importance.npy"
    if weights_path.exists() and mean_path.exists():
        logger.info("Found existing LIME results. Skipping analysis and regenerating plots.")
        weights_all = np.load(weights_path)
        mean_abs = np.load(mean_path)
    else:
        train_data = np.random.randint(0, 2, size=(max(200, num_features), num_features)).astype(float)

        explainer = LimeTabularExplainer(
            training_data=train_data,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=False,
        )

        weights_all = []
        instance = np.ones(num_features, dtype=float)

        logger.info("Computing LIME explanations...")
        for idx in range(explain_images.shape[0]):
            single = explain_images[idx : idx + 1]

            def predict_fn_single(mask_matrix: np.ndarray) -> np.ndarray:
                total = mask_matrix.shape[0]
                all_preds = []
                for chunk_start in range(0, total, infer_batch_size):
                    chunk_end = min(chunk_start + infer_batch_size, total)
                    chunk_size = chunk_end - chunk_start
                    chunk_imgs = single.repeat(chunk_size, 1, 1, 1, 1).clone()
                    for i in range(chunk_size):
                        off_features = np.where(mask_matrix[chunk_start + i] < 0.5)[0]
                        for feat in off_features:
                            t = feat // num_patches
                            p = feat % num_patches
                            y0, y1, x0, x1 = slices[p]
                            chunk_imgs[i, t, :, y0:y1, x0:x1] = baseline[t, :, y0:y1, x0:x1]
                    with torch.no_grad():
                        preds = model(chunk_imgs)
                        preds = preds.mean(dim=(1, 2, 3))
                    all_preds.append(preds.cpu().numpy())
                    del chunk_imgs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return np.concatenate(all_preds, axis=0).reshape(-1, 1)

            explanation = explainer.explain_instance(
                instance,
                predict_fn_single,
                num_features=min(20, num_features),
                num_samples=lime_samples,
            )

            weights = np.zeros(num_features, dtype=float)
            for feat_id, weight in explanation.as_map()[1]:
                weights[feat_id] = weight
            weights_all.append(weights)

        weights_all = np.stack(weights_all, axis=0)
        mean_abs = np.mean(np.abs(weights_all), axis=0)

        np.save(weights_path, weights_all)
        np.save(mean_path, mean_abs)

    time_scores = mean_abs.reshape(t_steps, num_patches).mean(axis=1)
    patch_scores = mean_abs.reshape(t_steps, num_patches).mean(axis=0)

    np.savetxt(out_dir / "lime_time.csv", time_scores, delimiter=",", header="mean_abs", comments="")
    np.savetxt(out_dir / "lime_patch_full.csv", patch_scores, delimiter=",", header="mean_abs", comments="")

    # Time plot
    plt.figure(figsize=(8, 4))
    plt.plot(time_scores, marker="o")
    plt.title("LIME Mean | Time")
    plt.xlabel("Time step")
    plt.ylabel("Mean |weight|")
    plt.tight_layout()
    plt.savefig(out_dir / "lime_time.png", dpi=160)
    plt.close()

    # Patch heatmap (re-chunked on the cropped region)
    patch_map_full = patch_scores.reshape(n_rows, n_cols)
    row_start, row_end = focus_rows
    col_start, col_end = focus_cols
    row_start = max(0, min(row_start, image_size[0]))
    row_end = max(row_start + 1, min(row_end, image_size[0]))
    col_start = max(0, min(col_start, image_size[1]))
    col_end = max(col_start + 1, min(col_end, image_size[1]))

    # Expand patch map back to pixel grid, then crop and re-bin
    pixel_map = np.repeat(np.repeat(patch_map_full, patch_size, axis=0), patch_size, axis=1)
    pixel_map = pixel_map[: image_size[0], : image_size[1]]
    pixel_focus = pixel_map[row_start:row_end, col_start:col_end]

    focus_h, focus_w = pixel_focus.shape
    pad_h = (-focus_h) % patch_size
    pad_w = (-focus_w) % patch_size
    if pad_h or pad_w:
        pixel_focus = np.pad(pixel_focus, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=np.nan)

    new_rows = pixel_focus.shape[0] // patch_size
    new_cols = pixel_focus.shape[1] // patch_size
    patch_map = pixel_focus.reshape(new_rows, patch_size, new_cols, patch_size)
    patch_map = np.nanmean(patch_map, axis=(1, 3))

    lon_min, lon_max, lat_min, lat_max = region
    height, width = image_size
    lon_step = (lon_max - lon_min) / max(1, width)
    lat_step = (lat_max - lat_min) / max(1, height)

    lon_min_focus = lon_min + col_start * lon_step
    lon_max_focus = lon_min + col_end * lon_step
    lat_max_focus = lat_max - row_start * lat_step
    lat_min_focus = lat_max - row_end * lat_step

    np.savetxt(out_dir / "lime_patch.csv", patch_map.reshape(-1), delimiter=",", header="mean_abs", comments="")
    plt.figure(figsize=(6, 5))
    plt.imshow(
        patch_map,
        cmap="magma",
        extent=[lon_min_focus, lon_max_focus, lat_min_focus, lat_max_focus],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
    )
    plt.colorbar(label="Mean |weight|")
    plt.title("LIME Mean | Spatial Patches")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    lon_ticks = np.linspace(lon_min_focus, lon_max_focus, 5)
    lat_ticks = np.linspace(lat_min_focus, lat_max_focus, 5)
    plt.xticks(lon_ticks, [f"{v:.2f}" for v in lon_ticks])
    plt.yticks(lat_ticks, [f"{v:.2f}" for v in lat_ticks])
    plt.tight_layout()
    plt.savefig(out_dir / "lime_patch.png", dpi=160)
    plt.close()

    # Feature list
    with open(out_dir / "lime_features.txt", "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    return weights_all


# --------------- Integrated Gradients (pixel-level) --------------- #
def run_pixel_attribution(
    model: torch.nn.Module,
    background: torch.Tensor,
    explain_images: torch.Tensor,
    region: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    focus_rows: Tuple[int, int],
    focus_cols: Tuple[int, int],
    out_dir: Path,
    logger: logging.Logger,
    ig_steps: int = 50,
    channel_names: List[str] | None = None,
) -> np.ndarray:
    """Compute pixel-level importance via Integrated Gradients."""
    out_dir.mkdir(parents=True, exist_ok=True)
    attr_path = out_dir / "pixel_attribution.npy"
    mean_path = out_dir / "pixel_attribution_mean.npy"

    if attr_path.exists() and mean_path.exists():
        logger.info("Found existing pixel attribution results. Regenerating plots.")
        attr_all = np.load(attr_path)
        mean_attr = np.load(mean_path)
    else:
        model.eval()
        with torch.no_grad():
            baseline = background.mean(dim=0)  # (T, C, H, W)

        num_samples = explain_images.shape[0]
        attr_all = []

        logger.info(f"Computing Integrated Gradients (steps={ig_steps}) for {num_samples} samples...")
        for idx in range(num_samples):
            single = explain_images[idx : idx + 1]  # (1, T, C, H, W)
            bl = baseline.unsqueeze(0)               # (1, T, C, H, W)
            delta = single - bl

            # Accumulate gradients along interpolation path
            accum_grad = torch.zeros_like(single)
            for step in range(ig_steps):
                alpha = step / max(ig_steps - 1, 1)
                interp = bl + alpha * delta
                interp = interp.clone().detach().requires_grad_(True)
                pred = model(interp)
                scalar = pred.mean()
                scalar.backward()
                accum_grad += interp.grad.detach()

            # Integrated Gradients = delta * mean(gradients)
            ig = (delta * accum_grad / ig_steps).detach().cpu().numpy()[0]
            # ig shape: (T, C, H, W) → aggregate over T and C
            ig_pixel = np.abs(ig).mean(axis=(0, 1))  # (H, W)
            attr_all.append(ig_pixel)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (idx + 1) % 10 == 0 or idx == num_samples - 1:
                logger.info(f"  Pixel attribution {idx + 1}/{num_samples}")

        attr_all = np.stack(attr_all, axis=0)  # (N, H, W)
        mean_attr = attr_all.mean(axis=0)       # (H, W)
        np.save(attr_path, attr_all)
        np.save(mean_path, mean_attr)

    # ---- Crop to focus region ---- #
    row_start, row_end = focus_rows
    col_start, col_end = focus_cols
    row_start = max(0, min(row_start, image_size[0]))
    row_end = max(row_start + 1, min(row_end, image_size[0]))
    col_start = max(0, min(col_start, image_size[1]))
    col_end = max(col_start + 1, min(col_end, image_size[1]))
    focus_attr = mean_attr[row_start:row_end, col_start:col_end]

    # ---- Lon / Lat extent ---- #
    lon_min, lon_max, lat_min, lat_max = region
    height, width = image_size
    lon_step = (lon_max - lon_min) / max(1, width)
    lat_step = (lat_max - lat_min) / max(1, height)
    lon_min_f = lon_min + col_start * lon_step
    lon_max_f = lon_min + col_end * lon_step
    lat_max_f = lat_max - row_start * lat_step
    lat_min_f = lat_max - row_end * lat_step

    # ---- Per-channel attribution (also pixel-level, aggregate spatially) ---- #
    ch_attr = None
    if channel_names is not None:
        ch_attr_path = out_dir / "pixel_attribution_per_channel.npy"
        if ch_attr_path.exists():
            ch_attr = np.load(ch_attr_path)
        else:
            model.eval()
            with torch.no_grad():
                bl_single = background.mean(dim=0).unsqueeze(0)
            ch_accum = []
            for idx in range(explain_images.shape[0]):
                single = explain_images[idx : idx + 1]
                delta = single - bl_single
                accum_grad = torch.zeros_like(single)
                for step in range(ig_steps):
                    alpha = step / max(ig_steps - 1, 1)
                    interp = (bl_single + alpha * delta).clone().detach().requires_grad_(True)
                    pred = model(interp)
                    pred.mean().backward()
                    accum_grad += interp.grad.detach()
                ig = (delta * accum_grad / ig_steps).detach().cpu().numpy()[0]
                ig_ch = np.abs(ig).mean(axis=0)[:, row_start:row_end, col_start:col_end].mean(axis=(1, 2))
                ch_accum.append(ig_ch)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            ch_attr = np.stack(ch_accum, axis=0).mean(axis=0)
            np.save(ch_attr_path, ch_attr)

    # ---- Combined figure: (a) channel + (b) pixel heatmap ---- #
    _plot_combined_attribution(
        ch_attr, channel_names, focus_attr,
        lon_min_f, lon_max_f, lat_min_f, lat_max_f,
        out_dir,
    )

    logger.info(f"Pixel attribution saved to {out_dir}")
    return attr_all


# --------------- combined (a)+(b) figure --------------- #
def _plot_combined_attribution(
    ch_attr,
    channel_names,
    focus_attr: np.ndarray,
    lon_min_f: float, lon_max_f: float,
    lat_min_f: float, lat_max_f: float,
    out_dir: Path,
):
    """Create a side-by-side figure: (a) channel importance, (b) pixel heatmap."""
    import matplotlib.gridspec as gridspec

    n_ch = ch_attr.shape[0] if ch_attr is not None else 10
    fig = plt.figure(figsize=(14, max(5.5, 0.4 * n_ch + 1.0)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.30)

    # ---- (a) channel bar chart ---- #
    ax_a = fig.add_subplot(gs[0])
    if ch_attr is not None and channel_names is not None:
        _plot_channel_bar(ch_attr, channel_names, ax=ax_a, title="(a) 输入变量重要性")
    else:
        ax_a.set_title("(a) 输入变量重要性", fontsize=12, pad=8)

    # ---- (b) pixel heatmap ---- #
    ax_b = fig.add_subplot(gs[1])
    from matplotlib.colors import LinearSegmentedColormap
    _white_red = LinearSegmentedColormap.from_list(
        "white_red", ["#ffffff", "#fee0d2", "#fc9272", "#de2d26", "#67000d"]
    )
    im = ax_b.imshow(
        focus_attr,
        cmap=_white_red,
        extent=[lon_min_f, lon_max_f, lat_min_f, lat_max_f],
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax_b, shrink=0.82, pad=0.03)
    cbar.set_label("mean |∇ × Δx|", fontsize=10)
    ax_b.set_title("(b) 逐像素重要性", fontsize=12, pad=8)
    ax_b.set_xlabel("经度 (°E)", fontsize=10)
    ax_b.set_ylabel("纬度 (°N)", fontsize=10)
    lon_ticks = np.linspace(lon_min_f, lon_max_f, 5)
    lat_ticks = np.linspace(lat_min_f, lat_max_f, 5)
    ax_b.set_xticks(lon_ticks)
    ax_b.set_xticklabels([f"{v:.2f}" for v in lon_ticks])
    ax_b.set_yticks(lat_ticks)
    ax_b.set_yticklabels([f"{v:.2f}" for v in lat_ticks])

    plt.savefig(out_dir / "pixel_attribution_combined.png", dpi=200, bbox_inches="tight")
    # Also save individual panels
    plt.savefig(out_dir / "pixel_attribution_spatial.png", dpi=200, bbox_inches="tight")
    plt.close()


# --------------- channel bar plot (SHAP-style) --------------- #
def _plot_channel_bar(
    mean_imp: np.ndarray,
    channel_names: List[str],
    save_path=None,
    ax=None,
    title: str = "输入变量重要性",
):
    """Horizontal bar chart sorted by importance, SHAP-style.

    If *ax* is given, draw into that axes (for combined figures).
    Otherwise create a standalone figure and save to *save_path*.
    """
    from matplotlib.colors import LinearSegmentedColormap

    n = mean_imp.shape[0]
    names = [channel_names[i] if i < len(channel_names) else f"c{i}" for i in range(n)]

    order = np.argsort(mean_imp)  # ascending
    sorted_imp = mean_imp[order]
    sorted_names = [names[i] for i in order]

    cmap = LinearSegmentedColormap.from_list(
        "shap_blue", ["#d0e1f9", "#1E88E5", "#0D47A1"]
    )
    norm_vals = sorted_imp / (sorted_imp.max() + 1e-12)
    colours = [cmap(v) for v in norm_vals]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 0.45 * n + 1.0))

    y_pos = np.arange(n)
    ax.barh(y_pos, sorted_imp, height=0.65, color=colours, edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("mean(|ΔPrediction|)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    x_max = sorted_imp.max()
    for i, v in enumerate(sorted_imp):
        ax.text(v + x_max * 0.015, i, f"{v:.4f}", va="center", fontsize=8, color="#333")

    if standalone:
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()


def run_channel_occlusion(
    model: torch.nn.Module,
    explain_images: torch.Tensor,
    out_dir: Path,
    batch_size: int,
    logger: logging.Logger,
    channel_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / "channel_occlusion_per_sample.npy"
    mean_path = out_dir / "channel_occlusion_mean.npy"

    if per_sample_path.exists() and mean_path.exists():
        logger.info("Found existing channel occlusion results. Regenerating plots.")
        per_sample = np.load(per_sample_path)
        mean_imp = np.load(mean_path)
        csv_path = out_dir / "channel_occlusion.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("channel,mean_abs\n")
            for idx, val in enumerate(mean_imp):
                name = channel_names[idx] if idx < len(channel_names) else f"c{idx}"
                f.write(f"{name},{val}\n")
        _plot_channel_bar(mean_imp, channel_names, out_dir / "channel_occlusion.png",
                        title="输入变量重要性")
        return per_sample, mean_imp

    model.eval()
    num_samples, _, num_channels, _, _ = explain_images.shape

    baseline_preds = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = explain_images[start:end]
        with torch.no_grad():
            preds = model(batch).mean(dim=(1, 2, 3))
        baseline_preds.append(preds.cpu().numpy())
    baseline = np.concatenate(baseline_preds, axis=0)

    per_sample = np.zeros((num_samples, num_channels), dtype=np.float32)

    logger.info("Computing channel occlusion importance...")
    for ch in range(num_channels):
        deltas = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch = explain_images[start:end].clone()
            batch[:, :, ch, :, :] = 0.0
            with torch.no_grad():
                preds = model(batch).mean(dim=(1, 2, 3))
            delta = np.abs(preds.cpu().numpy() - baseline[start:end])
            deltas.append(delta)
        per_sample[:, ch] = np.concatenate(deltas, axis=0)

    mean_imp = per_sample.mean(axis=0)
    np.save(per_sample_path, per_sample)
    np.save(mean_path, mean_imp)

    csv_path = out_dir / "channel_occlusion.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("channel,mean_abs\n")
        for idx, val in enumerate(mean_imp):
            name = channel_names[idx] if idx < len(channel_names) else f"c{idx}"
            f.write(f"{name},{val}\n")
    _plot_channel_bar(mean_imp, channel_names, out_dir / "channel_occlusion.png",
                    title="输入变量重要性")

    return per_sample, mean_imp


def main():
    parser = argparse.ArgumentParser(description="Run SHAP and LIME explainability analysis")
    parser.add_argument("--config", type=str, default="./data/config.yaml")
    parser.add_argument("--model", type=str, default="./output/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="./output/explain")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--background-size", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--focus-rows", type=str, default="86:169")
    parser.add_argument("--focus-cols", type=str, default="160:203")
    parser.add_argument("--lime-samples", type=int, default=200)
    parser.add_argument("--run-shap", action="store_true", help="Enable SHAP analysis")
    parser.add_argument("--skip-channel-occlusion", action="store_true", help="Skip channel occlusion analysis")
    parser.add_argument("--run-pixel-attribution", action="store_true", help="Run Integrated Gradients pixel-level attribution")
    parser.add_argument("--ig-steps", type=int, default=50, help="Interpolation steps for Integrated Gradients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logger = _setup_logger()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = ConfigLoader.load_config(args.config)
    device_str = args.device or config.get("train", {}).get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    dataset = StreamflowDataset(
        image_dir=config.get("data", {}).get("image_dir", "./data/images"),
        station_dir=config.get("data", {}).get("station_dir", "./data/stations"),
        config=config,
        normalize=True,
    )

    split_indices = _split_indices(len(dataset), args.split)
    if not split_indices:
        raise RuntimeError("No data available for the requested split.")

    explain_indices = _select_indices(split_indices, args.num_samples)
    background_indices = _select_indices(
        _split_indices(len(dataset), "train"),
        args.background_size,
    )
    if not background_indices:
        background_indices = _select_indices(split_indices, args.background_size)

    explain_images, explain_dates = _load_images(dataset, explain_indices)
    background_images, _ = _load_images(dataset, background_indices)

    model = StreamflowPredictionModel(config)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    explain_images = explain_images.to(device)
    background_images = background_images.to(device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": args.config,
        "model": args.model,
        "split": args.split,
        "num_samples": len(explain_indices),
        "background_size": len(background_indices),
        "patch_size": args.patch_size,
        "focus_rows": args.focus_rows,
        "focus_cols": args.focus_cols,
        "lime_samples": args.lime_samples,
        "run_shap": bool(args.run_shap),
        "skip_channel_occlusion": bool(args.skip_channel_occlusion),
        "run_pixel_attribution": bool(args.run_pixel_attribution),
        "ig_steps": args.ig_steps,
        "device": str(device),
    }
    with open(out_dir / "explain_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.savetxt(out_dir / "explain_indices.csv", np.array(explain_indices), delimiter=",", header="index", comments="")
    with open(out_dir / "explain_dates.txt", "w", encoding="utf-8") as f:
        for d in explain_dates:
            f.write(str(d) + "\n")

    lime_dir = out_dir / "lime"
    if args.run_shap:
        shap_dir = out_dir / "shap"
        run_shap(model, background_images, explain_images, shap_dir, args.batch_size, logger)
    region = tuple(config.get("data", {}).get("region", [0.0, 1.0, 0.0, 1.0]))
    img_size_cfg = config.get("data", {}).get("image_size", [128, 128])
    if isinstance(img_size_cfg, int):
        image_size = (int(img_size_cfg), int(img_size_cfg))
    else:
        image_size = (int(img_size_cfg[0]), int(img_size_cfg[1]))

    def _parse_slice(value: str, default: Tuple[int, int]) -> Tuple[int, int]:
        try:
            parts = value.split(":")
            if len(parts) != 2:
                return default
            return int(parts[0]), int(parts[1])
        except Exception:
            return default

    focus_rows = _parse_slice(args.focus_rows, (0, image_size[0]))
    focus_cols = _parse_slice(args.focus_cols, (0, image_size[1]))
    run_lime(
        model,
        background_images,
        explain_images,
        args.patch_size,
        args.lime_samples,
        region,
        image_size,
        focus_rows,
        focus_cols,
        lime_dir,
        logger,
        infer_batch_size=args.batch_size,
    )
    if not args.skip_channel_occlusion:
        channel_names = [
            "河流流量(GLOFAS)",
            "2m露点温度",
            "地表温度",
            "总降水量",
            "总蒸发量",
            "太阳辐射↓",
            "地下径流",
            "潜热通量",
            "净热辐射",
            "潜在蒸发",
        ]
        run_channel_occlusion(
            model,
            explain_images,
            out_dir / "channel_occlusion",
            args.batch_size,
            logger,
            channel_names,
        )

    if args.run_pixel_attribution:
        channel_names_ig = [
            "河流流量(GLOFAS)",
            "2m露点温度",
            "地表温度",
            "总降水量",
            "总蒸发量",
            "太阳辐射↓",
            "地下径流",
            "潜热通量",
            "净热辐射",
            "潜在蒸发",
        ]
        run_pixel_attribution(
            model,
            background_images,
            explain_images,
            region,
            image_size,
            focus_rows,
            focus_cols,
            out_dir / "pixel_attribution",
            logger,
            ig_steps=args.ig_steps,
            channel_names=channel_names_ig,
        )

    logger.info("Explainability analysis complete.")


if __name__ == "__main__":
    main()
