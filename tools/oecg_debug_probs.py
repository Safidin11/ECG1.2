"""Диагностика: что РЕАЛЬНО видит U-Net движка на снимке.

Отвечает на вопрос «сеть не видит отведение или видит, но его отфильтровала
пост-обработка»: выгружает сырую карту вероятностей класса «сигнал» ДО всех
порогов (label_thresh / threshold_sum / threshold_line_in_mask) и после
обрезки, и печатает вероятность по зонам раскладки.

Запускать интерпретатором ДВИЖКА с cwd = его каталог:
    external/Open-ECG-Digitizer/.venv/bin/python tools/oecg_debug_probs.py \
        -i фото.png -o папка_отчёта
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from src.config.default import get_cfg                     # noqa: E402
from src.utils import find_config_path, import_class_from_path  # noqa: E402
from torchvision.io import decode_image                    # noqa: E402


def load_wrapper(cfg_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(find_config_path(cfg_path))
    cls = import_class_from_path(cfg.MODEL.class_path)
    return cls(**cfg.MODEL.KWARGS), cfg


def save_heat(arr: np.ndarray, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 14 * arr.shape[0] / max(arr.shape[1], 1)), dpi=110)
    im = ax.imshow(arr, cmap="inferno", vmin=0, vmax=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def zone_report(prob: np.ndarray, rows: int, cols: int, label: str) -> None:
    """Средняя/максимальная вероятность сигнала по клеткам раскладки."""
    H, W = prob.shape
    print(f"\n=== {label}: карта {W}x{H}, зоны {rows}x{cols} ===")
    print("     " + "".join(f"  колонка {c+1:<10}" for c in range(cols)))
    for r in range(rows):
        y0, y1 = int(H * r / rows), int(H * (r + 1) / rows)
        cells = []
        for c in range(cols):
            x0, x1 = int(W * c / cols), int(W * (c + 1) / cols)
            z = prob[y0:y1, x0:x1]
            cells.append(f"  ср={z.mean():.3f} макс={z.max():.2f}")
        print(f"стр{r+1}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    wrapper, _ = load_wrapper(a.config)
    img = decode_image(a.image, mode="RGB").unsqueeze(0)

    with torch.no_grad():
        wrapper._check_image_dimensions(img)
        x = wrapper.min_max_normalize(img).to(wrapper.device)
        wrapper.times = {}
        x = wrapper._resample_image(x)
        signal_prob, grid_prob, text_prob = wrapper._get_feature_maps(x)

        raw = signal_prob.squeeze().float().cpu().numpy()
        graw = grid_prob.squeeze().float().cpu().numpy()
        traw = text_prob.squeeze().float().cpu().numpy()
        save_heat(graw, os.path.join(a.out_dir, "prob_grid.png"),
                  "Вероятность класса «сетка» — СЫРАЯ")
        zone_report(graw, a.rows, a.cols, "класс СЕТКА (сырая)")
        zone_report(traw, a.rows, a.cols, "класс ТЕКСТ/ФОН (сырая)")
        save_heat(raw, os.path.join(a.out_dir, "prob_raw.png"),
                  "Вероятность класса «сигнал» — СЫРАЯ, до порогов")
        zone_report(raw, a.rows, a.cols, "СЫРАЯ карта (до обрезки и порогов)")

        params = wrapper.perspective_detector(grid_prob)
        pts = wrapper.cropper(signal_prob, params)
        ai, asp, agp, atp = wrapper._align_feature_maps(x, signal_prob, grid_prob, text_prob, pts)
        aligned = asp.squeeze().float().cpu().numpy()
        save_heat(aligned, os.path.join(a.out_dir, "prob_aligned.png"),
                  "После обрезки и выравнивания перспективы")
        zone_report(aligned, a.rows, a.cols, "ПОСЛЕ обрезки")

        # что останется после порога label_thresh (первый фильтр пост-обработки)
        thr = wrapper.signal_extractor.label_thresh
        print(f"\n=== порог label_thresh={thr}: доля выживших пикселей по зонам ===")
        surv = (aligned > thr).astype(np.float32)
        zone_report(surv, a.rows, a.cols, f"маска > {thr}")
        save_heat(surv, os.path.join(a.out_dir, "prob_thresholded.png"),
                  f"После порога label_thresh={thr}")

    print(f"\nкартинки -> {a.out_dir}")


if __name__ == "__main__":
    main()
