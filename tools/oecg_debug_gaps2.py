"""Точная диагностика пропусков: что в них ВМЕСТО сигнала.

Для каждого пропуска берём ожидаемое положение линии (интерполяция по соседним
столбцам, где линия есть) и смотрим в этой точке вероятности всех классов.
Так видно, ушла ли трасса в класс «сетка» (тогда её можно вернуть, подавив
сетку на входе) или там действительно пусто (тогда нужен контраст/разрешение).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from src.config.default import get_cfg                          # noqa: E402
from src.utils import find_config_path, import_class_from_path  # noqa: E402
from torchvision.io import decode_image                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(find_config_path(a.config))
    w = import_class_from_path(cfg.MODEL.class_path)(**cfg.MODEL.KWARGS)
    img = decode_image(a.image, mode="RGB").unsqueeze(0)

    with torch.no_grad():
        w._check_image_dimensions(img)
        x = w._resample_image(w.min_max_normalize(img).to(w.device))
        sp, gp, tp = w._get_feature_maps(x)
        params = w.perspective_detector(gp)
        pts = w.cropper(sp, params)
        _, asp, agp, atp = w._align_feature_maps(x, sp, gp, tp, pts)
        sig = asp.squeeze().float().cpu().numpy()
        grd = agp.squeeze().float().cpu().numpy()
        txt = atp.squeeze().float().cpu().numpy()

    H, W = sig.shape
    print(f"карта {W}x{H}\n")
    print("строка | пропусков | В ТОЧКЕ, где ожидается линия:  сигнал / сетка / текст")
    for r in range(a.rows):
        y0, y1 = int(H * r / a.rows), int(H * (r + 1) / a.rows)
        band, gband, tband = sig[y0:y1], grd[y0:y1], txt[y0:y1]
        has = (band > a.thr).any(axis=0)
        if has.sum() < 10:
            print(f"  {r+1}    | строка пустая")
            continue
        # положение линии там, где она есть = центр масс по вертикали
        ys = np.full(W, np.nan)
        idx = np.flatnonzero(has)
        for c in idx:
            col = band[:, c]
            ys[c] = float((col * np.arange(len(col))).sum() / max(col.sum(), 1e-6))
        # интерполируем ожидаемое положение в пропусках
        gaps = np.flatnonzero(~has)
        if len(gaps) == 0:
            print(f"  {r+1}    |     0%    | пропусков нет")
            continue
        yq = np.interp(gaps, idx, ys[idx])
        s_v, g_v, t_v = [], [], []
        for c, y in zip(gaps, yq):
            lo, hi = max(0, int(y) - 3), min(band.shape[0], int(y) + 4)
            s_v.append(band[lo:hi, c].max())
            g_v.append(gband[lo:hi, c].max())
            t_v.append(tband[lo:hi, c].max())
        print(f"  {r+1}    |   {100*len(gaps)/W:5.1f}%  |   "
              f"{np.mean(s_v):.3f}  /  {np.mean(g_v):.3f}  /  {np.mean(t_v):.3f}")

    print("\nЕсли в пропусках «сетка» заметно выше «сигнала» — трасса ушла в класс сетки.")


if __name__ == "__main__":
    main()
