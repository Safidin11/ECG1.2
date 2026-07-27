"""Диагностика ПРОПУСКОВ в трассе: есть ли в них слабый сигнал?

Отвечает на вопрос «поднимать ли чувствительность»: для каждой строки
раскладки считает, какая доля столбцов не имеет ни одного пикселя выше порога
label_thresh, и сколько из этих пустых столбцов удалось бы вернуть при более
низком пороге. Если пропуски заполняются при снижении порога — чувствительность
поднимать имеет смысл; если там честный ноль — нет.

Запускать интерпретатором ДВИЖКА с cwd = его каталог.
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

BANDS = [0.02, 0.05, 0.10, 0.20, 0.40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--rows", type=int, default=4)
    a = ap.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(find_config_path(a.config))
    wrapper = import_class_from_path(cfg.MODEL.class_path)(**cfg.MODEL.KWARGS)
    img = decode_image(a.image, mode="RGB").unsqueeze(0)

    with torch.no_grad():
        wrapper._check_image_dimensions(img)
        x = wrapper._resample_image(wrapper.min_max_normalize(img).to(wrapper.device))
        sp, gp, tp = wrapper._get_feature_maps(x)
        params = wrapper.perspective_detector(gp)
        pts = wrapper.cropper(sp, params)
        _, asp, agp, _ = wrapper._align_feature_maps(x, sp, gp, tp, pts)
        prob = asp.squeeze().float().cpu().numpy()
        grid = agp.squeeze().float().cpu().numpy()

    H, W = prob.shape
    print(f"карта {W}x{H}, порог движка label_thresh={wrapper.signal_extractor.label_thresh}\n")
    print("Доля столбцов БЕЗ сигнала (пропуски) при разных порогах:")
    print("строка |" + "".join(f"  порог {t:<5}" for t in BANDS) + "  сетка в пропусках")
    for r in range(a.rows):
        y0, y1 = int(H * r / a.rows), int(H * (r + 1) / a.rows)
        band = prob[y0:y1]
        gband = grid[y0:y1]
        cells, empty_at_default = [], None
        for t in BANDS:
            empty = ~(band > t).any(axis=0)
            cells.append(f"   {100 * empty.mean():5.1f}%  ")
            if abs(t - 0.10) < 1e-9:
                empty_at_default = empty
        # что «видит» класс сетки там, где сигнала нет
        gmax = float(gband[:, empty_at_default].max()) if empty_at_default is not None \
            and empty_at_default.any() else 0.0
        print(f"  {r+1}    |" + "".join(cells) + f"   макс={gmax:.2f}")

    print("\nЕсли при снижении порога доля пропусков заметно падает — есть что вернуть.")


if __name__ == "__main__":
    main()
