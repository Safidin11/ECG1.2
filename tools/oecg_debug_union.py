"""Проверка объединения (TTA): даёт ли пара прогонов меньше пропусков.

Гоняем сеть на исходном снимке и на его контрастной версии, берём поточечный
максимум вероятностей класса «сигнал» и считаем пропуски. Смысл: варианты
теряют РАЗНЫЕ участки трассы, и объединение может закрыть больше, чем любой
поодиночке. Мы ничего не дорисовываем — берём только то, что сеть увидела
хотя бы в одном варианте.
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


def probs(w, path: str):
    img = decode_image(path, mode="RGB").unsqueeze(0)
    with torch.no_grad():
        w._check_image_dimensions(img)
        x = w._resample_image(w.min_max_normalize(img).to(w.device))
        sp, gp, tp = w._get_feature_maps(x)
        params = w.perspective_detector(gp)
        pts = w.cropper(sp, params)
        _, asp, _, _ = w._align_feature_maps(x, sp, gp, tp, pts)
    return asp.squeeze().float().cpu().numpy()


def gaps(prob: np.ndarray, rows: int, thr: float) -> list[float]:
    H, W = prob.shape
    out = []
    for r in range(rows):
        band = prob[int(H * r / rows):int(H * (r + 1) / rows)]
        out.append(100 * float((~(band > thr).any(axis=0)).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", required=True, help="вариант 1 (обычный)")
    ap.add_argument("-b", required=True, help="вариант 2 (контрастный)")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(find_config_path(a.config))
    w = import_class_from_path(cfg.MODEL.class_path)(**cfg.MODEL.KWARGS)

    pa, pb = probs(w, a.a), probs(w, a.b)
    if pa.shape != pb.shape:                     # обрезка могла дать разный размер
        h, wd = min(pa.shape[0], pb.shape[0]), min(pa.shape[1], pb.shape[1])
        pa, pb = pa[:h, :wd], pb[:h, :wd]
    pu = np.maximum(pa, pb)

    ga, gb, gu = gaps(pa, a.rows, a.thr), gaps(pb, a.rows, a.thr), gaps(pu, a.rows, a.thr)
    print("строка | обычный | контраст | ОБЪЕДИНЕНИЕ")
    for i in range(a.rows):
        print(f"  {i+1}    |  {ga[i]:5.1f}% |   {gb[i]:5.1f}%  |   {gu[i]:5.1f}%")
    print(f"среднее|  {np.mean(ga):5.1f}% |   {np.mean(gb):5.1f}%  |   {np.mean(gu):5.1f}%")


if __name__ == "__main__":
    main()
