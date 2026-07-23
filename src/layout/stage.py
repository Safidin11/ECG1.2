"""Стадия layout: НЕЗАВИСИМАЯ локализация каждого отведения по маске.

Почему по нашей раскладке, а не по меткам nnU-Net: модель присваивает
отведение по абсолютной позиции на своей синтетической странице и на реальных
фото путает, какое отведение где. Маску берём как «где чернила», а раскладку
3×4+ритм назначаем сами.

КАЖДОЕ отведение локализуется независимо (важно для форматов, где соседние
отведения одной строки стоят на РАЗНОЙ высоте):
  * строки находим по проекции маски (4 центра);
  * колонки — 4 доли контента (калибр-импульс слева обрезаем);
  * для каждой клетки считаем СВОЁ окно поиска (центр строки ± доля до соседней
    строки — соседи не мешают) и СВОЮ базовую линию (мода гистограммы y трассы).
Ритм-строка локализуется отдельно, с окном вниз до края листа.

Вход:  segment.json (mask_png, core_ready_image).
Выход: layout.json (+ блок 'layout' с per-lead window/baseline) + overlay.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir, color_ink  # noqa: E402

STAGE = "layout"
log = get_logger(STAGE)

TEMPLATE_3x4 = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II"
WIN_FRAC = 0.72     # доля расстояния до соседней строки, задающая окно поиска
LABEL_TRIM_MM = 6   # обрезка подписи отведения в начале каждой колонки


def detect_row_centers(trace: np.ndarray):
    H, W = trace.shape
    prof = cv2.blur(trace.sum(1).astype(np.float32).reshape(-1, 1),
                    (1, max(9, H // 40))).ravel()
    pk, _ = find_peaks(prof, distance=H // 8, height=prof.max() * 0.15)
    if len(pk) > 4:
        pk = pk[np.argsort(prof[pk])[::-1][:4]]
    return sorted(int(p) for p in pk)


def detect_mm_per_px(gray: np.ndarray) -> float:
    H, W = gray.shape
    row = 255 - gray[H // 2].astype(np.float32)
    row -= row.mean()
    ac = np.correlate(row, row, "full")[W - 1:]
    ac[:3] = 0
    pk, _ = find_peaks(ac[:60], height=ac.max() * 0.2)
    return float(pk[0]) if len(pk) else 8.0


def lead_baseline(ink: np.ndarray, x0, x1, lo, hi) -> int:
    """Своя базовая линия отведения = мода гистограммы y его трассы в окне.

    Считаем по цветовым «чернилам» (плотная трасса без сетки), а не по маске
    nnU-Net — иначе на разреженной маске мода уезжает на чужую трассу.
    """
    sub = ink[lo:hi, x0:x1]
    ys = np.where(sub > 0)[0]
    if len(ys) == 0:
        return (lo + hi) // 2
    h, _ = np.histogram(ys, bins=np.arange(0, hi - lo + 2, 2))
    return lo + int(np.argmax(h) * 2)


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mask_png = manifest.get("mask_png")
    if not mask_png or not Path(mask_png).exists():
        log.warning("STAGE %s: нет маски — пропуск раскладки (fallback к сигналу ядра)", STAGE)
        out_path = out_dir / "layout.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return str(out_path)

    mask = (cv2.imread(mask_png, cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8)
    core_img = manifest.get("core_ready_image")
    bgr = cv2.imread(core_img) if core_img and Path(core_img).exists() else None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr is not None else (1 - mask) * 255
    ink = color_ink(bgr) if bgr is not None else mask   # плотная трасса без сетки
    H, W = mask.shape

    # Строки ищем по МАСКЕ (в ней нет текста -> проекция чистая),
    # а базовую линию/трассу берём из цветовых чернил (плотнее).
    centers = detect_row_centers(mask)
    if len(centers) < 4:
        raise RuntimeError(f"layout: найдено строк {len(centers)} (< 4)")
    mm_px = detect_mm_per_px(gray)

    xs = np.where(mask.any(0))[0]
    xL, xR = int(xs.min()), int(xs.max())
    cal_trim = int(14 * mm_px)
    label_trim = int(LABEL_TRIM_MM * mm_px)
    colw = (xR - xL) / 4.0
    columns = [[int(xL + c * colw), int(xL + (c + 1) * colw)] for c in range(4)]
    columns[0][0] += cal_trim                      # калибр-импульс в 1-й колонке
    for c in (1, 2, 3):
        columns[c][0] += label_trim                # подпись отведения в начале колонки

    block_centers = centers[:3]
    rhy_center = centers[3]

    cells = {}
    for r, center in enumerate(block_centers):
        up = center - (block_centers[r - 1] if r > 0 else 0)
        dn = (block_centers[r + 1] if r < 2 else rhy_center) - center
        wlo = max(0, int(center - WIN_FRAC * up))
        whi = min(H, int(center + WIN_FRAC * dn))
        for c, (x0, x1) in enumerate(columns):
            lead = TEMPLATE_3x4[r][c]
            base = lead_baseline(ink, x0, x1, wlo, whi)
            cells[lead] = {"row": r, "col": c, "bbox": [x0, wlo, x1, whi],
                           "baseline": base, "seconds": 2.5}
    # ритм: своё окно вниз до края листа
    r_wlo = max(0, int(rhy_center - WIN_FRAC * (rhy_center - block_centers[2])))
    r_base = lead_baseline(ink, xL + cal_trim, xR, r_wlo, H)
    rhythm_cell = {"lead": RHYTHM_LEAD, "bbox": [xL + cal_trim, r_wlo, xR, H],
                   "baseline": r_base, "seconds": 10.0}

    manifest["layout"] = {
        "template": config.get("_stage_params", {}).get("template", "3x4_rhythm"),
        "mm_per_px": mm_px,
        "cal_trim_px": cal_trim,
        "content_x": [xL, xR],
        "row_centers": centers,
        "columns": columns,
        "cells": cells,
        "rhythm": rhythm_cell,
    }

    # overlay: окна отведений + базовые линии
    vis = cv2.imread(core_img) if core_img and Path(core_img).exists() else cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    for lead, cell in cells.items():
        x0, y0, x1, y1 = cell["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 150, 0), 1)
        cv2.line(vis, (x0, cell["baseline"]), (x1, cell["baseline"]), (255, 120, 0), 1)
        cv2.putText(vis, lead, (x0 + 5, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    rx0, ry0, rx1, ry1 = rhythm_cell["bbox"]
    cv2.rectangle(vis, (rx0, ry0), (rx1, ry1), (200, 0, 0), 1)
    cv2.line(vis, (rx0, r_base), (rx1, r_base), (255, 120, 0), 1)
    cv2.imwrite(str(out_dir / "overlay.png"), vis)

    out_path = out_dir / "layout.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("STAGE %s: %d отведений + ритм (независимые окна/baseline), строки=%s, mm/px=%.2f",
             STAGE, len(cells), centers, mm_px)
    return str(out_path)
