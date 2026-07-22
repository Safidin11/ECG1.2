"""Стадия layout: определить раскладку отведений по маске сегментации.

Почему по НАШЕЙ раскладке, а не по меткам nnU-Net: felixkrones-модель
присваивает отведение по абсолютной позиции на своей синтетической странице;
на реальных фото с другими пропорциями она путает, какое отведение где
(метки «переворачиваются» по вертикали, II налезает на всё). Проверено на
IMG_4074. Поэтому маску nnU-Net используем как БИНАРНУЮ трассу («где чернила»),
а отведения назначаем сами по известному шаблону 3×4 + ритм.

Строки определяем по проекции маски (в ней нет текста/сетки — устойчиво),
колонки — 4 равные доли контента, калибровочный импульс слева обрезаем.

Вход:  манифест segment.json (с полем mask_png).
Выход: layout.json = манифест + блок 'layout' (строки, колонки, cell->lead,
       mm_per_px). Рядом — overlay.png для проверки.

Шаблоны раскладок будут обобщены в Фазе 3 (Open-ECG-Digitizer templates).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir  # noqa: E402

STAGE = "layout"
log = get_logger(STAGE)

# Шаблон 3×4 + ритм: (row, col) -> отведение; ритм-строка отдельно.
TEMPLATE_3x4 = {
    (0, 0): "I", (0, 1): "aVR", (0, 2): "V1", (0, 3): "V4",
    (1, 0): "II", (1, 1): "aVL", (1, 2): "V2", (1, 3): "V5",
    (2, 0): "III", (2, 1): "aVF", (2, 2): "V3", (2, 3): "V6",
}
RHYTHM_LEAD = "II"


def detect_row_bands(trace: np.ndarray):
    """4 полосы строк по проекции маски. -> [(y_lo,y_hi,center), ...]."""
    H, W = trace.shape
    prof = trace.sum(1).astype(np.float32)
    prof = cv2.blur(prof.reshape(-1, 1), (1, max(9, H // 40))).ravel()
    pk, _ = find_peaks(prof, distance=H // 8, height=prof.max() * 0.15)
    if len(pk) > 4:
        pk = pk[np.argsort(prof[pk])[::-1][:4]]
    centers = sorted(int(p) for p in pk)
    bands = []
    for i, c in enumerate(centers):
        up = (c - centers[i - 1]) // 2 if i > 0 else H
        dn = (centers[i + 1] - c) // 2 if i < len(centers) - 1 else H
        half = min(max(up, dn), int(H * 0.14))
        bands.append((max(0, c - half), min(H, c + half), c))
    return bands


def detect_mm_per_px(gray: np.ndarray) -> float:
    """Шаг мелкой сетки (1 мм) по автокорреляции строки яркости."""
    H, W = gray.shape
    row = 255 - gray[H // 2, :].astype(np.float32)
    row -= row.mean()
    ac = np.correlate(row, row, "full")[W - 1:]
    ac[:3] = 0
    pk, _ = find_peaks(ac[:60], height=ac.max() * 0.2)
    return float(pk[0]) if len(pk) else 4.0


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mask_png = manifest.get("mask_png")
    if not mask_png or not Path(mask_png).exists():
        # Мягкая деградация: без маски раскладку не строим, прокидываем манифест
        # дальше (downstream откатится к сигналу felixkrones из segment).
        log.warning("STAGE %s: нет маски — пропуск раскладки (fallback к сигналу ядра)", STAGE)
        out_path = out_dir / "layout.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return str(out_path)

    mask = cv2.imread(mask_png, cv2.IMREAD_UNCHANGED)
    trace = (mask > 0).astype(np.uint8)
    core_img = manifest.get("core_ready_image")
    gray = cv2.cvtColor(cv2.imread(core_img), cv2.COLOR_BGR2GRAY) if core_img and Path(core_img).exists() else (255 - trace * 255)
    H, W = trace.shape

    bands = detect_row_bands(trace)
    if len(bands) < 4:
        raise RuntimeError(f"layout: найдено строк {len(bands)} (< 4), раскладка не распознана")
    mm_px = detect_mm_per_px(gray)

    xs_ink = np.where(trace.any(0))[0]
    xL, xR = int(xs_ink.min()), int(xs_ink.max())
    cal_trim = int(14 * mm_px)  # калибровочный импульс слева
    colw = (xR - xL) / 4.0
    columns = [[int(xL + c * colw), int(xL + (c + 1) * colw)] for c in range(4)]
    columns[0][0] += cal_trim

    rows3 = bands[:3]
    rhythm = bands[3]

    cells = {}
    for r, (y_lo, y_hi, _) in enumerate(rows3):
        for c, (x0, x1) in enumerate(columns):
            lead = TEMPLATE_3x4[(r, c)]
            cells[lead] = {"row": r, "col": c, "bbox": [x0, y_lo, x1, y_hi], "seconds": 2.5}
    rhythm_cell = {"lead": RHYTHM_LEAD, "bbox": [xL + cal_trim, rhythm[0], xR, rhythm[1]], "seconds": 10.0}

    manifest["layout"] = {
        "template": config.get("_stage_params", {}).get("template", "3x4_rhythm"),
        "mm_per_px": mm_px,
        "cal_trim_px": cal_trim,
        "content_x": [xL, xR],
        "rows": [[a, b, c] for (a, b, c) in bands],
        "columns": columns,
        "cells": cells,
        "rhythm": rhythm_cell,
    }

    # overlay для проверки
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else cv2.imread(core_img)
    for lead, cell in cells.items():
        x0, y0, x1, y1 = cell["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 150, 0), 2)
        cv2.putText(vis, lead, (x0 + 5, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    rx0, ry0, rx1, ry1 = rhythm_cell["bbox"]
    cv2.rectangle(vis, (rx0, ry0), (rx1, ry1), (200, 0, 0), 2)
    cv2.putText(vis, "II (rhythm)", (rx0 + 5, ry0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / "overlay.png"), vis)

    out_path = out_dir / "layout.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("STAGE %s: %d отведений + ритм, строки=%s, mm/px=%.2f -> %s",
             STAGE, len(cells), [b[2] for b in bands], mm_px, out_path)
    return str(out_path)
