"""Стадия layout: шаблонно-управляемая локализация отведений.

Раскладка больше не захардкожена: шаблоны описаны в configs/lead_layouts.yml
(схема как у Ahus-AIM/Open-ECG-Digitizer — grid из строк, строка с одинаковыми
отведениями = ритм-строка). Поддерживаются 3×4+ритм, 6×2+ритм, 12×1 и др.

Выбор шаблона: params.template = "auto" (по числу строк) или имя шаблона.
Число строк определяем по маске nnU-Net (в ней нет текста → проекция чистая),
а базовую линию/трассу берём из цветовых чернил (плотные, без сетки).

КАЖДОЕ отведение локализуется независимо: своё окно (центр строки ± доля до
соседней строки) и своя базовая линия (мода гистограммы y). Соседи не влияют.

Вход:  segment.json (mask_png, core_ready_image).
Выход: layout.json (+ блок 'layout': cells, rhythm_strips, cols, ...) + overlay.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir, color_ink  # noqa: E402

STAGE = "layout"
log = get_logger(STAGE)

WIN_FRAC = 0.72     # доля расстояния до соседней строки, задающая окно поиска
LABEL_TRIM_MM = 6   # обрезка подписи отведения в начале каждой колонки
CAL_TRIM_MM = 14    # обрезка калибр-импульса в 1-й колонке
LAYOUTS_CFG = Path(__file__).resolve().parent.parent.parent / "configs" / "lead_layouts.yml"


def load_layouts():
    with open(LAYOUTS_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("paper_seconds", 10), cfg.get("layouts", {})


def coverage_profile(ink: np.ndarray) -> np.ndarray:
    """Профиль ПОКРЫТИЯ по строкам (доля колонок с чернилами), сглаженный [0..1].

    По цветовым чернилам (плотные, без сетки) — работает и на бледных фото, где
    маска nnU-Net разрежена. Базовая (изоэлектрическая) линия каждой строки даёт
    самый заметный пик покрытия.
    """
    H = ink.shape[0]
    cov = cv2.blur(ink.mean(1).astype(np.float32).reshape(-1, 1), (1, max(7, H // 90))).ravel()
    m = cov.max()
    return cov / m if m > 0 else cov


def _refine(cov, y, win):
    lo = max(0, int(y - win))
    hi = min(len(cov), int(y + win))
    return lo + int(np.argmax(cov[lo:hi])) if hi > lo else int(y)


def detect_row_centers(cov: np.ndarray, R: int):
    """Ровно R центров строк: R равномерных позиций, уточнённых к максимуму покрытия."""
    ys = np.where(cov > 0.15)[0]
    if len(ys) == 0:
        return []
    yT, yB = int(ys.min()), int(ys.max())
    sp = (yB - yT) / R
    centers = sorted(set(_refine(cov, yT + sp * (i + 0.5), sp * 0.4) for i in range(R)))
    return centers


def score_layout(cov: np.ndarray, R: int) -> float:
    """Насколько хорошо R строк объясняют профиль покрытия (больше — лучше).

    on  — среднее покрытие на найденных строках;
    off — макс. покрытие в СЕРЕДИНЕ промежутков (там строки быть не должно);
    reg — неравномерность шага (у правильного R строки равномерны).
    """
    centers = detect_row_centers(cov, R)
    if len(centers) < R:
        return -1.0                      # рефайн схлопнул центры -> R завышен
    on = float(np.mean([cov[c] for c in centers]))
    offs = []
    for i in range(len(centers) - 1):
        a, b = centers[i], centers[i + 1]
        q = (b - a) // 4
        offs.append(float(cov[a + q:b - q].max()) if b - q > a + q else float(cov[(a + b) // 2]))
    off = float(np.mean(offs)) if offs else 0.0
    spac = np.diff(centers)
    reg = float(np.std(spac) / (np.mean(spac) + 1e-6))
    return (on - off) - 0.6 * reg


def detect_mm_per_px(gray: np.ndarray) -> float:
    H, W = gray.shape
    row = 255 - gray[H // 2].astype(np.float32)
    row -= row.mean()
    ac = np.correlate(row, row, "full")[W - 1:]
    ac[:3] = 0
    pk, _ = find_peaks(ac[:60], height=ac.max() * 0.2)
    return float(pk[0]) if len(pk) else 8.0


def lead_baseline(ink: np.ndarray, x0, x1, lo, hi) -> int:
    """Своя базовая линия отведения = мода гистограммы y его трассы (по чернилам)."""
    ys = np.where(ink[lo:hi, x0:x1] > 0)[0]
    if len(ys) == 0:
        return (lo + hi) // 2
    h, _ = np.histogram(ys, bins=np.arange(0, hi - lo + 2, 2))
    return lo + int(np.argmax(h) * 2)


def _pick_template(name, layouts, cov):
    """Выбрать шаблон: явно по имени или авто по фит-скорингу покрытия.

    Авто: перебираем шаблоны, выбираем тот, чьё число строк лучше всего
    объясняет профиль покрытия (score_layout). Устойчиво к разным форматам
    (3×4, 6×2, 12×1) и к высоким QRS / шапке.
    """
    if name and name != "auto":
        if name not in layouts:
            raise RuntimeError(f"layout: шаблон '{name}' не найден в lead_layouts.yml")
        return name, layouts[name]
    best, best_score = None, -1e9
    for tname, tpl in layouts.items():
        s = score_layout(cov, len(tpl["grid"]))
        log.info("STAGE %s: авто-скоринг %s (%d строк) = %.3f", STAGE, tname, len(tpl["grid"]), s)
        if s > best_score:
            best, best_score = (tname, tpl), s
    if best is None:
        raise RuntimeError("layout: не удалось подобрать шаблон (auto)")
    return best


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Раскладку строим по ЦВЕТОВЫМ ЧЕРНИЛАМ core_ready (плотные, без сетки;
    # работают и на бледных фото, где маска nnU-Net разрежена). Маска опциональна.
    core_img = manifest.get("core_ready_image")
    if not core_img or not Path(core_img).exists():
        log.warning("STAGE %s: нет core_ready — пропуск раскладки (fallback к сигналу ядра)", STAGE)
        out_path = out_dir / "layout.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return str(out_path)

    bgr = cv2.imread(core_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = color_ink(bgr)
    H, W = ink.shape
    cov = coverage_profile(ink)

    paper_sec, layouts = load_layouts()
    tname, tpl = _pick_template(config.get("_stage_params", {}).get("template", "auto"),
                                layouts, cov)
    grid = tpl["grid"]
    cols = tpl["cols"]
    R = len(grid)

    centers = detect_row_centers(cov, R)
    if len(centers) < R:
        raise RuntimeError(f"layout: нашёл строк {len(centers)} < {R} (шаблон {tname})")
    mm_px = detect_mm_per_px(gray)

    xs = np.where(ink.any(0))[0]
    xL, xR = int(xs.min()), int(xs.max())
    cal_trim = int(CAL_TRIM_MM * mm_px)
    label_trim = int(LABEL_TRIM_MM * mm_px)
    colw = (xR - xL) / cols
    columns = [[int(xL + c * colw), int(xL + (c + 1) * colw)] for c in range(cols)]
    columns[0][0] += cal_trim
    for c in range(1, cols):
        columns[c][0] += label_trim

    cells, rhythm_strips = {}, []
    for r, center in enumerate(centers):
        up = center - (centers[r - 1] if r > 0 else 0)
        dn = (centers[r + 1] if r < R - 1 else H) - center
        wlo = max(0, int(center - WIN_FRAC * up))
        whi = min(H, int(center + WIN_FRAC * dn))
        row = grid[r]
        if cols > 1 and all(l == row[0] for l in row):   # ритм-строка на всю ширину
            base = lead_baseline(ink, xL + cal_trim, xR, wlo, whi)
            rhythm_strips.append({"lead": row[0], "bbox": [xL + cal_trim, wlo, xR, whi],
                                  "baseline": base, "seconds": paper_sec})
        else:                                        # обычная строка блока
            for c in range(cols):
                lead = row[c]
                x0, x1 = columns[c]
                base = lead_baseline(ink, x0, x1, wlo, whi)
                cells[lead] = {"row": r, "col": c, "bbox": [x0, wlo, x1, whi],
                               "baseline": base, "seconds": paper_sec / cols}

    manifest["layout"] = {
        "template": tname,
        "cols": cols,
        "grid": grid,                 # раскладка для рендера в исходном виде
        "mm_per_px": mm_px,
        "cal_trim_px": cal_trim,
        "content_x": [xL, xR],
        "row_centers": centers,
        "cells": cells,
        "rhythm_strips": rhythm_strips,
    }

    # overlay
    vis = bgr.copy() if bgr is not None else cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    for lead, cell in cells.items():
        x0, y0, x1, y1 = cell["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 150, 0), 1)
        cv2.line(vis, (x0, cell["baseline"]), (x1, cell["baseline"]), (255, 120, 0), 1)
        cv2.putText(vis, lead, (x0 + 5, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for rs in rhythm_strips:
        x0, y0, x1, y1 = rs["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 0, 0), 1)
        cv2.line(vis, (x0, rs["baseline"]), (x1, rs["baseline"]), (255, 120, 0), 1)
        cv2.putText(vis, rs["lead"] + " (rhythm)", (x0 + 5, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / "overlay.png"), vis)

    out_path = out_dir / "layout.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("STAGE %s: шаблон=%s (%dx%d), %d клеток + %d ритм-строк, строки=%s, mm/px=%.2f",
             STAGE, tname, R, cols, len(cells), len(rhythm_strips), centers, mm_px)
    return str(out_path)
