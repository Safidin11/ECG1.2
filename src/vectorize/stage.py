"""Стадия vectorize: НЕЗАВИСИМАЯ трассировка каждого отведения.

Для каждой клетки (своё окно + своя базовая линия из layout) трассируем сигнал
«следованием»: в каждом столбце берём кластер пикселей, ближайший к предыдущей
точке (устойчиво к смещению соседних отведений, глубоким S-зубцам и толщине
штриха). Где маска дырявая — подхватываем тёмные пиксели полутона рядом с
текущей траекторией. Разрывы интерполируем и продолжаем (не останавливаемся).
Дрейф базовой линии снимаем скользящей медианой (чинит длинную ритм-строку).

Соседние отведения НЕ влияют друг на друга: у каждого своё окно и базовая линия.

Вход:  layout.json (layout.cells с per-lead bbox/baseline + mask_png + core_ready).
Выход: vectorize.json (signal_npy/leads/fs/coverage) + preview.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir, color_ink  # noqa: E402

STAGE = "vectorize"
log = get_logger(STAGE)

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SLEW_PX = 70          # макс. скачок трассы за столбец (px) — против «прямоугольников»
TEXT_DILATE_PX = 21   # окрестность маски, в которой оставляем чернила (убираем надписи)
TEXT_MIN_RATIO = 0.5  # если маска слишком разрежена — не чистим (fallback на сырьё)


def _suppress_text(ink, mask, bboxes):
    """Убрать надписи («1cm/mV», метки) из чернил, оставив трассу.

    Надписи есть в цветовых чернилах, но их НЕТ в маске nnU-Net (она сегментирует
    только кривую). Поэтому в каждой клетке оставляем чернила рядом с маской
    (dilate). Где маска слишком разрежена (мало покрытия) — не трогаем клетку,
    чтобы не потерять трассу (fallback на сырые чернила).
    """
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TEXT_DILATE_PX, TEXT_DILATE_PX))
    out = ink.copy()
    for (x0, y0, x1, y1) in bboxes:
        ic = ink[y0:y1, x0:x1]
        tot = int(ic.sum())
        if tot == 0:
            continue
        cl = ic * cv2.dilate(mask[y0:y1, x0:x1], ker)
        if cl.sum() >= TEXT_MIN_RATIO * tot:
            out[y0:y1, x0:x1] = cl
    return out


def _clusters(colpix):
    """Кластеры подряд идущих ненулевых пикселей столбца -> (центроид, высота)."""
    ys = np.where(colpix > 0)[0]
    if len(ys) == 0:
        return []
    out, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y - prev > 3:
            out.append(((start + prev) / 2, prev - start + 1))
            start = y
        prev = y
    out.append(((start + prev) / 2, prev - start + 1))
    return out


def _trace_follow(ink, bbox, baseline):
    """Трассировка отведения по цветовым чернилам следованием за кластером.

    В каждом столбце берём кластер, ближайший к текущей траектории (при равенстве
    — тоньше, чтобы не липнуть к разделителю/тексту). Скачок > SLEW_PX запрещён
    (иначе трасса «проваливается» в глубокий S/шум прямоугольником). Разрывы
    интерполируем и продолжаем.
    """
    x0, y0, x1, y1 = bbox
    n = x1 - x0
    ys = np.full(n, np.nan)
    prev = baseline - y0
    for i in range(n):
        cl = _clusters(ink[y0:y1, x0 + i])
        if not cl:
            continue
        c, _h = min(cl, key=lambda t: (abs(t[0] - prev), t[1]))
        if abs(c - prev) <= SLEW_PX:
            ys[i] = c
            prev = c
    idx = np.arange(n)
    good = ~np.isnan(ys)
    if good.sum() < 5:
        return None, 0.0
    cov = float(good.mean())
    ys = np.interp(idx, idx[good], ys[good]) + y0
    return ys, cov


def _to_mv(ys, mm_px, seconds, fs, clip):
    mV = -(ys - np.median(ys)) / (10.0 * mm_px)
    # снятие дрейфа базовой линии скользящей медианой (~0.6с)
    win = int(0.6 * fs)
    win = min(win if win % 2 else win + 1, (len(mV) // 2) * 2 - 1)
    if 3 <= win < len(mV):
        mV = mV - median_filter(mV, size=win)
    mV = np.clip(mV, -clip, clip)
    target = int(fs * seconds)
    return np.interp(np.linspace(0, 1, target), np.linspace(0, 1, len(mV)), mV)


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    clip_mV = float(config.get("_stage_params", {}).get("clip_mV", 3.0))

    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    layout = manifest.get("layout")
    if not layout:
        log.warning("STAGE %s: нет layout — пропуск (оставляю сигнал ядра)", STAGE)
        out_path = out_dir / "vectorize.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return str(out_path)

    core_img = manifest.get("core_ready_image")
    mask = (cv2.imread(manifest["mask_png"], cv2.IMREAD_UNCHANGED) > 0).astype(np.uint8) \
        if manifest.get("mask_png") else None
    ink = color_ink(cv2.imread(core_img)) if core_img and Path(core_img).exists() else mask
    # Подавление надписей по маске (там, где маска достаточно плотная)
    if mask is not None and ink is not None:
        bboxes = [tuple(c["bbox"]) for c in layout["cells"].values()]
        if layout.get("rhythm"):
            bboxes.append(tuple(layout["rhythm"]["bbox"]))
        ink = _suppress_text(ink, mask, bboxes)
    mm_px = layout["mm_per_px"]
    fs = manifest.get("fs", 500)
    n_full = int(fs * 10)

    signals, coverage = {}, {}
    for lead, cell in layout["cells"].items():
        ys, cov = _trace_follow(ink, cell["bbox"], cell["baseline"])
        if ys is not None:
            signals[lead] = _to_mv(ys, mm_px, cell["seconds"], fs, clip_mV)
            coverage[lead] = round(cov, 3)
    rhythm_sig = None
    rc = layout.get("rhythm")
    if rc:
        ys, rcov = _trace_follow(ink, rc["bbox"], rc["baseline"])
        if ys is not None:
            rhythm_sig = _to_mv(ys, mm_px, rc["seconds"], fs, clip_mV)
            coverage["II_rhythm"] = round(rcov, 3)

    mat = np.full((n_full, len(LEAD_ORDER)), np.nan, dtype=np.float32)
    for j, lead in enumerate(LEAD_ORDER):
        if lead == "II" and rhythm_sig is not None:
            mat[:, j] = rhythm_sig[:n_full]
        elif lead in signals:
            s = signals[lead]
            mat[: len(s), j] = s
    signal_npy = out_dir / "signal.npy"
    np.save(signal_npy, mat)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(12, 1, figsize=(12, 16))
    for ax, lead, j in zip(axs, LEAD_ORDER, range(12)):
        s = mat[:, j]
        ax.plot(np.arange(len(s)) / fs, np.nan_to_num(s), lw=0.7, color="black")
        cov = coverage.get("II_rhythm" if lead == "II" else lead, 0.0)
        ax.set_ylabel(f"{lead}\ncov={cov:.0%}", rotation=0, labelpad=32, fontsize=9, va="center")
        ax.set_ylim(-2, 2.5)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("сек")
    fig.suptitle("ECG1.2 — независимая реконструкция по отведениям (demo, не медизделие)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    preview = out_dir / "preview.png"
    plt.savefig(preview, dpi=110)
    plt.close()

    manifest["signal_npy"] = str(signal_npy)
    manifest["preview"] = str(preview)
    manifest["leads"] = LEAD_ORDER
    manifest["coverage"] = coverage
    manifest["vectorizer"] = "per-lead independent trace-following (own baseline/ROI, drift removal)"

    out_path = out_dir / "vectorize.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    lowcov = [l for l, c in coverage.items() if c < 0.5]
    log.info("STAGE %s: 12 отведений (независимо), низкое покрытие=%s", STAGE, lowcov or "нет")
    return str(out_path)
