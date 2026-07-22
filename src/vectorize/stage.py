"""Стадия vectorize: маска + раскладка -> сигнал по каждому отведению.

Для каждой клетки (bbox отведения из layout) берём столбцы маски-трассы,
по каждому столбцу — средняя y пикселей трассы, переводим в мВ через масштаб
сетки (10 мм/мВ), изолинию сдвигаем к нулю. Пропуски интерполируем, выбросы
от разрывов маски гасим медианным фильтром и клипом.

Это НАША векторизация (не felixkrones): она устойчива к тому, что nnU-Net
путает метки отведений на реальных фото — мы уже знаем, где какое отведение,
из стадии layout.

Вход:  манифест layout.json (layout.cells + mask_png).
Выход: vectorize.json (манифест + signal_npy/leads/fs) + preview.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import medfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir  # noqa: E402

STAGE = "vectorize"
log = get_logger(STAGE)

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _vectorise_cell(trace, bbox, mm_px, seconds, fs, clip_mV):
    x0, y0, x1, y1 = bbox
    sub = trace[y0:y1, x0:x1]
    n = x1 - x0
    ys = np.full(n, np.nan)
    for i in range(n):
        rows = np.where(sub[:, i] > 0)[0]
        if len(rows):
            ys[i] = rows.mean() + y0
    idx = np.arange(n)
    good = ~np.isnan(ys)
    if good.sum() < 5:
        return None, 0.0
    coverage = float(good.mean())
    ys = np.interp(idx, idx[good], ys[good])
    mV = -(ys - np.median(ys)) / (10.0 * mm_px)   # вверх = положительно
    mV = medfilt(mV, 5)
    mV = np.clip(mV, -clip_mV, clip_mV)
    target = int(fs * seconds)
    xp = np.linspace(0, 1, n)
    return np.interp(np.linspace(0, 1, target), xp, mV), coverage


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    clip_mV = float(config.get("_stage_params", {}).get("clip_mV", 3.0))

    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    layout = manifest.get("layout")
    if not layout:
        # Мягкая деградация: без раскладки оставляем сигнал ядра felixkrones.
        log.warning("STAGE %s: нет layout — пропуск (оставляю сигнал ядра)", STAGE)
        out_path = out_dir / "vectorize.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return str(out_path)

    mask = cv2.imread(manifest["mask_png"], cv2.IMREAD_UNCHANGED)
    trace = (mask > 0).astype(np.uint8)
    mm_px = layout["mm_per_px"]
    fs = manifest.get("fs", 500)
    n_full = int(fs * 10)

    signals = {}
    coverage = {}
    for lead, cell in layout["cells"].items():
        sig, cov = _vectorise_cell(trace, cell["bbox"], mm_px, cell["seconds"], fs, clip_mV)
        if sig is not None:
            signals[lead] = sig
            coverage[lead] = round(cov, 3)
    # ритм-строка (полные 10с ведущего отведения)
    rc = layout.get("rhythm")
    rhythm_sig = None
    if rc:
        rhythm_sig, rcov = _vectorise_cell(trace, rc["bbox"], mm_px, rc["seconds"], fs, clip_mV)

    # Собираем матрицу 12 × n_full (короткие отведения дополняем NaN до 10с)
    mat = np.full((n_full, len(LEAD_ORDER)), np.nan, dtype=np.float32)
    for j, lead in enumerate(LEAD_ORDER):
        if lead == "II" and rhythm_sig is not None:
            mat[:, j] = rhythm_sig[:n_full]
        elif lead in signals:
            s = signals[lead]
            mat[: len(s), j] = s
    signal_npy = out_dir / "signal.npy"
    np.save(signal_npy, mat)

    # превью
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(12, 1, figsize=(12, 16))
    for ax, lead, j in zip(axs, LEAD_ORDER, range(12)):
        s = mat[:, j]
        t = np.arange(len(s)) / fs
        ax.plot(t, np.nan_to_num(s), lw=0.7, color="black")
        cov = coverage.get(lead, 1.0 if lead == "II" else 0.0)
        ax.set_ylabel(f"{lead}\ncov={cov:.0%}", rotation=0, labelpad=32, fontsize=9, va="center")
        ax.set_ylim(-2, 2.5)
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("сек")
    fig.suptitle("ECG1.2 — реконструкция по нашей раскладке (demo, не медизделие)", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    preview = out_dir / "preview.png"
    plt.savefig(preview, dpi=110)
    plt.close()

    manifest["signal_npy"] = str(signal_npy)
    manifest["preview"] = str(preview)
    manifest["leads"] = LEAD_ORDER
    manifest["coverage"] = coverage
    manifest["vectorizer"] = "layout-aware (own), trace-mask from felixkrones nnU-Net"

    out_path = out_dir / "vectorize.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    lowcov = [l for l, c in coverage.items() if c < 0.5]
    log.info("STAGE %s: 12 отведений, низкое покрытие=%s -> %s", STAGE, lowcov or "нет", out_path)
    return str(out_path)
