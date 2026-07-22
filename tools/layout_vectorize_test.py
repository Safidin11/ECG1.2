"""Проверка фикса: раскладка-НАША, метки nnU-Net игнорируем.

nnU-Net на реальном фото путает, какое отведение где (перевёрнутая/сбитая
нумерация). Но маску «где чернила» он даёт неплохую. Поэтому:
  1) маску берём как БИНАРНУЮ трассу (любая метка > 0),
  2) строки находим по калибровочным импульсам (левое поле),
  3) колонки = 4 равные доли блока 3x4 + ритм-строка на всю ширину,
  4) отведения назначаем по ИЗВЕСТНОЙ раскладке 3x4+ритм,
  5) векторизуем сами: по колонкам берём среднюю y трассы, переводим в мВ
     через калибровку (высота калибр-импульса = 1 мВ) и в секунды через сетку.
"""
import cv2, numpy as np
from scipy.signal import find_peaks

LAYOUT = {  # (row_idx, col_idx) -> lead ; row 0..2 = блок 3x4, ритм отдельно
    (0, 0): "I", (0, 1): "aVR", (0, 2): "V1", (0, 3): "V4",
    (1, 0): "II", (1, 1): "aVL", (1, 2): "V2", (1, 3): "V5",
    (2, 0): "III", (2, 1): "aVF", (2, 2): "V3", (2, 3): "V6",
}
RHYTHM_LEAD = "II"
FS = 500


def detect_row_bands(trace):
    """4 строки по проекции МАСКИ (в ней нет текста/сетки -> устойчиво).

    Возвращает список (y_lo, y_hi, y_center) по 4 полосам трасс.
    """
    H, W = trace.shape
    prof = trace.sum(1).astype(np.float32)
    prof = cv2.blur(prof.reshape(-1, 1), (1, max(9, H // 40))).ravel()
    pk, _ = find_peaks(prof, distance=H // 8, height=prof.max() * 0.15)
    # оставим до 4 сильнейших, по порядку сверху вниз
    if len(pk) > 4:
        pk = pk[np.argsort(prof[pk])[::-1][:4]]
    centers = sorted(pk.tolist())
    bands = []
    for i, c in enumerate(centers):
        up = (c - centers[i - 1]) // 2 if i > 0 else H
        dn = (centers[i + 1] - c) // 2 if i < len(centers) - 1 else H
        half = min(max(up, dn), int(H * 0.14))
        bands.append((max(0, c - half), min(H, c + half), c))
    return bands


def detect_mm_per_px(gray):
    """Шаг мелкой сетки (1 мм) по автокорреляции строки яркости."""
    H, W = gray.shape
    row = 255 - gray[H // 2, :].astype(np.float32)
    row -= row.mean()
    ac = np.correlate(row, row, "full")[W - 1:]
    ac[:3] = 0
    pk, _ = find_peaks(ac[:60], height=ac.max() * 0.2)
    return float(pk[0]) if len(pk) else 4.0  # px на 1 мм


def vectorise_cell(trace_mask, x0, x1, y_lo, y_hi, mm_px, seconds):
    """Вернуть сигнал (мВ) по клетке [x0:x1] в полосе строки [y_lo:y_hi]."""
    sub = trace_mask[y_lo:y_hi, x0:x1]
    n = x1 - x0
    ys = np.full(n, np.nan)
    for i in range(n):
        rows = np.where(sub[:, i] > 0)[0]
        if len(rows):
            ys[i] = rows.mean() + y_lo
    # интерполяция пропусков
    idx = np.arange(n)
    good = ~np.isnan(ys)
    if good.sum() < 5:
        return None
    ys = np.interp(idx, idx[good], ys[good])
    # px -> мВ : вверх = положительно; 10 мм/мВ * mm_px пикселей на мВ
    mV = -(ys - np.median(ys)) / (10.0 * mm_px)
    # подавление выбросов от разрывов маски: медианный фильтр + клип
    from scipy.signal import medfilt
    mV = medfilt(mV, 5)
    mV = np.clip(mV, -3.0, 3.0)
    # ресемпл до FS*seconds
    target = int(FS * seconds)
    xp = np.linspace(0, 1, n)
    return np.interp(np.linspace(0, 1, target), xp, mV)


def run(core_ready_path, mask_path, out_png):
    bgr = cv2.imread(core_ready_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    trace = (mask > 0).astype(np.uint8)
    H, W = gray.shape

    bands = detect_row_bands(trace)
    print("row bands (lo,hi,center):", bands)
    mm_px = detect_mm_per_px(gray)
    print(f"mm_per_px ~ {mm_px:.2f}")

    rows3 = bands[:3]          # блок 3x4
    rhythm = bands[3] if len(bands) > 3 else None

    # 4 равные колонки блока по ширине контента.
    # Калибровочный импульс в самом левом поле -> обрезаем его из 1-й колонки.
    xs_ink = np.where(trace.any(0))[0]
    xL, xR = xs_ink.min(), xs_ink.max()
    cal_trim = int(14 * mm_px)   # ~14 мм калибр-импульс
    colw = (xR - xL) / 4.0
    cols = [(int(xL + c * colw), int(xL + (c + 1) * colw)) for c in range(4)]
    cols[0] = (cols[0][0] + cal_trim, cols[0][1])

    signals = {}
    for r, (y_lo, y_hi, _) in enumerate(rows3):
        for c, (x0, x1) in enumerate(cols):
            lead = LAYOUT[(r, c)]
            sig = vectorise_cell(trace, x0, x1, y_lo, y_hi, mm_px, 2.5)
            if sig is not None:
                signals[lead] = sig
    if rhythm is not None:
        signals[RHYTHM_LEAD + "_rhythm"] = vectorise_cell(
            trace, xL + cal_trim, xR, rhythm[0], rhythm[1], mm_px, 10.0)

    # превью
    order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", "II_rhythm"]
    order = [l for l in order if l in signals]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(order), 1, figsize=(12, 1.4 * len(order)))
    for ax, lead in zip(axs, order):
        s = signals[lead]
        ax.plot(np.arange(len(s)) / FS, s, lw=0.8, color="black")
        ax.set_ylabel(lead, rotation=0, labelpad=25, fontsize=10)
        ax.grid(alpha=0.3); ax.set_ylim(-2, 2.5)
    axs[-1].set_xlabel("сек")
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()
    print("written", out_png, "leads:", order)


if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2], sys.argv[3])
