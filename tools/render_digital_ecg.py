"""Рендер цифровой ЭКГ на «миллиметровке» из реконструированного сигнала.

Берёт сигнал 12×N (.npy, порядок отведений LEAD_ORDER) и рисует стандартную
ЭКГ-распечатку: розовая сетка (1 мм мелкая / 5 мм крупная), 25 мм/с, 10 мм/мВ,
раскладка 3×4 (по 2.5с) + ритм-строка II на 10с, метки отведений, калибр-импульс.

Demo-инструмент, НЕ медизделие.

Запуск:
    ./.venv/bin/python tools/render_digital_ecg.py -i <signal.npy> -o <out.png>
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Раскладка 3x4: строки -> отведения по колонкам.
GRID_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM = "II"

MM_PER_S = 25.0     # скорость
MM_PER_MV = 10.0    # усиление
COL_SEC = 2.5       # длина клетки блока
COL_MM = MM_PER_S * COL_SEC   # 62.5 мм

# Геометрия листа (мм)
LEFT = 12.0
TOP = 16.0
ROW_H = 40.0        # высота строки (±2 мВ)
RHY_GAP = 12.0
CLIP_MV = 1.9       # ограничение амплитуды в клетке


def _draw_grid(ax, W, H):
    minor = "#f4c9d0"
    major = "#e79aa6"
    # мелкая сетка 1 мм
    for x in np.arange(0, W + 0.1, 1):
        ax.plot([x, x], [0, H], color=minor, lw=0.3, zorder=0)
    for y in np.arange(0, H + 0.1, 1):
        ax.plot([0, W], [y, y], color=minor, lw=0.3, zorder=0)
    # крупная 5 мм
    for x in np.arange(0, W + 0.1, 5):
        ax.plot([x, x], [0, H], color=major, lw=0.6, zorder=0)
    for y in np.arange(0, H + 0.1, 5):
        ax.plot([0, W], [y, y], color=major, lw=0.6, zorder=0)


def _plot_trace(ax, sig, x0, baseline, seconds, fs):
    n = int(seconds * fs)
    s = np.nan_to_num(sig[:n]).astype(float)
    s = np.clip(s, -CLIP_MV, CLIP_MV)
    t = np.arange(len(s)) / fs
    x = x0 + t * MM_PER_S
    y = baseline - s * MM_PER_MV      # вверх = положительно (ось y инвертируем ниже)
    ax.plot(x, y, color="black", lw=0.9, zorder=3, solid_joinstyle="round")


def _cal_pulse(ax, x0, baseline):
    """Калибр-импульс 1 мВ (10 мм) шириной 5 мм перед строкой."""
    xs = [x0 - 8, x0 - 8, x0 - 3, x0 - 3]
    ys = [baseline, baseline - MM_PER_MV, baseline - MM_PER_MV, baseline]
    ax.plot(xs, ys, color="black", lw=0.9, zorder=3)


def render(sig_path, out_path, fs=500, title="ECG1.2 — цифровая реконструкция"):
    mat = np.load(sig_path)  # (N, 12)
    data = {lead: mat[:, j] for j, lead in enumerate(LEAD_ORDER)}

    W = LEFT + 4 * COL_MM + 6
    n_rows = 3
    H = TOP + n_rows * ROW_H + RHY_GAP + ROW_H + 6

    fig, ax = plt.subplots(figsize=(W / 25.4, H / 25.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    _draw_grid(ax, W, H)

    # блок 3x4
    for r, row in enumerate(GRID_LAYOUT):
        baseline = TOP + r * ROW_H + ROW_H / 2
        _cal_pulse(ax, LEFT, baseline)
        for c, lead in enumerate(row):
            x0 = LEFT + c * COL_MM
            if c > 0:  # тонкий разделитель колонок
                ax.plot([x0, x0], [baseline - ROW_H / 2 + 4, baseline + ROW_H / 2 - 4],
                        color="black", lw=0.5, zorder=2)
            if lead in data:
                _plot_trace(ax, data[lead], x0 + 2, baseline, COL_SEC, fs)
            ax.text(x0 + 3, baseline - ROW_H / 2 + 5, lead, fontsize=9,
                    fontweight="bold", zorder=4)

    # ритм-строка (II, 10с на всю ширину)
    ry = TOP + n_rows * ROW_H + RHY_GAP + ROW_H / 2
    _cal_pulse(ax, LEFT, ry)
    _plot_trace(ax, data[RHYTHM], LEFT + 2, ry, 10.0, fs)
    ax.text(LEFT + 3, ry - ROW_H / 2 + 5, RHYTHM, fontsize=9, fontweight="bold", zorder=4)

    # подписи скорости/усиления
    ax.text(LEFT, H - 3, "25 mm/s    10 mm/mV", fontsize=8, color="#333")
    ax.set_title(f"{title}  (demo, не медизделие)", fontsize=11, pad=8)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)   # инверсия: 0 сверху
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="signal.npy (N x 12)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--fs", type=int, default=500)
    a = ap.parse_args()
    render(a.input, a.output, a.fs)
