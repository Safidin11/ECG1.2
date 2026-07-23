"""Рендер цифровой ЭКГ на «миллиметровке» в ИСХОДНОЙ раскладке.

Рисует сигнал в той же раскладке, что была на входе (3×4+ритм, 6×2+ритм,
12×1 и т.п.) — по grid из стадии layout. Стандарт: розовая сетка (1/5 мм),
25 мм/с, 10 мм/мВ, метки отведений, калибр-импульс на строку.

Строка grid, где все отведения одинаковы (и колонок > 1) — ритм-строка на всю
ширину (10с). Иначе каждая клетка показывает 10с/cols. Раскладка 12×1 (cols=1)
рисуется как 12 строк на всю ширину.

Demo-инструмент, НЕ медизделие.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEFAULT_GRID = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
                ["III", "aVF", "V3", "V6"], ["II", "II", "II", "II"]]

MM_PER_S = 25.0
MM_PER_MV = 10.0
PAPER_SEC = 10.0
LEFT = 12.0
TOP = 16.0
ROW_H = 40.0
CLIP_MV = 1.9


def _draw_grid(ax, W, H):
    minor, major = "#f4c9d0", "#e79aa6"
    for x in np.arange(0, W + 0.1, 1):
        ax.plot([x, x], [0, H], color=minor, lw=0.3, zorder=0)
    for y in np.arange(0, H + 0.1, 1):
        ax.plot([0, W], [y, y], color=minor, lw=0.3, zorder=0)
    for x in np.arange(0, W + 0.1, 5):
        ax.plot([x, x], [0, H], color=major, lw=0.6, zorder=0)
    for y in np.arange(0, H + 0.1, 5):
        ax.plot([0, W], [y, y], color=major, lw=0.6, zorder=0)


def _plot_trace(ax, sig, x0, baseline, seconds, fs):
    n = int(seconds * fs)
    s = np.clip(np.nan_to_num(sig[:n]).astype(float), -CLIP_MV, CLIP_MV)
    t = np.arange(len(s)) / fs
    ax.plot(x0 + t * MM_PER_S, baseline - s * MM_PER_MV, color="black",
            lw=0.9, zorder=3, solid_joinstyle="round")


def _cal_pulse(ax, x0, baseline):
    ax.plot([x0 - 8, x0 - 8, x0 - 3, x0 - 3],
            [baseline, baseline - MM_PER_MV, baseline - MM_PER_MV, baseline],
            color="black", lw=0.9, zorder=3)


def render(sig_path, out_path, fs=500, grid=None, cols=None,
           title="ECG1.2 — цифровая реконструкция"):
    mat = np.load(sig_path) if isinstance(sig_path, str) else sig_path
    data = {lead: mat[:, j] for j, lead in enumerate(LEAD_ORDER) if j < mat.shape[1]}
    if grid is None:
        grid, cols = DEFAULT_GRID, 4
    if cols is None:
        cols = max(len(r) for r in grid)
    rows = len(grid)
    sec_per_col = PAPER_SEC / cols
    col_mm = MM_PER_S * sec_per_col

    W = LEFT + cols * col_mm + 6
    H = TOP + rows * ROW_H + 6
    fig, ax = plt.subplots(figsize=(W / 25.4, H / 25.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    _draw_grid(ax, W, H)

    for r, row in enumerate(grid):
        baseline = TOP + r * ROW_H + ROW_H / 2
        _cal_pulse(ax, LEFT, baseline)
        is_rhythm = cols > 1 and all(l == row[0] for l in row)
        if is_rhythm:
            if row[0] in data:
                _plot_trace(ax, data[row[0]], LEFT + 2, baseline, PAPER_SEC, fs)
            ax.text(LEFT + 3, baseline - ROW_H / 2 + 5, row[0], fontsize=9,
                    fontweight="bold", zorder=4)
        else:
            for c, lead in enumerate(row):
                x0 = LEFT + c * col_mm
                if c > 0:
                    ax.plot([x0, x0], [baseline - ROW_H / 2 + 4, baseline + ROW_H / 2 - 4],
                            color="black", lw=0.5, zorder=2)
                if lead in data:
                    _plot_trace(ax, data[lead], x0 + 2, baseline, sec_per_col, fs)
                ax.text(x0 + 3, baseline - ROW_H / 2 + 5, lead, fontsize=9,
                        fontweight="bold", zorder=4)

    ax.text(LEFT, H - 3, "25 mm/s    10 mm/mV", fontsize=8, color="#333")
    ax.set_title(f"{title}  (demo, не медизделие)", fontsize=11, pad=8)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
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
