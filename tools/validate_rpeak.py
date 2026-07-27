"""Насколько занижается амплитуда зубца R — на настоящих ЭКГ, а не на пробах.

Опыт с треугольниками (tools/probe_gain.py) показал, что чем острее вершина,
тем сильнее её срезает. Но у него есть слабое место: треугольник в 20 мс при
печати 200 dpi занимает 4 пикселя в основании, и непонятно, что мы меряем —
оцифровку или неспособность рендера такое напечатать.

Здесь возражение снято: меряем ровно ту величину, которая нужна в деле —
высоту зубца R на настоящих записях PTB-XL. Зубцы ищем в ЭТАЛОНЕ (там они
известны точно), а высоту берём в обоих сигналах в одной и той же точке.

Оцифровка уже сделана стендом validate_ptbxl.py, GPU не нужен:

    .venv/bin/python tools/validate_rpeak.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, correlate, filtfilt, find_peaks, resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from oecg_render import load_csv                          # noqa: E402
from validate_ptbxl import LEAD_ORDER, load_truth, longest_run   # noqa: E402

FS_TRUTH, FS_OURS = 500, 1000
MAX_LAG = 400


def r_peaks(x: np.ndarray, fs: int) -> np.ndarray:
    """Положения комплексов, схема Пана-Томпкинса (полоса 5-15 Гц)."""
    b, a = butter(3, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype="band")
    y = filtfilt(b, a, x - np.median(x))
    y = np.diff(y, prepend=y[0]) ** 2
    w = max(1, int(0.15 * fs))
    y = np.convolve(y, np.ones(w) / w, mode="same")
    pk, _ = find_peaks(y, height=0.3 * np.percentile(y, 98), distance=int(0.28 * fs))
    return pk


def main():
    work = ROOT / "output" / "validate"
    pairs = []
    for d in sorted(p for p in work.iterdir() if p.is_dir()):
        hea = ROOT / "data" / "ptbxl" / f"{d.name}.hea"
        csvs = list(d.glob("*_timeseries_canonical.csv"))
        if not hea.exists() or not csvs:
            continue
        truth = load_truth(hea)
        if truth is None:
            continue
        truth = resample_poly(np.nan_to_num(truth), FS_OURS // FS_TRUTH, 1, axis=0)
        ours, names = load_csv(str(csvs[0]))

        for j, lead in enumerate(LEAD_ORDER):
            if lead not in names:
                continue
            span = longest_run(ours[:, names.index(lead)])
            if span is None or span[1] - span[0] < FS_OURS:
                continue
            seg = ours[span[0]:span[1], names.index(lead)]
            a = seg - np.median(seg)
            # Настоящая ЭКГ не периодична, поэтому сдвиг определяется однозначно.
            pad = np.concatenate([np.zeros(MAX_LAG), truth[:len(seg) + MAX_LAG, j]])
            if len(pad) < len(seg) + MAX_LAG + 1:
                pad = np.concatenate([pad, np.zeros(len(seg) + MAX_LAG + 1 - len(pad))])
            idx = int(np.argmax(correlate(pad - np.median(pad), a, mode="valid")))
            t = pad[idx:idx + len(seg)]
            t = t - np.median(t)
            if np.std(t) < 1e-6:
                continue

            for p in r_peaks(t, FS_OURS):
                if p < 30 or p > len(t) - 30:
                    continue
                # вершина: экстремум по модулю в окне ±30 мс вокруг комплекса
                w_t = t[p - 30:p + 30]
                w_o = a[p - 30:p + 30]
                k = int(np.argmax(np.abs(w_t)))
                amp_t = float(w_t[k])
                if abs(amp_t) < 0.2:                 # мелочь не считаем
                    continue
                # у нас берём экстремум того же знака рядом — сдвиг вершины на
                # пару отсчётов не должен считаться потерей амплитуды
                near = w_o[max(k - 8, 0):k + 9]
                amp_o = float(near.max() if amp_t > 0 else near.min())
                pairs.append((abs(amp_t), abs(amp_o), lead))

    if not pairs:
        raise SystemExit("нет данных — сначала прогони tools/validate_ptbxl.py")

    t_all = np.array([p[0] for p in pairs])
    o_all = np.array([p[1] for p in pairs])
    ratio = o_all / t_all
    print(f"комплексов измерено: {len(pairs)} в {len(set(p[2] for p in pairs))} отведениях")
    print(f"\nамплитуда зубца R: медиана нашей / эталонной = {np.median(ratio):.3f} "
          f"({100*(np.median(ratio)-1):+.1f}%)")
    print(f"разброс: 25-й перцентиль {np.percentile(ratio,25):.3f}, "
          f"75-й {np.percentile(ratio,75):.3f}")
    print(f"медиана недобора в мВ: {np.median(t_all-o_all):.3f} "
          f"(при медианной высоте зубца {np.median(t_all):.2f} мВ)")

    print(f"\n{'высота зубца':>14s} {'шт':>6s} {'наша/эталон':>12s}")
    edges = [0.2, 0.5, 1.0, 1.5, 2.5, 10.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t_all >= lo) & (t_all < hi)
        if m.sum() < 5:
            continue
        print(f"{lo:5.1f}-{hi:<4.1f} мВ {m.sum():6d} {np.median(ratio[m]):12.3f}")

    print(f"\n{'отв.':6s} {'шт':>6s} {'наша/эталон':>12s}")
    for lead in LEAD_ORDER:
        m = np.array([p[2] == lead for p in pairs])
        if m.sum() < 5:
            continue
        print(f"{lead:6s} {m.sum():6d} {np.median(ratio[m]):12.3f}")


if __name__ == "__main__":
    main()
