"""Результат нового движка (CSV 12 отведений) -> цифровая ЭКГ на миллиметровке.

Движок Open-ECG-Digitizer отдаёт `<имя>_timeseries_canonical.csv`: 10 000
отсчётов × 12 отведений в микровольтах (10 с при 1000 Гц), пропуски = пусто.
Здесь мы читаем его и рисуем нашим рендером в стандартной раскладке.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 1000                       # движок отдаёт 10 000 отсчётов на 10 с


def load_csv(csv_path: str) -> tuple[np.ndarray, list[str]]:
    """CSV движка -> (N, 12) в МИЛЛИвольтах + имена отведений."""
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    header = rows[0]
    data = np.array([[float(x) if x not in ("", "nan", "NaN") else np.nan for x in r]
                     for r in rows[1:]], dtype=np.float64)
    return data / 1000.0, header          # мкВ -> мВ


def coverage(sig: np.ndarray, leads: list[str]) -> dict[str, float]:
    return {name: float((~np.isnan(sig[:, i])).mean()) for i, name in enumerate(leads)}


def render(csv_path: str, out_png: str, grid=None, cols: int | None = None) -> str:
    """Нарисовать цифровую ЭКГ из CSV движка."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_digital_ecg import render as render_signal      # noqa: E402

    sig, leads = load_csv(csv_path)
    # наш рендер ждёт .npy (N,12) в мВ и порядок LEAD_ORDER
    order = [leads.index(n) if n in leads else None for n in LEAD_ORDER]
    out = np.full((sig.shape[0], 12), np.nan)
    for j, src in enumerate(order):
        if src is None:
            continue
        col = sig[:, src]
        # Движок хранит отведение на его МЕСТЕ во времени (aVR = 2.5-5 c),
        # а рендер рисует с начала клетки -> сдвигаем каждое к нулю.
        valid = np.flatnonzero(~np.isnan(col))
        if len(valid) == 0:
            continue
        seg = col[valid[0]:valid[-1] + 1]
        out[:len(seg), j] = seg
    tmp_npy = Path(out_png).with_suffix(".npy")
    np.save(tmp_npy, out)
    render_signal(str(tmp_npy), out_png, fs=FS, grid=grid, cols=cols)
    tmp_npy.unlink(missing_ok=True)
    return out_png


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--csv", required=True)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    print(render(a.csv, a.out))
