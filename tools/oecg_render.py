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


LAYOUTS_YML = Path(__file__).resolve().parent.parent / "configs" / "oecg_layouts.yml"


def grid_for(layout_name: str, cov: dict[str, float] | None = None):
    """Формат движка -> (сетка, число колонок) для нашего рендера.

    В configs/oecg_layouts.yml сетка задана явно (`leads` = список строк).
    Ритм-строки идут отдельным списком `rhythm_leads`; "Any" означает «любое
    отведение», поэтому конкретное берём по покрытию: у ритм-строки она на всю
    ширину (~100%), у обычной клетки — доля 1/cols.
    """
    import yaml
    layouts = yaml.safe_load(LAYOUTS_YML.read_text(encoding="utf-8")) or {}
    lay = layouts.get(layout_name)
    if not lay:
        return None, None
    # `leads` бывает списком строк (3×4: строка = список отведений) и плоским
    # списком (12×1: строка = одно отведение) — нормализуем к списку строк.
    grid = [[row] if isinstance(row, str) else list(row) for row in lay.get("leads", [])]
    cols = int(lay.get("layout", {}).get("cols", max((len(r) for r in grid), default=1)))
    for _ in lay.get("rhythm_leads", []) or []:
        name = "II"
        if cov:                                  # ритм = самое «длинное» отведение
            best = max(cov, key=lambda k: cov[k])
            if cov[best] > 0.6:
                name = best
        grid.append([name] * max(cols, 1))
    return grid, cols


def render(csv_path: str, out_png: str, grid=None, cols: int | None = None,
           layout: str | None = None, sig=None, leads=None) -> str:
    """Нарисовать цифровую ЭКГ из CSV движка.

    layout — имя формата движка; тогда сетка берётся из него (иначе рендер
    рисует стандартную 3×4+ритм независимо от реального формата снимка).
    sig/leads — уже готовый сигнал (напр. после восстановления по связям
    отведений); тогда csv_path не читается.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_digital_ecg import render as render_signal      # noqa: E402

    if sig is None:
        sig, leads = load_csv(csv_path)
    if grid is None and layout:
        grid, cols = grid_for(layout, coverage(sig, leads))
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
    ap.add_argument("--layout", default=None, help="формат движка (для правильной сетки)")
    a = ap.parse_args()
    print(render(a.csv, a.out, layout=a.layout))
