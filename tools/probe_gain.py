"""Откуда берётся систематическая ошибка амплитуды: шкала или срезание вершин.

На PTB-XL стенд показывает усиление ~0.94 — амплитуды стабильно ниже эталона.
Причин две, и лечатся они по-разному:

  промах по шкале   — движок неверно измерил шаг сетки, все амплитуды
                      уезжают одинаково, лечится калибровкой;
  срезание вершин   — маска линии толстая, из неё берётся центр, и острая
                      вершина сглаживается; тупые формы при этом целы.

Опыт: печатаем треугольные пики РАЗНОЙ ширины, все ровно 1 мВ, и смотрим,
что вернётся. Ровный недобор на всех ширинах — шкала. Падение амплитуды по
мере сужения пика — вершины.

Прямоугольные импульсы для этого не годятся: их вертикальные грани — не линия,
а вертикальный штрих в 10 мм, трассировщик на них рвётся и выбрасывает фигуру
целиком (проверено). Отсюда, кстати, следует, что и настоящий калибр-импульс
на плёнке движок читать не станет.

    .venv/bin/python tools/probe_gain.py          # посчитать по готовому прогону
    .venv/bin/python tools/probe_gain.py --again  # перепечатать и оцифровать заново
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import oecg_digitize as od          # noqa: E402
import render_digital_ecg as rd     # noqa: E402
from oecg_render import load_csv    # noqa: E402

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
AMP = 1.0                            # мВ — ровно 10 мм на бумаге
WIDTHS_MS = [20, 40, 80, 160]        # полная ширина основания треугольника
PERIOD_S = 0.4                       # шесть пиков на клетку 2.5 с


def probe_signal() -> np.ndarray:
    """(5000, 12) мВ: треугольники разной ширины, вершина ровно ±1 мВ."""
    n = 10 * FS
    x = np.zeros(n)
    k = 0
    t = 0.0
    while t + PERIOD_S <= 10.0:
        w = WIDTHS_MS[k % len(WIDTHS_MS)]
        sign = 1.0 if k % 2 == 0 else -1.0
        half = max(1, int(w / 2000 * FS))
        c = int((t + PERIOD_S / 2) * FS)
        j = np.arange(-half, half + 1)
        x[c - half:c + half + 1] = sign * AMP * (1 - np.abs(j) / (half + 1))
        k += 1
        t += PERIOD_S
    return np.tile(x[:, None], (1, 12))


def _png_width(path) -> int:
    from PIL import Image
    with Image.open(path) as im:
        return im.size[0]


def runs_of(col: np.ndarray) -> list[tuple[int, int]]:
    """Непрерывные куски отведения без пропусков."""
    ok = ~np.isnan(col)
    d = np.diff(np.concatenate([[0], ok.view(np.int8), [0]]))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1),
                                             np.flatnonzero(d == -1)) if b - a > 30]


def peaks_in(seg: np.ndarray, fs: int) -> list[tuple[float, float]]:
    """Пики куска -> список (ширина на полувысоте в мс, амплитуда в мВ).

    Привязка ко времени не нужна: эталонная амплитуда у всех пиков одна и та же,
    а ширину каждого меряем по нему самому.
    """
    x = seg - np.median(seg)
    top = float(np.percentile(np.abs(x), 99))
    if top < 0.3:
        return []
    ok = np.abs(x) > 0.25 * top
    d = np.diff(np.concatenate([[0], ok.view(np.int8), [0]]))
    out = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if a == 0 or b >= len(x):                  # обрезан краем куска
            continue
        part = np.abs(x[a:b])
        amp = float(part.max())
        if amp < 0.3:
            continue
        half_w = 1000.0 * float(np.count_nonzero(part > amp / 2)) / fs
        out.append((half_w, amp))
    return out


def main():
    work = ROOT / "output" / "probe_gain"
    work.mkdir(parents=True, exist_ok=True)
    def opt(flag, default):
        return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default

    dpi = opt("--dpi", 200)
    target_w = opt("--target", od.TARGET_W)
    npy = work / "probe.npy"
    png = work / f"probe_{dpi}.png"
    np.save(npy, probe_signal())
    rd.CLIP_MV = 3.0
    rd.DPI = dpi
    rd.render(str(npy), str(png), fs=FS, title="проба амплитуды")

    # Внутренний размер сети — подозреваемый номер один: вершина пика занимает
    # считаные пиксели, и чем их меньше, тем сильнее её срезает.
    resample = opt("--resample", od.RESAMPLE)
    out = work / f"run_{dpi}_{target_w}_{resample}"
    if not list(out.glob("*_timeseries_canonical.csv")) or "--again" in sys.argv:
        od.digitize(str(png), str(out), layout="standard_3x4_with_r1",
                    resample=resample, target_w=target_w)
    px_mm = 5 * dpi / 25.4 / 5 * min(1.0, target_w / max(1, _png_width(png)))
    print(f"печать {dpi} dpi, вход {target_w} px, сеть {resample} "
          f"-> {px_mm:.1f} px/мм")
    csvs = list(out.glob("*_timeseries_canonical.csv"))
    if not csvs:
        raise SystemExit("движок не вернул сигнал")
    ours, names = load_csv(str(csvs[0]))

    found = []
    for lead in LEAD_ORDER:
        if lead not in names:
            continue
        col = ours[:, names.index(lead)]
        for a, b in runs_of(col):
            found += peaks_in(col[a:b], 1000)
    if not found:
        raise SystemExit("пики не найдены — смотри output/probe_gain/run/*_twin.png")

    # у треугольника ширина на полувысоте = половина основания
    nominal = [w / 2 for w in WIDTHS_MS]
    print(f"{'ширина пика':>12s} {'найдено':>8s} {'амплитуда':>10s} {'ошибка':>8s}")
    for w, nom in zip(WIDTHS_MS, nominal):
        amps = [a for hw, a in found
                if min(nominal, key=lambda m: abs(m - hw)) == nom]
        if not amps:
            print(f"{w:10d} мс {'—':>8s}")
            continue
        med = float(np.median(amps))
        print(f"{w:10d} мс {len(amps):8d} {med:9.3f} мВ {100*(med-AMP)/AMP:+7.1f}%")

    all_amp = [a for _, a in found]
    print(f"\nвсего пиков: {len(all_amp)}, медиана {np.median(all_amp):.3f} мВ")


if __name__ == "__main__":
    main()
