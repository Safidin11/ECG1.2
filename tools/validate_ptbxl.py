r"""Проверка точности оцифровки на открытой базе PTB-XL.

Идея простая: берём запись, для которой ТОЧНО знаем сигнал, печатаем из неё
бумажную плёнку, прогоняем нашим конвейером и сравниваем с оригиналом.

    эталон (WFDB, 500 Гц)  ->  картинка плёнки  ->  движок  ->  наш сигнал
                    \                                              /
                     \-------------- сравнение --------------------/

Что меряем по каждому отведению:
  покрытие  — какую долю удалось прочитать вообще;
  корреляция — совпадает ли форма;
  усиление  — общий масштаб, методом наименьших квадратов (1.00 = верно,
              0.50 = прочитали вдвое мельче, т.е. промах по калибровке);
  зубец R   — отношение ВЫСОТ зубцов R. Отдельно от усиления: клинические
              пороги заданы высотой зубца, и её занижение не видно ни по
              корреляции, ни по SNR;
  SNR, дБ   — отношение мощности сигнала к мощности ошибки. Даём два числа:
              как есть и после исправления усиления. Если второе заметно выше
              первого, значит форму берём правильно, а масштаб — нет.

Важные оговорки, без которых цифры врут:
  * Плёнку печатаем СВОИМ рендером — чистую, без бликов, смятия и перспективы.
    Это верхняя граница: на фотографии будет хуже.
  * Формат задаём движку жёстко, иначе ошибка распознавания раскладки
    перемешала бы отведения и утопила бы все остальные цифры.
  * Сэмплы, которые рендер обрезал по краю строки, из подсчёта исключаем —
    иначе мерили бы наш рендер, а не оцифровку.

Запуск:
    .venv/bin/python tools/validate_ptbxl.py -n 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import correlate, resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import oecg_digitize as od          # noqa: E402
import render_digital_ecg as rd     # noqa: E402
from ecg_measure import heart_rate as measure_hr   # noqa: E402
from ecg_measure import longest_run                # noqa: E402
from oecg_render import load_csv    # noqa: E402

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS_TRUTH = 500
FS_OURS = 1000                      # движок всегда отдаёт 10 000 отсчётов на 10 с
CLIP_MV = 3.0                       # предел рендера по амплитуде
LAYOUT = "standard_3x4_with_r1"
MAX_LAG = 400                       # поиск сдвига, отсчётов при 1000 Гц (±0.4 с)


def load_truth(hea_path: Path) -> np.ndarray | None:
    """Эталон -> (5000, 12) мВ в нашем порядке отведений."""
    rec = wfdb.rdrecord(str(hea_path.with_suffix("")))
    names = [n.strip() for n in rec.sig_name]
    alias = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
    names = [alias.get(n.upper(), n.upper() if n.upper().startswith("V") else n)
             for n in names]
    out = np.full((rec.p_signal.shape[0], 12), np.nan)
    for j, want in enumerate(LEAD_ORDER):
        if want in names:
            out[:, j] = rec.p_signal[:, names.index(want)]
    return out if not np.isnan(out).all() else None


def compare_lead(ours: np.ndarray, truth: np.ndarray) -> dict | None:
    """Наш кусок отведения против эталона того же отведения.

    Рендер печатает в каждой клетке НАЧАЛО отведения, поэтому эталон берём с
    нулевого отсчёта, а точный сдвиг (поля, толщина линии) находим взаимной
    корреляцией — иначе пара миллиметров смещения обвалила бы SNR на ровном месте.
    """
    n = len(ours)
    if n < FS_OURS // 2:                       # меньше полусекунды — не считаем
        return None
    a = ours - np.median(ours)
    pad = np.concatenate([np.zeros(MAX_LAG), truth[:n + MAX_LAG]])
    if len(pad) < n + MAX_LAG + 1:
        pad = np.concatenate([pad, np.zeros(n + MAX_LAG + 1 - len(pad))])
    # Сдвиг ищем по НОРМИРОВАННОЙ взаимной корреляции. Без нормировки выигрывает
    # положение, где под окно попал кусок с самой большой энергией, а не тот, где
    # формы совпали: на записи с одним высоким комплексом сдвиг уезжал к нему.
    p2 = pad - np.median(pad)
    e = np.sqrt(np.convolve(p2 ** 2, np.ones(n), mode="valid"))
    c = correlate(p2, a, mode="valid") / np.maximum(e, 1e-12)
    idx = int(np.argmax(c))
    t = pad[idx:idx + n]
    t = t - np.median(t)

    keep = np.abs(t) < CLIP_MV - 0.1           # обрезанное рендером не считаем
    if keep.sum() < n // 2:
        return None
    a, t = a[keep], t[keep]
    if np.std(t) < 1e-6:
        return None

    err = float(np.sum((a - t) ** 2))
    sig = float(np.sum(t ** 2))
    # Усиление — методом наименьших квадратов, а НЕ отношением разбросов.
    # Разброс складывается из сигнала и шума, поэтому шум оцифровки сам по себе
    # поднимает std(a) и «усиление» получается больше единицы даже там, где
    # амплитуды переданы верно: после восстановления вершин отношение разбросов
    # показывало 1.025, тогда как настоящая высота зубца R была 0.992 от эталона.
    # У наименьших квадратов шум не коррелирован с эталоном и смещения не даёт.
    gain = float(a @ t / (t @ t))
    err_g = float(np.sum((a / gain - t) ** 2)) if abs(gain) > 1e-6 else np.inf
    return {
        "corr": float(np.corrcoef(a, t)[0, 1]),
        "gain": gain,
        "snr": 10 * np.log10(sig / err) if err > 0 else np.inf,
        "snr_gain_fixed": 10 * np.log10(sig / err_g) if err_g > 0 else np.inf,
        "rms_uv": float(np.sqrt(np.mean((a - t) ** 2)) * 1000),
        "lag_ms": (idx - MAX_LAG) * 1000.0 / FS_OURS,
        "n": int(len(a)),
        "r_amp": r_amp_ratio(a, t, FS_OURS),
    }


def r_amp_ratio(a: np.ndarray, t: np.ndarray, fs: int) -> float | None:
    """Отношение высот зубцов R — то, что на самом деле читают с плёнки.

    Отдельно от усиления, потому что клинические пороги (гипертрофия, критерии
    инфаркта) заданы именно высотой зубца, а не среднеквадратичным масштабом:
    занижение вершин на 10% не видно ни по корреляции, ни по SNR, но сдвигает
    именно эти пороги.
    """
    from ecg_measure import r_peaks               # noqa: PLC0415
    pk = r_peaks(t, fs)
    half = int(0.05 * fs)
    rat = []
    for p in pk:
        lo, hi = max(0, p - half), min(len(t), p + half)
        vt = float(np.max(np.abs(t[lo:hi])))
        if vt > 0.15:                            # мелкие зубцы отношение шумят
            rat.append(float(np.max(np.abs(a[lo:hi]))) / vt)
    return float(np.median(rat)) if len(rat) >= 2 else None


def heart_rate(x: np.ndarray, fs: int) -> float | None:
    """ЧСС, уд/мин. Считает тот же код, что и на сайте — иначе проверка
    перестала бы что-либо значить (tools/ecg_measure.py)."""
    r = measure_hr(x, fs)
    return float(r["bpm"]) if r else None


def run_one(hea: Path, work: Path, reuse: bool = False) -> dict | None:
    truth = load_truth(hea)
    if truth is None:
        return None
    name = hea.stem
    out = work / name
    csvs = list(out.glob("*_timeseries_canonical.csv"))
    if not (reuse and csvs):
        npy = work / f"{name}.npy"
        png = work / f"{name}.png"
        np.save(npy, truth)
        rd.CLIP_MV = CLIP_MV
        rd.render(str(npy), str(png), fs=FS_TRUTH, title=f"PTB-XL {name}")
        # Печать в разном разрешении = разная детализация входа. Конвейер мелкие
        # снимки РАСТЯГИВАЕТ до TARGET_W, поэтому так воспроизводится ровно тот
        # случай, на котором ломается реальное фото: размер большой, деталей мало.
        od.digitize(str(png), str(out), layout=LAYOUT)
        csvs = list(out.glob("*_timeseries_canonical.csv"))
    if not csvs:
        return {"record": name, "failed": "движок не вернул сигнал"}
    ours, names = load_csv(str(csvs[0]))

    truth_1k = resample_poly(np.nan_to_num(truth), FS_OURS // FS_TRUTH, 1, axis=0)
    res, cover = {}, {}
    for j, lead in enumerate(LEAD_ORDER):
        if lead not in names:
            continue
        col = ours[:, names.index(lead)]
        cover[lead] = float(np.mean(~np.isnan(col)))
        span = longest_run(col)
        if span is None:
            continue
        m = compare_lead(col[span[0]:span[1]], truth_1k[:, j])
        if m:
            res[lead] = m

    hr_t = heart_rate(truth_1k[:, 1], FS_OURS)                    # эталонное II
    best = max(cover, key=cover.get) if cover else None
    hr_o = heart_rate(ours[:, names.index(best)], FS_OURS) if best else None
    return {"record": name, "leads": res, "coverage": cover,
            "hr_truth": hr_t, "hr_ours": hr_o}


def summarize(rows: list[dict]) -> None:
    ok = [r for r in rows if r.get("leads")]
    if not ok:
        print("нечего сводить")
        return
    print(f"\n{'отв.':6s} {'покрытие':>9s} {'корр.':>7s} {'усил.':>7s} "
          f"{'зубец R':>8s} {'SNR дБ':>8s} {'SNR испр':>9s} {'RMS мкВ':>9s}")
    for lead in LEAD_ORDER:
        vals = [r["leads"][lead] for r in ok if lead in r["leads"]]
        cov = [r["coverage"].get(lead, 0) for r in ok]
        if not vals:
            print(f"{lead:6s} {100*np.mean(cov):8.0f}% {'—':>7s}")
            continue
        med = lambda k: np.median([v[k] for v in vals])      # noqa: E731
        ra = [v["r_amp"] for v in vals if v.get("r_amp")]
        print(f"{lead:6s} {100*np.mean(cov):8.0f}% {med('corr'):7.3f} "
              f"{med('gain'):7.2f} {(f'{np.median(ra):.3f}' if ra else '—'):>8s} "
              f"{med('snr'):8.1f} {med('snr_gain_fixed'):9.1f} {med('rms_uv'):9.0f}")

    everything = [v for r in ok for v in r["leads"].values()]
    ra = [v["r_amp"] for v in everything if v.get("r_amp")]
    print(f"\nвсего записей: {len(rows)}, с сигналом: {len(ok)}, "
          f"отведений сравнено: {len(everything)}")
    print(f"медиана: корреляция {np.median([v['corr'] for v in everything]):.3f}, "
          f"усиление {np.median([v['gain'] for v in everything]):.3f}, "
          f"SNR {np.median([v['snr'] for v in everything]):.1f} дБ "
          f"(после исправления усиления {np.median([v['snr_gain_fixed'] for v in everything]):.1f} дБ)")
    if ra:
        print(f"высота зубца R к эталону: {np.median(ra):.3f} "
              f"(в пределах 5% — {100*np.mean([abs(x-1) < 0.05 for x in ra]):.0f}% отведений)")

    hr = [(r["hr_truth"], r["hr_ours"]) for r in ok
          if r.get("hr_truth") and r.get("hr_ours")]
    if hr:
        err = [abs(a - b) for a, b in hr]
        print(f"ЧСС: сравнено {len(hr)}, медиана ошибки {np.median(err):.1f} уд/мин, "
              f"в пределах 5 уд/мин — {100*np.mean([e < 5 for e in err]):.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=40)
    ap.add_argument("--data", default=str(ROOT / "data" / "ptbxl"))
    ap.add_argument("-o", "--out", default=str(ROOT / "output" / "validate"))
    ap.add_argument("--fresh", action="store_true", help="считать заново, не докатывать")
    ap.add_argument("--reuse", action="store_true",
                    help="не гонять движок заново, взять уже готовые CSV — для пересчёта метрик")
    ap.add_argument("--dpi", type=int, default=rd.DPI,
                    help="разрешение печати плёнки: 200 dpi ≈ 7.9 px/мм")
    a = ap.parse_args()

    rd.DPI = a.dpi
    work = Path(a.out if a.dpi == 200 else f"{a.out}_dpi{a.dpi}")
    work.mkdir(parents=True, exist_ok=True)
    print(f"печать {a.dpi} dpi = {a.dpi / 25.4:.1f} px/мм детализации")
    heas = sorted(Path(a.data).glob("*.hea"))[:a.count]
    if not heas:
        raise SystemExit(f"нет записей в {a.data} — сначала скачай PTB-XL")

    # Досчитываем с того места, где остановились: прогон долгий, терять уже
    # посчитанное из-за перезапуска незачем.
    rows, jsonl, done = [], work / "results.jsonl", set()
    if a.fresh or not jsonl.exists():
        jsonl.write_text("", encoding="utf-8")
    else:
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
                done.add(rows[-1]["record"])
        if done:
            print(f"уже посчитано {len(done)} — пропускаю их")
    for i, hea in enumerate(heas, 1):
        if hea.stem in done:
            continue
        try:
            r = run_one(hea, work, reuse=a.reuse)
        except Exception as exc:
            # Не обрезаем в лог: из 120 символов диагноза не собрать, а падения
            # движка — самое частое, что тут приходится разбирать.
            print(f"--- {hea.stem} упал ---\n{exc}\n", flush=True)
            r = {"record": hea.stem, "failed": str(exc)[-400:]}
        if r is None:
            continue
        rows.append(r)
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_ok = len(r.get("leads", {}))
        print(f"[{i}/{len(heas)}] {r['record']}: "
              f"{r.get('failed') or f'{n_ok} отведений сравнено'}", flush=True)

    summarize(rows)
    print(f"\nподробности: {jsonl}")


if __name__ == "__main__":
    main()
