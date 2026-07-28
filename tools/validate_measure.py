r"""Проверка измерений против разметки медицинского анализатора.

Эталон — PTB-XL+, где для тех же записей PTB-XL выложены интервалы, посчитанные
анализатором University of Glasgow. Значит точность можно назвать в
миллисекундах, а не «на глаз похоже».

Меряем в два этапа, и это принципиально: иначе не понять, чья ошибка.

  этап A: эталонный сигнал ->              наши измерения -> сравнение
          (одновременные отведения, 10 с, ничего не терялось)
          = ошибка САМОГО АЛГОРИТМА разметки

  этап B: эталонный сигнал -> плёнка -> движок -> наши измерения -> сравнение
          = ошибка ВСЕЙ цепочки

Разница B - A — цена оцифровки: потерянные детали плюс то, что в раскладке 3x4
отведения сняты не одновременно и их приходится совмещать.

Допуски взяты из практики: разброс между двумя разными фирменными анализаторами
на одной и той же записи сам по себе десятки миллисекунд, поэтому требовать
совпадения до миллисекунды бессмысленно.

Запуск:
    .venv/bin/python tools/validate_measure.py                # оба этапа
    .venv/bin/python tools/validate_measure.py --calibrate    # подбор порогов
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import ecg_delineate as dl              # noqa: E402
from oecg_render import load_csv        # noqa: E402

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS_TRUTH = 500
FS_OURS = 1000

# допуск, в пределах которого измерение считаем совпавшим
TOL = {"pr_ms": 20.0, "qrs_ms": 12.0, "qt_ms": 25.0, "axis": 15.0}


# Два независимых эталона на одних и тех же записях: анализаторы University of
# Glasgow и GE 12SL. Второй нужен не для точности, а для масштаба: расхождение
# между двумя фирменными приборами показывает, что вообще значит «совпало».
# Требовать от себя большего, чем они сходятся между собой, бессмысленно.
SOURCES = {
    "glasgow": ("unig_features.csv", "QRS_AxisFront_Global", "ST_Amp80ms_{}"),
    "12sl": ("12sl_features.csv", "R_AxisFrontal_Global", None),
}


def truth_table(path: Path, axis_col: str, st_col: str | None) -> dict[int, dict]:
    """Разметка анализатора: ecg_id -> нужные нам поля.

    Значения в файле уже в мс/мВ/градусах — множители из feature_description
    применять не надо, иначе всё уедет в тысячу раз.
    """
    out = {}
    with path.open(encoding="utf-8-sig") as f:
        for d in csv.DictReader(f):
            try:
                i = int(d["ecg_id"])
            except (KeyError, ValueError):
                continue
            num = lambda k: (float(d[k]) if d.get(k) not in (None, "", "None") else None)  # noqa: E731
            # Ось сверяем не всегда: Glasgow отдельным флагом отмечает записи,
            # где сам не смог её определить (комплекс почти изоэлектричный во
            # фронтальной плоскости). Там в поле лежит заглушка, и сравнение с
            # ней меряло бы не нашу ошибку, а чужую отписку.
            indet = num("QRS_AxisIndet_Global")
            out[i] = {
                "pr_ms": num("PR_Int_Global"),
                "qrs_ms": num("QRS_Dur_Global"),
                "qt_ms": num("QT_Int_Global"),
                "axis": None if indet else num(axis_col),
                "hr": num("HR__Global"),
                "rr_ms": num("RR_Mean_Global"),
                "st": {ld: (num(st_col.format(ld)) if st_col else None)
                       for ld in LEAD_ORDER},
            }
    return out


def analyzer_spread(a: dict[int, dict], b: dict[int, dict],
                    ids: list[int]) -> list[dict]:
    """Расхождение двух эталонов между собой — на тех же записях."""
    rows = []
    for i in ids:
        if i in a and i in b:
            rows.append(diffs({**a[i], "axis": {"qrs": a[i]["axis"]}, "flags": []},
                              b[i]))
    return rows


def load_truth(hea: Path) -> tuple[np.ndarray, list[str]] | None:
    rec = wfdb.rdrecord(str(hea.with_suffix("")))
    alias = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
    names = [alias.get(n.strip().upper(), n.strip()) for n in rec.sig_name]
    names = [n if n in LEAD_ORDER else n.upper() for n in names]
    keep = [(i, n) for i, n in enumerate(names) if n in LEAD_ORDER]
    if len(keep) < 8:
        return None
    return rec.p_signal[:, [i for i, _ in keep]], [n for _, n in keep]


def diffs(ours: dict, want: dict) -> dict[str, float]:
    """Расхождения по каждому показателю. Ось — по кратчайшей дуге."""
    out = {}
    for k in ("pr_ms", "qrs_ms", "qt_ms"):
        if ours.get(k) is not None and want.get(k) is not None:
            out[k] = float(ours[k] - want[k])
    a, b = (ours.get("axis") or {}).get("qrs"), want.get("axis")
    if a is not None and b is not None:
        out["axis"] = float((a - b + 180) % 360 - 180)
    if ours.get("hr") and want.get("hr"):
        out["hr"] = float(ours["hr"] - want["hr"])
    if "qt" in ours.get("flags", []):
        out.pop("qt_ms", None)          # сами пометили как ненадёжное
        out["_flagged_qt"] = 1.0
    st_err = [ours["st"][ld]["j80"] - want["st"][ld]
              for ld in LEAD_ORDER
              if ld in ours.get("st", {}) and want["st"].get(ld) is not None
              and ours["st"][ld].get("j80") is not None]
    if st_err:
        out["st_uv"] = float(np.median(np.abs(st_err)) * 1000)
    return out


def report(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("нет данных")
        return
    print(f"{'показатель':12s} {'n':>4s} {'медиана':>9s} {'|ошибка|':>9s} "
          f"{'90%':>7s} {'в допуске':>10s}")
    for k, label in (("pr_ms", "PR, мс"), ("qrs_ms", "QRS, мс"), ("qt_ms", "QT, мс"),
                     ("axis", "ось, °"), ("hr", "ЧСС, /мин"), ("st_uv", "ST, мкВ")):
        v = np.array([r[k] for r in rows if k in r])
        if not len(v):
            print(f"{label:12s} {'—':>4s}")
            continue
        tol = TOL.get(k)
        ok = f"{100 * np.mean(np.abs(v) <= tol):9.0f}%" if tol else "—".rjust(10)
        print(f"{label:12s} {len(v):4d} {np.median(v):+9.1f} "
              f"{np.median(np.abs(v)):9.1f} {np.percentile(np.abs(v), 90):7.1f} {ok:>10s}")
    flag = sum(int("_flagged_qt" in r) for r in rows)
    print(f"записей сравнено: {len(rows)}"
          + (f", из них QT помечен нами как ненадёжный: {flag}" if flag else ""))


def run(hea_list: list[Path], gt: dict[int, dict], digit_dir: Path,
        stages: str) -> tuple[list[dict], list[dict], list[str]]:
    a_rows, b_rows, notes = [], [], []
    for hea in hea_list:
        eid = int(hea.stem.split("_")[0])
        want = gt.get(eid)
        if not want:
            continue
        if "a" in stages:
            t = load_truth(hea)
            if t:
                m = dl.measure(t[0], t[1], FS_TRUTH, simultaneous=True)
                if m:
                    a_rows.append(diffs(m, want))
                else:
                    notes.append(f"A {hea.stem}: разметка не сошлась")
        if "b" in stages:
            csvs = list((digit_dir / hea.stem).glob("*_timeseries_canonical.csv"))
            if not csvs:
                continue
            sig, names = load_csv(str(csvs[0]))
            m = dl.measure(sig, names, FS_OURS, simultaneous=False)
            if m:
                b_rows.append(diffs(m, want))
            else:
                notes.append(f"B {hea.stem}: разметка не сошлась")
    return a_rows, b_rows, notes


def calibrate(hea_list: list[Path], gt: dict[int, dict]) -> None:
    """Подбор порогов по эталонной разметке.

    Порог берём один на всех, а не подгоняем под каждую запись: подгонка под
    запись показала бы точность, которой на новых данных не будет.
    """
    data = []
    for hea in hea_list:
        eid = int(hea.stem.split("_")[0])
        if eid not in gt:
            continue
        t = load_truth(hea)
        if t:
            data.append((t, gt[eid]))
    print(f"подбор на {len(data)} записях")

    for name, values in (("K_QRS_ON", [0.03, 0.045, 0.055, 0.07, 0.09, 0.12]),
                         ("K_QRS_OFF", [0.03, 0.045, 0.055, 0.07, 0.09, 0.12]),
                         ("K_P", [0.10, 0.15, 0.20, 0.28, 0.35])):
        base = getattr(dl, name)
        print(f"\n{name}:")
        for v in values:
            setattr(dl, name, v)
            errs = []
            for (sig, names), want in data:
                m = dl.measure(sig, names, FS_TRUTH, simultaneous=True)
                if not m:
                    continue
                d = diffs(m, want)
                key = "pr_ms" if name == "K_P" else "qrs_ms"
                if key in d:
                    errs.append(d[key])
            if errs:
                e = np.abs(errs)
                print(f"  {v:5.3f}  n={len(errs):3d}  смещение {np.median(errs):+6.1f}  "
                      f"|ошибка| {np.median(e):5.1f}  90% {np.percentile(e, 90):5.1f}")
        setattr(dl, name, base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--count", type=int, default=40)
    ap.add_argument("--data", default=str(ROOT / "data" / "ptbxl"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "ptbxl_plus"))
    ap.add_argument("--ref", default="glasgow", choices=list(SOURCES))
    ap.add_argument("--digitized", default=str(ROOT / "output" / "validate"))
    ap.add_argument("--stages", default="ab", help="a = эталон, b = после оцифровки")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    gtdir = Path(a.gt)
    fn, axis_col, st_col = SOURCES[a.ref]
    gt = truth_table(gtdir / fn, axis_col, st_col)
    heas = sorted(Path(a.data).glob("*.hea"))[:a.count]
    if not heas:
        raise SystemExit(f"нет записей в {a.data}")
    have = [h for h in heas if int(h.stem.split('_')[0]) in gt]
    print(f"записей: {len(heas)}, с разметкой {a.ref}: {len(have)}")

    if a.calibrate:
        calibrate(have, gt)
        return

    a_rows, b_rows, notes = run(have, gt, Path(a.digitized), a.stages)
    report(a_rows, f"этап A — эталонный сигнал против {a.ref} (ошибка алгоритма)")
    report(b_rows, f"этап B — после оцифровки против {a.ref} (ошибка всей цепочки)")

    # Точка отсчёта: насколько два фирменных анализатора расходятся друг с
    # другом на этих же записях. Наша ошибка имеет смысл только рядом с ней.
    other = "12sl" if a.ref == "glasgow" else "glasgow"
    fn2, axis2, st2 = SOURCES[other]
    if (gtdir / fn2).exists():
        gt2 = truth_table(gtdir / fn2, axis2, st2)
        ids = [int(h.stem.split('_')[0]) for h in have]
        report(analyzer_spread(gt, gt2, ids),
               f"для сравнения — расхождение {a.ref} и {other} между собой")
    for n in notes[:20]:
        print("  ", n)


if __name__ == "__main__":
    main()
