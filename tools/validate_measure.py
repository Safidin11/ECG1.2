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
TOL = {"pr_ms": 20.0, "qrs_ms": 12.0, "qt_ms": 25.0, "axis": 15.0,
       "p_ms": 20.0, "p_axis": 25.0, "t_axis": 25.0}
# Ось зубца P допуск шире, чем ось комплекса, и не от лени: зубец P на порядок
# мельче комплекса, поэтому его направление куда чувствительнее к изолинии.
# Между двумя фирменными анализаторами расхождение по оси P само по себе
# больше 20 градусов — требовать от себя меньшего бессмысленно.


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
            # «Зубец P есть» — отдельный признак, и он ценнее длительности: по
            # нему видно, теряем ли мы зубец вовсе. Потеря и ошибка на 10 мс —
            # разные беды, и мерить их надо порознь.
            #
            # Считаем зубец существующим, если анализатор назвал его
            # длительность, а не по флагу P_Found_Global. Флаг означает не то,
            # что кажется: на пяти записях этого набора Glasgow ставит его в
            # ноль, но тут же выдаёт и длительность зубца P, и интервал PQ, а
            # второй анализатор на тех же записях ставит флаг в единицу. Все
            # пять — тахикардия 127-151, где зубец P наезжает на зубец T. По
            # флагу выходило бы, что мы пять раз «нашли несуществующее», хотя
            # оба прибора зубец видят и меряют.
            found = num("P_Dur_Global")
            out[i] = {
                "pr_ms": num("PR_Int_Global"),
                "qrs_ms": num("QRS_Dur_Global"),
                "qt_ms": num("QT_Int_Global"),
                "axis": None if indet else num(axis_col),
                "p_found": None if found is None else bool(found),
                "p_ms": num("P_Dur_Global"),
                "p_axis": num("P_AxisFront_Global"),
                "t_axis": num("T_AxisFront_Global"),
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
            rows.append(diffs({**a[i], "flags": [],
                               "axis": {"qrs": a[i]["axis"], "p": a[i]["p_axis"],
                                        "t": a[i]["t_axis"]}}, b[i]))
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
    for k in ("pr_ms", "qrs_ms", "qt_ms", "p_ms"):
        if ours.get(k) is not None and want.get(k) is not None:
            out[k] = float(ours[k] - want[k])
    arc = lambda u, v: float((u - v + 180) % 360 - 180)          # noqa: E731
    for key, mine, theirs in (("axis", "qrs", "axis"), ("p_axis", "p", "p_axis"),
                              ("t_axis", "t", "t_axis")):
        a, b = (ours.get("axis") or {}).get(mine), want.get(theirs)
        if a is not None and b is not None:
            out[key] = arc(a, b)
    # Потеря зубца — отдельный счёт, не ошибка в миллисекундах. Анализатор
    # сказал «зубец есть», а мы его не нашли: интервала PQ просто нет, и в
    # сравнение длительностей такая запись не попадёт вовсе — то есть молча
    # улучшит статистику, выкинув из неё самые трудные случаи.
    if want.get("p_found") is not None:
        out["_p_want"] = 1.0 if want["p_found"] else 0.0
        out["_p_got"] = 1.0 if ours.get("pr_ms") is not None else 0.0
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
    for k, label in (("p_ms", "P, мс"), ("pr_ms", "PQ, мс"), ("qrs_ms", "QRS, мс"),
                     ("qt_ms", "QT, мс"), ("axis", "ось QRS, °"),
                     ("p_axis", "ось P, °"), ("t_axis", "ось T, °"),
                     ("hr", "ЧСС, /мин"), ("st_uv", "ST, мкВ")):
        v = np.array([r[k] for r in rows if k in r])
        if not len(v):
            print(f"{label:12s} {'—':>4s}")
            continue
        tol = TOL.get(k)
        ok = f"{100 * np.mean(np.abs(v) <= tol):9.0f}%" if tol else "—".rjust(10)
        print(f"{label:12s} {len(v):4d} {np.median(v):+9.1f} "
              f"{np.median(np.abs(v)):9.1f} {np.percentile(np.abs(v), 90):7.1f} {ok:>10s}")
    # Находимость зубца P — отдельно от точности. Показатель «в допуске 94%»
    # почти ничего не стоит, если половина записей в сравнение не попала из-за
    # того, что зубца мы вовсе не нашли: выпадают как раз мелкие зубцы, то есть
    # самые трудные, и статистика улучшается ровно от того, что мы их потеряли.
    pw = [r for r in rows if "_p_want" in r]
    if pw:
        yes = [r for r in pw if r["_p_want"]]
        no = [r for r in pw if not r["_p_want"]]
        miss = sum(1 for r in yes if not r["_p_got"])
        false = sum(1 for r in no if r["_p_got"])
        print(f"зубец P: анализатор нашёл на {len(yes)} записях, мы потеряли "
              f"{miss} ({100 * (1 - miss / max(len(yes), 1)):.0f}% находимость)"
              + (f"; анализатор не нашёл на {len(no)}, мы «нашли» {false}" if no else ""))
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


def erase_p(sig: np.ndarray, fs: int, rng: np.random.Generator,
            span: tuple[int, int]) -> np.ndarray:
    """Стереть зубец P, оставив всё остальное как было.

    Нужно для замера ЛОЖНЫХ срабатываний. Находимость зубца P одна ничего не
    доказывает: детектор, который говорит «зубец есть» всегда, покажет 100%.
    А записей без зубца P (мерцательная аритмия, узловой ритм) в этом наборе
    нет — значит их надо изготовить.

    span — где зубец лежит относительно вершины R, в отсчётах; берётся из
    собственной разметки этой же записи. Три попытки назначить это место по
    геометрии («столько-то миллисекунд до комплекса», «первая точка на уровне
    изолинии») кончились одинаково: стиралось не то, а тест исправно считал
    оставшийся нетронутым зубец P ложной тревогой — 9, потом 22, потом 14
    случаев из сорока, и все свои. Место надо брать измеренное.

    Стираем прямой между концами участка плюс шум САМОГО участка — его же
    дрожание, снятое как разница со сглаженной копией. Так исчезает только
    зубец P, а уровень шума, на который детектор и опирается, остаётся прежним.
    Заменить нулями было бы поблажкой: идеально гладкий участок распознать как
    «пусто» легко. Зубец T при этом не трогаем: в настоящей мерцательной
    аритмии он на месте, и не спутать его с зубцом P — часть задачи.
    """
    out = np.array(sig, float, copy=True)
    j = int(np.argmax([np.nanstd(sig[:, i]) for i in range(sig.shape[1])]))
    col0 = np.nan_to_num(sig[:, j])
    peaks = dl.refine_peaks(col0, dl.r_peaks(col0, fs), fs)
    pad = dl._ms(25, fs)
    for i in range(sig.shape[1]):
        col = np.nan_to_num(sig[:, i])
        for r in peaks:
            a, b = int(r) + span[0] - pad, int(r) + span[1] + pad
            if a < 0 or b >= len(col) or b - a <= dl._ms(30, fs):
                continue
            seg = col[a:b]
            sigma = float(np.std(seg - dl._smooth(seg, dl._ms(30, fs))))
            out[a:b, i] = np.linspace(col[a], col[b], b - a) + rng.normal(0, sigma, b - a)
    return out


def specificity(hea_list: list[Path]) -> None:
    """Как часто детектор находит зубец P там, где его нет."""
    rng = np.random.default_rng(0)
    false, total = 0, 0
    for hea in hea_list:
        t = load_truth(hea)
        if not t:
            continue
        sig, names = t
        m = dl.measure(sig, names, FS_TRUTH, simultaneous=True)
        if not m or m["marks"]["p_on"] is None:
            continue                    # стирать нечего: зубец и так не найден
        mk, r = m["marks"], m["marks"]["r"]
        m2 = dl.measure(erase_p(sig, FS_TRUTH, rng, (mk["p_on"] - r, mk["p_off"] - r)),
                        names, FS_TRUTH, simultaneous=True)
        if m2 is None:
            continue
        total += 1
        if m2.get("pr_ms") is not None:
            false += 1
    print("\n=== ложные срабатывания: записи со СТЁРТЫМ зубцом P ===")
    print(f"проверено {total}; зубец P «найден» на {false} "
          f"({100 * false / max(total, 1):.0f}%)")


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
    ap.add_argument("--specificity", action="store_true",
                    help="ложные срабатывания на записях со стёртым зубцом P")
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
    if a.specificity:
        specificity(have)
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
