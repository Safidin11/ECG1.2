r"""Разметка на записях с разными диагнозами — ошибка ПО ГРУППАМ.

Зачем отдельно от tools/validate_measure.py. Тот меряет среднюю точность на
подряд взятых записях, и это правильный вопрос — «насколько мы обычно правы».
Но у него есть слепое пятно: набор из подряд взятых записей почти весь состоит
из обычных синусовых ЭКГ, а ломается разметка на краях. Двугорбый зубец P,
широкий комплекс при блокаде ножки, зубец P, наехавший на зубец T при
тахикардии, кардиостимулятор — каждого такого случая в общем наборе по одному,
и его ошибка тонет в среднем.

Здесь наоборот: записи подобраны по диагнозам (tools/fetch_ptbxl_cases.py), по
нескольку на группу, и ошибка показывается по каждой группе отдельно. Средняя
цифра тут была бы бессмысленна — состав набора выбран нами, а не природой.

    .venv/bin/python tools/fetch_ptbxl_cases.py     # сначала скачать
    .venv/bin/python tools/validate_cases.py
    .venv/bin/python tools/validate_cases.py --show LAO/LAE   # записи группы

Demo-инструмент, НЕ медизделие.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import ecg_delineate as dl                                    # noqa: E402
from validate_measure import (SOURCES, TOL, diffs, load_truth,  # noqa: E402
                              truth_table)

FS = 500


def cases(path: Path) -> dict[int, str]:
    return {int(r["ecg_id"]): r["group"]
            for r in csv.DictReader(path.open(encoding="utf-8"))}


def run(data: Path, groups: dict[int, str], gt: dict[int, dict],
        only: str | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for hea in sorted(data.glob("*.hea")):
        eid = int(hea.stem.split("_")[0])
        g = groups.get(eid)
        if not g or (only and g != only) or eid not in gt:
            continue
        t = load_truth(hea)
        if not t:
            continue
        try:
            m = dl.measure(t[0], t[1], FS, simultaneous=True)
        except Exception as exc:                     # noqa: BLE001
            out.setdefault(g, []).append({"_crash": str(exc), "_id": eid})
            continue
        if m is None:
            out.setdefault(g, []).append({"_none": 1.0, "_id": eid})
            continue
        out.setdefault(g, []).append(diffs(m, gt[eid]) | {"_id": eid})
    return out


def line(rows: list[dict], key: str) -> str:
    v = np.array([r[key] for r in rows if key in r])
    if not len(v):
        return f"{'—':>13s}"
    tol = TOL.get(key)
    ok = f" {100 * np.mean(np.abs(v) <= tol):3.0f}%" if tol else "     "
    return f"{np.median(v):+6.0f}/{np.median(np.abs(v)):4.0f}{ok}"


def report(res: dict[str, list[dict]], why: dict[str, str]) -> None:
    keys = (("p_ms", "P"), ("pr_ms", "PQ"), ("qrs_ms", "QRS"), ("qt_ms", "QT"),
            ("axis", "ось QRS"))
    print(f"\n{'группа':10s} {'n':>3s} {'бр':>3s}", end="")
    for _, lab in keys:
        print(f" {lab:>13s}", end="")
    print("   зубец P")
    print("  " + "-" * 96)
    for g in sorted(res):
        rows = res[g]
        broke = sum(1 for r in rows if "_none" in r or "_crash" in r)
        good = [r for r in rows if "_none" not in r and "_crash" not in r]
        pw = [r for r in good if "_p_want" in r]
        miss = sum(1 for r in pw if r["_p_want"] and not r["_p_got"])
        fake = sum(1 for r in pw if not r["_p_want"] and r["_p_got"])
        p = (f"{len(pw) - miss}/{len(pw)} найден" if pw else "—")
        if fake:
            p += f", {fake} лишних"
        print(f"{g:10s} {len(rows):3d} {broke:3d}", end="")
        for k, _ in keys:
            print(f" {line(good, k):>13s}", end="")
        print(f"   {p}")
    print("\n  медиана ошибки / медиана |ошибки| / доля в допуске; "
          "«бр» — разметка не сошлась")
    print("  допуск: P и PQ 20 мс, QRS 12 мс, QT 25 мс, ось 15°\n")
    for g in sorted(res):
        print(f"  {g:10s} — {why.get(g, '')}")


def detail(res: dict[str, list[dict]], gt: dict[int, dict], group: str) -> None:
    print(f"\n=== {group} — по записям ===")
    print(f"{'запись':8s} {'P нам/им':>12s} {'PQ нам/им':>12s} "
          f"{'QRS нам/им':>12s} {'QT нам/им':>12s}")
    for r in sorted(res.get(group, []), key=lambda x: x["_id"]):
        eid = r["_id"]
        w = gt[eid]
        if "_none" in r or "_crash" in r:
            print(f"{eid:08d} разметка не сошлась"
                  + (f": {r['_crash'][:60]}" if "_crash" in r else ""))
            continue
        cell = lambda k: (  # noqa: E731
            f"{(w[k] or 0) + r[k]:>5.0f}/{w[k]:<5.0f}" if k in r and w.get(k)
            else f"{'—':>5s}/{w.get(k) or '—':<5}")
        print(f"{eid:08d} {cell('p_ms'):>12s} {cell('pr_ms'):>12s} "
              f"{cell('qrs_ms'):>12s} {cell('qt_ms'):>12s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "ptbxl_cases"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "ptbxl_plus"))
    ap.add_argument("--ref", default="glasgow", choices=list(SOURCES))
    ap.add_argument("--show", help="показать записи одной группы")
    a = ap.parse_args()

    data = Path(a.data)
    cs = data / "cases.csv"
    if not cs.exists():
        raise SystemExit(f"нет {cs} — сначала tools/fetch_ptbxl_cases.py")
    groups = cases(cs)
    why = {r["group"]: r["why"] for r in csv.DictReader(cs.open(encoding="utf-8"))}
    fn, axis_col, st_col = SOURCES[a.ref]
    gt = truth_table(Path(a.gt) / fn, axis_col, st_col)
    res = run(data, groups, gt, a.show)
    if a.show:
        detail(res, gt, a.show)
    else:
        report(res, why)


if __name__ == "__main__":
    main()
