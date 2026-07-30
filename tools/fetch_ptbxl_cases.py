"""Набор записей PTB-XL, подобранный по диагнозам, а не по номерам подряд.

Зачем отдельно от tools/fetch_ptbxl.sh. Тот берёт первые N записей базы, и
получается набор, в котором почти всё — обычные синусовые ЭКГ: разметке на них
легко, и стенд показывает благополучие, которого нет. Настоящие поломки живут в
краях: там, где зубца P нет вовсе (мерцательная аритмия), где он двугорбый
(перегрузка левого предсердия), где комплекс широкий (блокада ножки), где ритм
частый и зубец P наезжает на зубец T, где стоит кардиостимулятор.

Поэтому берём по нескольку записей на каждую группу диагнозов. Группы названы
кодами SCP из самой базы, так что выбор не наш и не подгонка: что врач в PTB-XL
пометил, то и берём.

    .venv/bin/python tools/fetch_ptbxl_cases.py                # по 6 на группу
    .venv/bin/python tools/fetch_ptbxl_cases.py --per-group 10

Кладёт сигналы в data/ptbxl_cases/ и рядом cases.csv — какая запись к какой
группе относится, чтобы стенд мог показать ошибку ПО ГРУППАМ. Средняя ошибка по
всему набору прячет ровно то, ради чего он собран.
"""
from __future__ import annotations

import argparse
import ast
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://physionet.org/files/ptb-xl/1.0.3"

# Группы: код SCP -> зачем он тут. Ровно то, что ломает разметку по-разному.
GROUPS = {
    "NORM": "норма — точка отсчёта",
    "AFIB": "мерцательная аритмия — зубца P нет вовсе",
    "LAO/LAE": "перегрузка левого предсердия — зубец P широкий и двугорбый",
    "RAO/RAE": "перегрузка правого предсердия — зубец P высокий и острый",
    "1AVB": "атриовентрикулярная блокада 1 ст. — интервал PQ удлинён",
    "LPR": "удлинённый PQ без названного диагноза блокады",
    "STACH": "тахикардия — зубец P наезжает на зубец T",
    "SBRAD": "брадикардия — окно поиска зубца P широкое",
    "SARRH": "синусовая аритмия — интервалы между ударами гуляют",
    "CLBBB": "полная блокада левой ножки — комплекс широкий",
    "CRBBB": "полная блокада правой ножки — комплекс широкий и зазубрен",
    "IVCD": "неспецифическое замедление внутри желудочков",
    "PVC": "желудочковая экстрасистолия — чужеродные удары",
    "PAC": "предсердная экстрасистолия",
    "PACE": "кардиостимулятор — на плёнке пики стимула",
    "LVH": "гипертрофия левого желудочка — высокие зубцы",
    "LVOLT": "низкий вольтаж — всё мелкое",
    "IMI": "нижний инфаркт — зубцы Q, изменения ST",
    "ASMI": "переднеперегородочный инфаркт",
    "LNGQT": "удлинённый QT — конец зубца T далеко",
    "INVT": "отрицательные зубцы T — касательная строится по-другому",
}


def pick(per_group: int, seed: int) -> dict[int, str]:
    """Выбрать записи: по несколько на группу, каждая запись только в одну.

    Запись попадает в группу по коду с наибольшей уверенностью врача. Одна и та
    же ЭКГ несёт несколько пометок сразу, и если разрешить ей числиться везде,
    набор перекосится: «мерцательная аритмия» наберётся из записей, где она
    указана мимоходом, а главным был инфаркт.
    """
    import random
    rng = random.Random(seed)
    pool: dict[str, list[int]] = {g: [] for g in GROUPS}
    for r in csv.DictReader((ROOT / "data" / "ptbxl_database.csv").open(encoding="utf-8")):
        codes = ast.literal_eval(r["scp_codes"])
        mine = {k: v for k, v in codes.items() if k in GROUPS}
        if not mine:
            continue
        # NORM ставится вместе с SR почти всегда; берём его, только если он и
        # есть главный, иначе группа «норма» наберётся из больных записей.
        best = max(mine, key=lambda k: (mine[k], k != "NORM"))
        pool[best].append(int(r["ecg_id"]))
    out = {}
    for g, ids in pool.items():
        rng.shuffle(ids)
        for i in ids[:per_group]:
            out[i] = g
        if len(ids) < per_group:
            print(f"  {g}: всего {len(ids)} записей, беру все")
    return out


def fetch(ids: dict[int, str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for n, (eid, group) in enumerate(sorted(ids.items()), 1):
        name = f"{eid:05d}_hr"
        sub = f"{eid // 1000 * 1000:05d}"
        for ext in ("hea", "dat"):
            f = dest / f"{name}.{ext}"
            if f.exists() and f.stat().st_size > 0:
                continue
            url = f"{BASE}/records500/{sub}/{name}.{ext}"
            r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(f), url])
            if r.returncode:
                print(f"  пропуск {name}.{ext}")
        if n % 20 == 0:
            print(f"  {n}/{len(ids)}", flush=True)
    with (dest / "cases.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ecg_id", "group", "why"])
        for eid, g in sorted(ids.items()):
            w.writerow([eid, g, GROUPS[g]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "ptbxl_cases"))
    a = ap.parse_args()

    db = ROOT / "data" / "ptbxl_database.csv"
    if not db.exists():
        sys.exit(f"нет {db} — скачай:\n  curl -o {db} {BASE}/ptbxl_database.csv")
    ids = pick(a.per_group, a.seed)
    print(f"выбрано {len(ids)} записей в {len(GROUPS)} группах")
    fetch(ids, Path(a.out))
    print(f"готово: {a.out}")


if __name__ == "__main__":
    main()
