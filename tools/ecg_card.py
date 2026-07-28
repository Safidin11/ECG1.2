"""Карточка измерений: числа + картинка, по которой их видно.

Главное решение здесь — не прятать измерение за цифрой. Число «QT 396 мс»
проверить нельзя, а вот усреднённый комплекс с отметками начала P, начала и
конца QRS и конца T проверить можно глазами за секунду: видно, куда алгоритм
поставил границы и не уехал ли он.

Поэтому карточка состоит из трёх частей:
  плитки   — сами величины с полосой нормы, чтобы понять «много или мало»;
  комплекс — тот самый усреднённый удар на миллиметровке с отметками;
  круг оси — направление электрической оси на шестиосевой схеме.

Всё рисуется как inline-SVG на переменных темы страницы, без внешних файлов.

Demo-инструмент, НЕ медизделие.
"""
from __future__ import annotations

import numpy as np

# Развёртка как на настоящей плёнке: 25 мм/с и 10 мм/мВ. Отступать от этого
# нельзя — иначе привычка «клеточка = 40 мс» перестаёт работать, а она и есть
# главный способ проверить прибор на глаз.
MM_S, MM_MV = 25.0, 10.0
PX_MM = 13.0                      # экранных пикселей в миллиметре плёнки

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Нормы. Взяты как общепринятые взрослые границы; служат только подсказкой
# «много или мало», не заключением.
NORMS = {
    "hr": (60, 100, 30, 160, "уд/мин"),
    "pr_ms": (120, 200, 80, 320, "мс"),
    "qrs_ms": (70, 110, 40, 180, "мс"),
    "qtc": (350, 450, 280, 560, "мс"),
    "sokolow": (0, 35, 0, 60, "мм"),
}

# Цвета интервалов на комплексах. Синий — PQ, красный — комплекс, янтарный — QT.
BANDS = (("p_on", "qrs_on", "pq", "#2563eb", 0.30),
         ("qrs_on", "qrs_off", "qrs", "#dc2626", 0.30),
         ("qrs_on", "t_off", "qt", "#d97706", 0.16))


def _fmt(v, digits=0):
    return "—" if v is None else f"{v:.{digits}f}"


def tiles(m: dict) -> list[dict]:
    """Плитки с величинами и положением на шкале нормы."""
    qtc = (m.get("qtc") or {}).get("bazett")
    sk = m.get("sokolow")
    items = [
        ("ЧСС", "hr", m.get("hr"), "уд/мин", "по интервалам между комплексами"),
        ("PQ", "pr_ms", m.get("pr_ms"), "мс", "от начала зубца P до начала комплекса"),
        ("QRS", "qrs_ms", m.get("qrs_ms"), "мс", "длительность комплекса"),
        ("QTc", "qtc", qtc, "мс", "QT с поправкой на частоту, по Базетту"),
        ("Соколов", "sokolow", (sk or {}).get("mm"), "мм",
         (f"S в V1 + R в {sk['lead']}, порог 35 мм" if sk
          else "S в V1 + R в V5 или V6")),
    ]
    out = []
    for label, key, val, unit, hint in items:
        lo, hi, dmin, dmax, _ = NORMS[key]
        pos = None if val is None else 100 * (min(max(val, dmin), dmax) - dmin) / (dmax - dmin)
        out.append({
            "label": label, "key": key, "unit": unit, "hint": hint,
            "value": _fmt(val), "pos": pos,
            "band": (100 * (lo - dmin) / (dmax - dmin), 100 * (hi - lo) / (dmax - dmin)),
            "state": ("na" if val is None else
                      "ok" if lo <= val <= hi else
                      "low" if val < lo else "high"),
            "flag": key.startswith("qt") and "qt" in m.get("flags", []),
        })
    return out


def qt_detail(m: dict) -> str:
    """Подпись к QTc: сам QT и разброс между формулами поправки."""
    qc, qt = m.get("qtc"), m.get("qt_ms")
    if not qc or not qt:
        return ""
    v = sorted(qc.values())
    return (f"QT {qt:.0f} мс · поправки {v[0]:.0f}–{v[-1]:.0f} мс "
            f"(Базетт, Фридерисия, Framingham, Ходжес)")


def _grid(w: float, h: float) -> str:
    """Миллиметровка: тонкая сетка 1 мм и толстая 5 мм."""
    p = [f'<rect width="{w:.0f}" height="{h:.0f}" fill="var(--paper)"/>']
    for step, cls in ((PX_MM, "gm"), (PX_MM * 5, "gM")):
        d = []
        x = 0.0
        while x <= w:
            d.append(f"M{x:.1f} 0V{h:.0f}")
            x += step
        y = 0.0
        while y <= h:
            d.append(f"M0 {y:.1f}H{w:.0f}")
            y += step
        p.append(f'<path class="{cls}" d="{"".join(d)}"/>')
    return "".join(p)


def beats_svg(m: dict) -> tuple[list[dict], float] | None:
    """Двенадцать усреднённых комплексов с общими отметками границ.

    Именно все двенадцать и именно в ОДНОМ вертикальном масштабе — так печатает
    отчёт настоящий кардиограф. Один комплекс показал бы, где стоят отметки, но
    не дал бы сравнить отведения между собой, а разница между ними и есть то,
    ради чего ЭКГ снимают в двенадцати отведениях.

    Отметки границ общие для всех клеток: они и получены сразу по всем
    отведениям, а не по каждому отдельно.
    """
    beats, mk = m.get("beats"), m.get("marks")
    if not beats or not mk:
        return None
    fs = m["fs"]

    # Поля: 60 мс до начала P и 80 мс после конца T — чтобы отметка на краю
    # читалась как отметка, а не как обрез.
    left = (mk["p_on"] if mk["p_on"] is not None else mk["qrs_on"]) - int(0.06 * fs)
    right = (mk["t_off"] if mk["t_off"] is not None else mk["qrs_off"]) + int(0.08 * fs)
    span = min(len(w) for w in beats.values())
    left, right = max(0, left), min(span, right)
    if right - left < int(0.15 * fs):
        return None

    shown = [n for n in LEAD_ORDER if n in beats]
    span = max(float(np.max(np.abs(beats[n][left:right] - m["baseline"][n])))
               for n in shown)
    # Усиление режем вдвое, если так не влезает — ровно как кардиограф, который
    # печатает «1/2» при высоких зубцах. Иначе клетка перестала бы быть клеткой.
    gain = MM_MV if span * MM_MV <= 13 else MM_MV / 2
    half_mm = max(6.0, min(15.0, span * gain * 1.15))

    px_s = MM_S * PX_MM
    width = (right - left) / fs * px_s
    height = 2 * half_mm * PX_MM
    x = lambda i: (i - left) / fs * px_s                    # noqa: E731

    # Полосы интервалов заливаем не плашкой, а вертикальным градиентом: густо у
    # краёв кадра и почти прозрачно посередине, где идёт сама кривая. Так
    # интервал читается, а линию ничем не заслоняет — плашка её притеняла.
    # Порядок обратный густоте: широкий QT кладём первым, чтобы более узкие и
    # более важные PQ и QRS ложились поверх.
    def bands(uid: str) -> str:
        defs, rects = [], []
        for a, b, cls, colour, top in reversed(BANDS):
            i, j = mk.get(a), mk.get(b)
            if i is None or j is None:
                continue
            gid = f"g{cls}{uid}"
            stops = "".join(
                f'<stop offset="{o}" stop-color="{colour}" stop-opacity="{v:.3f}"/>'
                for o, v in ((0, top), (0.40, top * 0.10), (0.60, top * 0.10), (1, top)))
            defs.append(f'<linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
                        f'{stops}</linearGradient>')
            rects.append(f'<rect x="{x(i):.1f}" y="0" width="{x(j) - x(i):.1f}" '
                         f'height="{height:.0f}" fill="url(#{gid})"/>')
        return f'<defs>{"".join(defs)}</defs>{"".join(rects)}'

    lines = "".join(f'<path class="mk" d="M{x(mk[k]):.1f} 0V{height:.0f}"/>'
                    for k in ("p_on", "qrs_on", "qrs_off", "t_off")
                    if mk.get(k) is not None)

    out = []
    for name in shown:
        base = m["baseline"][name]
        seg = np.asarray(beats[name], float)[left:right] - base
        y = lambda v: height / 2 - v * gain * PX_MM         # noqa: E731
        d = "".join(("M" if k == 0 else "L") + f"{x(left + k):.1f} {y(v):.1f}"
                    for k, v in enumerate(seg))
        off = name in (m.get("misaligned") or [])
        out.append({"lead": name, "off": off, "svg": (
            f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
            f'aria-label="усреднённый комплекс, отведение {name}">'
            # id градиентов должны быть уникальны в пределах страницы: на ней
            # двенадцать таких SVG, и совпадение имён склеило бы их заливки
            f'{_grid(width, height)}{bands(name)}{lines}'
            f'<path class="iso" d="M0 {height / 2:.1f}H{width:.0f}"/>'
            f'<path class="tr" d="{d}"/>'
            f'<text class="ld{" off" if off else ""}" x="5" y="15">{name}</text>'
            + ('<text class="warn" x="5" y="30">не совмещено</text>' if off else "")
            + "</svg>")})
    return out, gain


def axis_svg(deg: float | None) -> str:
    """Шестиосевая схема с направлением электрической оси.

    Норму (-30..+90) закрашиваем сектором: положение стрелки внутри или вне
    сектора читается мгновенно, в отличие от числа в градусах.
    """
    R, C = 58.0, 76.0                 # запас до края: подписи осей идут снаружи
    pt = lambda a, r: (C + r * np.cos(np.radians(a)), C + r * np.sin(np.radians(a)))  # noqa: E731
    p = []
    a0, a1 = pt(-30, R), pt(90, R)
    p.append(f'<path class="norm" d="M{C} {C}L{a0[0]:.1f} {a0[1]:.1f}'
             f'A{R} {R} 0 0 1 {a1[0]:.1f} {a1[1]:.1f}Z"/>')
    for a, name in ((0, "I"), (60, "II"), (120, "III"), (-150, "aVR"),
                    (-30, "aVL"), (90, "aVF")):
        x1, y1 = pt(a, R)
        x2, y2 = pt(a + 180, R)
        p.append(f'<path class="ax" d="M{x2:.1f} {y2:.1f}L{x1:.1f} {y1:.1f}"/>')
        tx, ty = pt(a, R + 11)
        p.append(f'<text class="axl" x="{tx:.1f}" y="{ty + 3:.1f}">{name}</text>')
    if deg is not None:
        hx, hy = pt(deg, R - 8)
        p.append(f'<path class="vec" d="M{C} {C}L{hx:.1f} {hy:.1f}"/>')
        p.append(f'<circle class="vec" cx="{hx:.1f}" cy="{hy:.1f}" r="3.5"/>')
    return (f'<svg class="dial" viewBox="0 0 {2 * C} {2 * C}" role="img" '
            f'aria-label="электрическая ось">{"".join(p)}</svg>')


def axis_word(deg: float | None) -> str:
    """Словесное название положения оси — то, как о ней говорят."""
    if deg is None:
        return "не определена"
    if -30 <= deg <= 90:
        return "нормальная"
    if -90 <= deg < -30:
        return "отклонена влево"
    if 90 < deg <= 180:
        return "отклонена вправо"
    return "резкое отклонение"


def st_rows(m: dict) -> list[dict]:
    """Уровень ST по отведениям, в миллиметрах плёнки.

    В миллиметрах, а не в милливольтах, потому что пороги, которыми пользуются
    («подъём на 1 мм», «на 2 мм в грудных»), заданы именно в клетках плёнки.
    """
    st = m.get("st") or {}
    out = []
    for n in LEAD_ORDER:
        v = (st.get(n) or {}).get("j60")
        mm = None if v is None else v * MM_MV
        # Порог заметности выше в V2-V3: там подъём точки J встречается и у
        # здоровых, и общая мерка давала бы сплошные ложные отметки.
        lim = 2.0 if n in ("V2", "V3") else 1.0
        out.append({"lead": n, "mm": ("—" if mm is None else f"{mm:+.1f}"),
                    "state": ("na" if mm is None else
                              "up" if mm >= lim else
                              "down" if mm <= -1.0 else "ok")})
    return out


FLAG_TEXT = {
    "qt": ("QT ненадёжен", "Поправленный QT вышел за пределы правдоподобия. "
           "Обычно так бывает при частом ритме, когда зубец T сливается со "
           "следующим зубцом P и конец T поставить некуда."),
    "p": ("Зубец P не найден", "Интервал PR не показан. Так бывает при "
          "мерцательной аритмии, при узловом ритме и просто при мелком зубце P."),
    "t": ("Конец зубца T не найден", "QT и QTc не показаны. Касательная к спаду "
          "зубца T получилась ненадёжной меньше чем в двух отведениях — обычно "
          "потому, что зубец T плоский или линия прочитана с разрывами."),
    "beats": ("Мало ударов", "Комплекс усреднён меньше чем по трём ударам — "
              "шум подавлен слабо, границы могут гулять."),
    "leads": ("Мало отведений", "Пространственная скорость собрана меньше чем "
              "по шести отведениям, границы менее устойчивы."),
}


def card(m: dict | None) -> dict | None:
    """Всё, что нужно шаблону страницы."""
    if not m:
        return None
    got = beats_svg(m)
    ax = (m.get("axis") or {}).get("qrs")
    return {
        "tiles": tiles(m), "qt_detail": qt_detail(m),
        "beats_svg": got[0] if got else None,
        "gain": (None if not got or got[1] == MM_MV else "1/2"),
        "dial": axis_svg(ax), "axis": ("—" if ax is None else f"{ax:+.0f}°"),
        "axis_word": axis_word(ax), "st": st_rows(m),
        "beats": m.get("n_beats"), "leads": m.get("n_leads"),
        "misaligned": m.get("misaligned") or [],
        "flags": [{"title": FLAG_TEXT[f][0], "text": FLAG_TEXT[f][1]}
                  for f in m.get("flags", []) if f in FLAG_TEXT],
    }
