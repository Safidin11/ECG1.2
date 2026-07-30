"""Разметка комплекса и измерения: PQ, QRS, QT/QTc, ось, ST, индекс Соколова.

Идея та же, что у настоящих кардиографов: интервалы меряют не по одному
отведению, а по всем сразу. Деполяризация начинается в один момент во всём
сердце, но в отдельном отведении её начало может попасть на изолинию и быть
невидимым — вектор в этот момент направлен перпендикулярно оси отведения.
Поэтому строим ПРОСТРАНСТВЕННУЮ СКОРОСТЬ:

    sv(t) = sqrt( sum_i (dx_i/dt)^2 )

сумма по независимым отведениям. Когда сердце электрически молчит, sv ~ 0 в
любом отведении сразу; когда фронт пошёл — sv поднимается, где бы ни лежал
вектор. Границы ищем по порогу от пика sv.

Независимых отведений всего восемь: I, II, V1..V6. Остальные четыре —
арифметика от первых двух (III = II - I, aVR = -(I+II)/2, aVL = I - II/2,
aVF = II - I/2), и включать их в сумму значит просто трижды посчитать одно и
то же, перекосив sv в сторону фронтальной плоскости.

Так ищутся границы комплекса и зубца T. Зубец P — исключение: он мелкий и
пологий, его производная тонет в шуме, и по скорости граница садилась то на
случайный шип оцифровки, то на середину подъёма. Его ищем по ВЕЛИЧИНЕ
отклонения от изолинии (p_magnitude) — так же, как читают глазами.

Важная оговорка про наш случай. В раскладке 3x4 отведения СНЯТЫ НЕ
ОДНОВРЕМЕННО: в каждой клетке свои 2.5 с. Общего времени у них нет, поэтому
представительные комплексы приходится совмещать по всплеску энергии QRS. Это
приближение, и его цена измерена отдельно (tools/validate_measure.py:
режим simultaneous против совмещения).

Demo-инструмент, НЕ медизделие.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ecg_measure import (_shift, longest_run, r_peaks, refine_peaks,
                         representative_beat)

# Восемь линейно независимых отведений. Четыре оставшихся — их комбинации.
INDEP = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# Пороги разметки. K_QRS_* — в долях пика пространственной скорости, K_P — в
# долях высоты зубца P над тишиной (зубец P ищется по величине отклонения, а не
# по скорости, см. p_magnitude). Подобраны перебором
# на 40 записях PTB-XL против разметки анализатора University of Glasgow
# (tools/validate_measure.py --calibrate). Поверхность гладкая и монотонная по
# смещению, так что выбирали не «лучшую клетку», а нулевое систематическое
# смещение: оно означает, что мы не меряем комплекс стабильно шире или уже.
# Конец требует более высокого порога, чем начало, — комплекс не обрывается,
# а плавно переходит в сегмент ST, и низкий порог тянет границу в него.
K_QRS_ON = 0.055
K_QRS_OFF = 0.12
# K_P пересчитан после смены опоры на ближайшую впадину: при прежних 0.16
# длительность зубца P выходила на 6 мс короче эталонной, при 0.12 смещение
# ровно нулевое, а разброс наименьший и на эталонном сигнале, и после
# оцифровки (8.0 и 11.0 мс против 8.0 и 13.0). Ложных срабатываний при этом не
# прибавляется; при 0.22 они растут вдвое, а длительность уходит в -14 мс.
K_P = 0.12

# Поправка конца зубца T, мс. Метод касательной по построению даёт конец
# РАНЬШЕ, чем определение «кривая вернулась к изолинии»: касательная проведена
# в точке максимальной крутизны и пересекает изолинию, не дожидаясь пологого
# хвоста. Расхождение с разметкой Glasgow оказалось ровным сдвигом -16 мс на
# всех 40 записях (разброс вокруг него 6 мс), поэтому его просто снимаем.
# Это приведение к чужому определению, а не подгонка точности: без поправки
# разброс тот же самый, смещён только ноль.
T_END_MS = 16.0

# Границы правдоподобия QTc, мс. За ними измерение почти всегда сломано, а не
# удивительно: при тахикардии зубец T сливается со следующим P, и конец уезжает.
# Настоящий синдром удлинённого QT доходит до 550; 600+ — это отказ разметки.
QTC_RANGE = (300.0, 600.0)

# Отдельного признака «PR ненадёжен» здесь нет. Пока зубец P искался по
# пространственной скорости, интервал промахивался на 40-90 мс и пометить эти
# случаи не удавалось: проверялись и высота всплеска над тишиной, и
# устойчивость границы к смене сглаживания — обе мерки на промахнувшихся
# записях выглядели прекрасно, разметка уверенно находила НЕ ТОТ зубец.
# Вместо мерки уверенности сменили сам способ поиска (см. p_magnitude): 90-й
# процентиль ошибки упал с 42 до 18 мс, в допуск попало 94% вместо 76%.
# Оставшаяся защита — p_confirmed: если настоящего отклонения на найденном
# месте нет, интервал не показываем вовсе.


def _ms(v: float, fs: int) -> int:
    """Миллисекунды -> отсчёты. Всё окно задаётся в мс и переводится здесь,
    иначе при смене частоты дискретизации пороги молча поедут."""
    return int(round(v * fs / 1000.0))


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Скользящее среднее НЕЧЁТНЫМ окном.

    Чётное окно сдвигает результат на полотсчёта; на границах, которые мы
    ищем с точностью до нескольких миллисекунд, такой сдвиг систематический
    и целиком уходит в измеренную длительность.
    """
    n = max(1, n | 1)
    return np.convolve(x, np.ones(n) / n, mode="same")


def complete_leads(sig: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Достроить недостающие из шести фронтальных по I и II.

    Треугольник Эйнтховена: III, aVR, aVL, aVF выражаются через I и II точно,
    без всяких допущений. Если отведение прочиталось плохо, вычисленное будет
    даже честнее прочитанного.
    """
    idx = {n: i for i, n in enumerate(names)}
    if "I" not in idx or "II" not in idx:
        return sig, names
    a, b = sig[:, idx["I"]], sig[:, idx["II"]]
    derived = {"III": b - a, "aVR": -(a + b) / 2, "aVL": a - b / 2, "aVF": b - a / 2}
    out, out_names = list(sig.T), list(names)
    for k, v in derived.items():
        if k not in idx:
            out.append(v)
            out_names.append(k)
    return np.array(out).T, out_names


def _beat_window(rr_ms: float) -> tuple[int, int]:
    """Окно комплекса вокруг R, миллисекунды.

    Тянется за частотой: при редком ритме нужно захватить зубец P далеко до R
    и зубец T далеко после, при частом — нельзя залезть в соседний удар,
    иначе усреднение размажет чужой комплекс в свой же зубец T.
    """
    pre = min(400.0, max(250.0, rr_ms - 150.0))
    post = min(550.0, 0.70 * rr_ms)
    return int(pre), int(post)


def _clean_peaks(peaks: np.ndarray, fs: int) -> np.ndarray:
    """Выбросить чужеродные удары по интервалу до них.

    Экстрасистола приходит раньше срока и имеет другую форму; в среднем
    комплексе она размазывает и границы, и амплитуды. Отбрасываем по интервалу
    до соседнего удара: отклонение больше 15% от медианы — не наш.
    """
    if len(peaks) < 4:
        return peaks
    rr = np.diff(peaks)
    med = float(np.median(rr))
    keep = [peaks[0]]
    for i in range(1, len(peaks)):
        if 0.85 * med <= rr[i - 1] <= 1.15 * med:
            keep.append(peaks[i])
    return np.array(keep) if len(keep) >= 3 else peaks


def _lead_beats(
    sig: np.ndarray, names: list[str], fs: int, simultaneous: bool
) -> tuple[dict[str, np.ndarray], int, float, int] | None:
    """Представительный комплекс по каждому отведению + позиция R и средний RR.

    Два режима.

    simultaneous=True — отведения сняты одновременно (эталонная запись). Метки
    комплексов берём ОДИН раз по лучшему отведению и применяем ко всем: так
    комплексы выровнены точно, как в настоящем кардиографе.

    simultaneous=False — наш случай, раскладка 3x4. У каждой клетки своё время,
    общих меток нет. Ищем комплексы в каждом отведении отдельно, а потом
    совмещаем по всплеску энергии QRS (см. _align_by_energy).
    """
    span_len = {n: (lambda s: s[1] - s[0] if s else 0)(longest_run(sig[:, i]))
                for i, n in enumerate(names)}
    anchor = max(span_len, key=span_len.get)
    if span_len[anchor] < _ms(1200, fs):
        return None

    ai = names.index(anchor)
    a0, a1 = longest_run(sig[:, ai])
    apk = _clean_peaks(refine_peaks(sig[a0:a1, ai], r_peaks(sig[a0:a1, ai], fs), fs), fs)
    if len(apk) < 2:
        return None
    rr_ms = float(np.median(np.diff(apk))) * 1000.0 / fs
    pre, post = _beat_window(rr_ms)

    beats, counts = {}, {}
    for i, n in enumerate(names):
        span = longest_run(sig[:, i])
        if span is None or span[1] - span[0] < _ms(600, fs):
            continue
        seg = sig[span[0]:span[1], i]
        if simultaneous:
            # те же метки, пересчитанные в координаты этого куска
            pk = apk + (a0 - span[0])
            pk = pk[(pk >= 0) & (pk < len(seg))]
        else:
            pk = _clean_peaks(refine_peaks(seg, r_peaks(seg, fs), fs), fs)
        if len(pk) < 1:
            continue
        got = representative_beat(seg, pk, fs, pre_ms=pre, post_ms=post)
        if got:
            beats[n], counts[n] = got
    if len(beats) < 3:
        return None
    return beats, _ms(pre, fs), rr_ms, max(counts.values())


def _align_by_energy(beats: dict[str, np.ndarray], r_idx: int, fs: int,
                     max_ms: float = 30) -> dict[str, np.ndarray]:
    """Совместить комплексы разных отведений по всплеску энергии QRS.

    Каждый комплекс выровнен по вершине СВОЕГО зубца R, а вершины в разных
    отведениях приходятся на разные моменты — вектор поворачивается. Для
    пространственной скорости это яд: расхождение в 15 мс размажет фронт и
    удлинит измеренный QRS на те же 15 мс.

    Опорной берём огибающую |dx/dt|: всплеск деполяризации — общее для всех
    отведений событие, в отличие от вершины конкретного зубца.
    """
    env = {n: _smooth(np.abs(np.diff(w, prepend=w[0])), _ms(16, fs))
           for n, w in beats.items()}
    ref_name = max(env, key=lambda n: env[n].max())
    ref = env[ref_name]
    core = slice(max(0, r_idx - _ms(80, fs)), r_idx + _ms(80, fs))
    lim = _ms(max_ms, fs)

    # Сдвиг БЕЗ заворачивания. np.roll перенёс бы хвост окна в его начало —
    # то есть зубец T предыдущего удара оказался бы ровно там, где мы ищем
    # зубец P. Так и было: интервал PR вылезал за 300 мс на записях, где на
    # эталонном сигнале он мерился верно.
    out = {}
    for n, w in beats.items():
        best, best_score = 0, -np.inf
        for s in range(-lim, lim + 1):
            u = _shift(env[n], s)[core]
            v = ref[core]
            d = np.linalg.norm(u) * np.linalg.norm(v)
            score = float(u @ v / d) if d > 1e-12 else -np.inf
            if score > best_score:
                best_score, best = score, s
        out[n] = _shift(w, best)
    return out


def spatial_velocity(beats: dict[str, np.ndarray], fs: int,
                     smooth_ms: float = 12) -> np.ndarray | None:
    """Пространственная скорость по независимым отведениям."""
    use = [beats[n] for n in INDEP if n in beats]
    if len(use) < 2:
        use = list(beats.values())
    if not use:
        return None
    d = np.array([np.diff(w, prepend=w[0]) for w in use])
    return _smooth(np.sqrt((d ** 2).sum(0)), _ms(smooth_ms, fs))


def _walk(sv: np.ndarray, start: int, thr: float, step: int, hold: int) -> int:
    """Уйти от пика до места, где скорость упала ниже порога И там осталась.

    Условие "и там осталась" обязательно: внутри комплекса скорость проваливается
    между зубцами (например между R и S), и без выдержки граница села бы в эту
    ямку вместо настоящего конца.
    """
    i = start
    while 0 <= i < len(sv):
        if sv[i] < thr:
            j0, j1 = (i - hold, i) if step < 0 else (i, i + hold)
            if np.all(sv[max(0, j0):min(len(sv), j1)] < thr):
                return i
        i += step
    return 0 if step < 0 else len(sv) - 1


def qrs_bounds(sv: np.ndarray, r_idx: int, fs: int) -> tuple[int, int]:
    """Начало и конец комплекса по пространственной скорости."""
    core = slice(max(0, r_idx - _ms(70, fs)), min(len(sv), r_idx + _ms(70, fs)))
    peak_i = core.start + int(np.argmax(sv[core]))
    peak = float(sv[peak_i])
    # Шум меряем в конце окна — там уже всё закончилось; если пик едва выше
    # шума, порог поднимаем, иначе граница уползёт в шум на десятки мс.
    noise = float(np.median(sv[-_ms(60, fs):])) if len(sv) > _ms(60, fs) else 0.0
    hold = _ms(12, fs)
    on = _walk(sv, peak_i, max(K_QRS_ON * peak, 1.5 * noise), -1, hold)
    off = _walk(sv, peak_i, max(K_QRS_OFF * peak, 1.5 * noise), +1, hold)

    # Ограничитель хода. При частом ритме зубец P прижимается к комплексу, и
    # скорость между ними уже не успевает упасть до порога — поиск начала
    # проскакивает сквозь P и уезжает на сотню миллисекунд. Дальше предела
    # границу не пускаем, а берём самое тихое место внутри него: это всегда
    # промежуток PQ, даже когда он короткий.
    far_on, far_off = peak_i - _ms(120, fs), peak_i + _ms(160, fs)
    if on < far_on:
        lo = max(0, far_on)
        on = lo + int(np.argmin(sv[lo:peak_i])) if peak_i > lo else lo
    if off > far_off:
        hi = min(len(sv), far_off)
        off = peak_i + int(np.argmin(sv[peak_i:hi])) if hi > peak_i else hi
    return on, off


def p_magnitude(beats: dict[str, np.ndarray], base: dict[str, float],
                fs: int, bad: set[str] | None = None) -> np.ndarray | None:
    """Величина отклонения предсердного вектора по независимым отведениям.

    Корень из суммы квадратов отклонений от изолинии PQ — то есть длина вектора,
    а не его скорость. Для зубца P это принципиально: он мелкий и пологий, его
    производная тонет в шуме, и по пространственной скорости граница садилась то
    на артефакт-шип в одном отведении, то на середину пологого подъёма.
    А по величине зубец P виден ровно так же, как его читают глазами: линия
    отошла от изолинии — значит началось.
    """
    use = [np.asarray(beats[n], float) - base[n]
           for n in INDEP if n in beats and n not in (bad or set())]
    if len(use) < 2:
        use = [np.asarray(w, float) - base[n] for n, w in beats.items()]
    if not use:
        return None
    return _smooth(np.sqrt((np.array(use) ** 2).sum(0)), _ms(20, fs))


def _quiet(mag: np.ndarray, lo: int, hi: int, fs: int, win_ms: float = 40) -> float:
    """Шум величины: размах в самом спокойном месте окна.

    Нужен, чтобы отличить настоящий мелкий зубец P от дрожания линии. Мерить
    шум средним по всему окну нельзя — в окне лежат и зубец P, и хвост зубца T,
    и они бы этот «шум» и задали. А вот САМОЕ ТИХОЕ место окна — это отрезок TP,
    где предсердия уже молчат, а желудочки ещё не начали: там остаётся только
    дрожание. Скользим коротким окном и берём наименьший размах.
    """
    n = _ms(win_ms, fs)
    if hi - lo <= n:
        return float(np.ptp(mag[lo:hi])) if hi > lo else 0.0
    v = np.lib.stride_tricks.sliding_window_view(mag[lo:hi], n)
    return float(np.min(v.max(1) - v.min(1)))


def _trough(mag: np.ndarray, peak_i: int, limit: int, fs: int,
            hold_ms: float = 20) -> int:
    """Ближайшая к вершине впадина в сторону предела.

    Именно ближайшая, а не самая глубокая. Самую глубокую пробовал — она на
    быстром ритме оказывается за вершиной чужого зубца T, у дальнего края окна.
    Порог, отсчитанный от такого дна, лежит НИЖЕ этой вершины, и поиск начала
    зубца P спокойно проходит сквозь неё: длительность выходила 295-298 мс и
    запись отбраковывалась как невозможная. Опора должна быть той впадиной, из
    которой зубец P поднялся, — то есть отрезком TP, а он рядом с ним.

    Подъём подтверждаем выдержкой: mag сглажена, но мелкая рябь на ней
    остаётся, и без подтверждения дном объявлялась бы первая же рябинка.
    """
    step = 1 if limit > peak_i else -1
    n = _ms(hold_ms, fs)
    best = i = peak_i
    while i != limit:
        i += step
        if mag[i] <= mag[best]:
            best = i
        elif abs(i - best) >= n:
            break                       # держимся выше дна — значит оно позади
    return best


def p_bounds(mag: np.ndarray, qrs_on: int, fs: int,
             rr_ms: float) -> tuple[int, int] | None:
    """Начало и конец зубца P по величине отклонения.

    Ищем ВЫПУКЛОСТЬ горба, а не его высоту над общим уровнем окна. Разница
    решающая. В окно поиска при ЧСС от 80 попадает хвост зубца T предыдущего
    удара, а он в разы крупнее зубца P. Пока порог кандидата считался в долях
    размаха окна, его задавал этот чужой зубец T, и настоящий зубец P до порога
    просто не дотягивал: на 40 записях PTB-XL так терялось 6 зубцов из 35, все
    — совершенно обычные, 98-126 мс. Выпуклость же меряет, насколько горб
    поднялся над СВОИМИ окрестностями, и что там творится в другом конце окна,
    её не касается.

    По той же причине опора для порога у каждой границы своя: слева — низшая
    точка между зубцом T и зубцом P, справа — промежуток PQ. Общая опора «тихое
    место окна» на быстром ритме не находилась вовсе, поиск упирался в край
    окна, и запись выбрасывалась целиком.
    """
    lo = max(0, qrs_on - _ms(min(340.0, 0.50 * rr_ms), fs))
    hi = qrs_on - _ms(25, fs)
    if hi - lo < _ms(80, fs):
        return None

    # Порог выпуклости: либо предел различимости на плёнке, либо дрожание линии
    # с запасом — что больше. Второе включается на шумной оцифровке, где доли
    # миллиметра с плёнки уже неотличимы от ряби.
    #
    # Требование к ШИРИНЕ горба пробовал добавить — идея была отсеивать рябь
    # оцифровки, которая узкая, а зубец P идёт 80-140 мс. Замер показал вред:
    # при 45 мс находимость упала с 40 до 39 записей на эталоне и с 39 до 34
    # после оцифровки, а ложных срабатываний убавилось всего с 15 до 14. При
    # 70 мс находимость обвалилась втрое. Причина в том, что ширина у scipy
    # меряется на половине выпуклости, и у зубца P, сидящего на спаде зубца T,
    # она куда меньше самого зубца. Убрано.
    prom = max(P_MIN_MV, P_SNR * _quiet(mag, lo, hi, fs))
    cand = list(lo + find_peaks(mag[lo:hi], prominence=prom)[0])

    # Зубец P вплотную к комплексу: его горб срезан правым краем окна и
    # максимумом не является. Признак — линия у края всё ещё идёт вверх.
    edge = hi - 1
    rise = _ms(20, fs)
    if (mag[edge] - float(np.min(mag[lo:hi])) >= prom
            and edge - rise >= lo and mag[edge] > mag[edge - rise]):
        cand.append(edge)
    if not cand:
        return None                    # зубца P не видно: мерцание, узловой ритм

    # Берём ПОСЛЕДНИЙ горб перед комплексом, а не самый высокий: зубец P —
    # всегда последнее событие перед деполяризацией желудочков, а выше него
    # запросто оказывается и зубец T, и артефакт оцифровки.
    peak_i = int(cand[-1])
    peak = float(mag[peak_i])

    def bound(limit: int, step: int) -> int:
        """Дойти от вершины до порога над МЕСТНОЙ опорой, не дальше предела."""
        foot = _trough(mag, peak_i, limit, fs)
        floor = float(mag[foot])
        thr = floor + K_P * (peak - floor)
        i = peak_i
        while i != foot:
            if mag[i] < thr:
                return i
            i += step
        return foot

    on, off = bound(lo, -1), bound(hi - 1, +1)
    # Начало упёрлось в край окна — значит его не видно, а видно только то, что
    # горб продолжается за границу поиска. Мерить там нечего: интервал PQ вышел
    # бы ровно равен ширине окна, то есть был бы не измерением, а её отпечатком.
    # Так ловится хвост чужого зубца T, дотянувшийся до края: из пяти ложных
    # срабатываний на записях со стёртым зубцом P три давали PQ 338-340 мс при
    # пределе окна 340.
    if on <= lo or off - on < _ms(40, fs) or off - on > _ms(220, fs):
        return None
    return on, off


# Предел различимости зубца P: 0.4 мм на плёнке. Опускать пробовал — при 0.2 мм
# находимость после оцифровки поднимается на одну запись из сорока, а ложных
# срабатываний прибавляется шесть. Мена невыгодная.
P_MIN_MV = 0.04
# Во сколько раз горб должен превышать дрожание линии. Дрожание меряется как
# размах на самом тихом отрезке (_quiet), а размах для шума — это примерно
# четыре сигмы, так что двукратный запас здесь — это уже восемь сигм.
P_SNR = 2.0
P_MIN_LEADS = 2                 # и он должен быть виден не в одном отведении


def p_confirmed(beats: dict[str, np.ndarray], base: dict[str, float],
                p: tuple[int, int], bad: set[str]) -> bool:
    """Проверить, что найденный всплеск — действительно зубец P.

    Пространственная скорость говорит только «здесь что-то шевельнулось». Этого
    мало: при мерцательной аритмии предсердия шевелятся непрерывно, и всплеск
    находится всегда, хотя зубца P нет вовсе. Поэтому спрашиваем отдельно, есть
    ли на найденном месте НАСТОЯЩЕЕ отклонение кривой.

    Два условия, и оба нужны.

    Величина. Зубец P мельче 0.4 мм на плёнке неразличим и в бумажном чтении.

    Не меньше двух отведений. Волны мерцания и шум оцифровки в разных
    отведениях независимы и после усреднения по ударам гасятся, а настоящий
    зубец P — общее событие: он переживает усреднение и виден сразу в
    нескольких отведениях. Одно отведение — это почти наверняка случайность.
    """
    a, b = p
    seen = 0
    for n, w in beats.items():
        if n in bad or b <= a or b > len(w):
            continue
        if float(np.max(np.abs(w[a:b] - base[n]))) >= P_MIN_MV:
            seen += 1
    return seen >= P_MIN_LEADS


def _baseline(w: np.ndarray, qrs_on: int, fs: int) -> float:
    """Изолиния: ровный участок PQ прямо перед комплексом.

    Именно PQ, а не «среднее по удару»: от этого уровня отсчитывается смещение
    ST, и любой другой выбор сместил бы все ST разом.
    """
    a = max(0, qrs_on - _ms(30, fs))
    return float(np.median(w[a:max(a + 1, qrs_on)]))


def t_window(sv: np.ndarray, qrs_off: int, fs: int,
             rr_ms: float) -> tuple[int, int] | None:
    """Где вообще искать зубец T — по пространственной скорости.

    Без этой рамки поиск вершины «по максимуму отклонения» регулярно ловит
    зубец U: в отведениях с мелким зубцом T (I, V5, V6) волна U бывает выше
    самого T, и вершина уезжает на 300+ мс. Волна U при этом почти не даёт
    вклада в пространственную скорость — она пологая и не синхронна между
    отведениями, — поэтому рамка по sv её отсекает.
    """
    lo = qrs_off + _ms(50, fs)
    hi = min(len(sv), qrs_off + _ms(min(600.0, 0.68 * rr_ms), fs))
    if hi - lo < _ms(100, fs):
        return None
    sp = _smooth(sv, _ms(30, fs))
    floor = float(np.percentile(sp[lo:hi], 20))
    peak_i = lo + int(np.argmax(sp[lo:hi]))
    peak = float(sp[peak_i])
    if peak < 1.5 * max(floor, 1e-12):
        return None
    thr = floor + 0.25 * (peak - floor)
    # Выдержка длинная: у зубца T два всплеска скорости — подъём и спад, — и
    # между ними на вершине провал. С короткой выдержкой конец сел бы в него.
    end = _walk(sp[:hi], peak_i, thr, +1, _ms(60, fs))
    return lo, min(hi, max(end, peak_i + _ms(40, fs)))


def t_offset(w: np.ndarray, t_lo: int, t_hi: int, fs: int,
             base: float) -> tuple[int, int, float] | None:
    """Вершина и конец зубца T методом касательной.

    Конец T — не там, где кривая «легла»: у пологого зубца это место
    определяется шумом и гуляет на сотню миллисекунд. Общепринятый приём:
    провести касательную в точке самого крутого спада зубца и взять её
    пересечение с изолинией. Такая точка устойчива, потому что задаётся
    участком с максимальной производной, а не хвостом.
    """
    t_hi = min(t_hi, len(w))
    if t_hi - t_lo < _ms(60, fs):
        return None
    t_i = t_lo + int(np.argmax(np.abs(w[t_lo:t_hi] - base)))
    amp = float(w[t_i] - base)

    # Сглаживание 30 мс, а не 16: производная на спаде зубца T шумит, и точка
    # «максимальной крутизны» прыгает. На 40 записях разброс конца T упал с 8
    # до 6 мс — крутизну надо мерить по участку, а не по паре отсчётов.
    d = np.diff(_smooth(w, _ms(30, fs)), prepend=w[0])
    tail = slice(t_i + _ms(10, fs), min(len(w), t_hi + _ms(40, fs)))
    if tail.stop - tail.start < _ms(20, fs):
        return None
    # спад по модулю: у отрицательного зубца T «спад» — это подъём вверх
    k_i = tail.start + int(np.argmax(-np.sign(amp) * d[tail]))
    slope = float(d[k_i])
    if abs(slope) < 1e-9:
        return None
    end = k_i + (base - w[k_i]) / slope
    # Пологий спад даёт касательную, уходящую на полсекунды вперёд. Такое
    # значение не «примерно верное», а бессмысленное — отбрасываем целиком,
    # раньше оно молча обрезалось по краю окна и выглядело как измерение.
    if not (t_i < end < k_i + _ms(160, fs)) or end >= len(w):
        return None
    return t_i, int(round(end)), amp


def _net(w: np.ndarray, a: int, b: int, base: float) -> float:
    """Итоговое отклонение на интервале: вершина вверх плюс вершина вниз.

    Для комплекса это R + S (S отрицательный) — школьная «алгебраическая сумма
    зубцов», по ней ось и определяют.

    Пробовал считать по площади под кривой — теоретически она правильнее,
    ведь отражает весь фронт, а не один момент. На 39 записях против разметки
    Glasgow вышло хуже вдвое: медиана 5.3° против 2.7°, худший случай 78°
    против 47°. Площадь чувствительна к выбору изолинии и к тому, где именно
    поставлена граница комплекса, а амплитуда — нет. Оставил амплитуду.
    """
    if b <= a:
        return 0.0
    seg = w[a:b] - base
    return float(np.max(seg) + np.min(seg))


def axis_deg(a_i: float, a_avf: float) -> float | None:
    """Электрическая ось во фронтальной плоскости, градусы.

    Наивная формула atan2(aVF, I) неверна: усиленные отведения меряют проекцию
    в масштабе sqrt(3)/2 от двухполюсных. Из треугольника Эйнтховена
    aVF = (II + III)/2 = Hy*sqrt(3)/2, тогда как I = Hx. Без деления на
    sqrt(3)/2 ось врёт до 4 градусов — на границе нормы это меняет заключение.
    """
    hx, hy = a_i, a_avf / (np.sqrt(3) / 2)
    if abs(hx) < 1e-9 and abs(hy) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(hy, hx)))


def qtc(qt: float, rr_ms: float) -> dict[str, float]:
    """Поправки QT на частоту. Даём четыре, а не одну: они расходятся.

    Базетта завышает при тахикардии и занижает при брадикардии — при ЧСС 100
    разница с Фридерисией доходит до 20 мс. Одна цифра создала бы ложную
    уверенность, поэтому показываем разброс.
    """
    rr = rr_ms / 1000.0
    hr = 60.0 / rr
    return {"bazett": qt / np.sqrt(rr), "fridericia": qt / rr ** (1 / 3),
            "framingham": qt + 154 * (1 - rr), "hodges": qt + 1.75 * (hr - 60)}


# Какую долю от самого быстрого движения в отведении должен давать комплекс,
# чтобы считать удар в этом отведении совмещённым с остальными.
MIN_QRS_SHARE = 0.35

SOKOLOW_MM = 35.0                   # порог Соколова-Лайона, мм плёнки


def sokolow_lyon(amp: dict[str, dict]) -> dict | None:
    """Индекс Соколова-Лайона: глубина S в V1 плюс высота R в V5 или V6.

    Смысл в том, что при утолщении левого желудочка вектор деполяризации
    сильнее уходит влево-назад: в V1 (справа спереди) он даёт более глубокий S,
    в V5-V6 (слева) — более высокий R. Складывают именно эти два, потому что
    складываются и вклады. Порог 35 мм на плёнке = 3.5 мВ.

    Индекс целиком держится на ВЫСОТЕ зубцов, поэтому он тут не случайно: это
    прямая проверка того, ради чего восстанавливались срезанные вершины.
    Занижение амплитуды на 10% сдвигает индекс на 3-4 мм — как раз через порог.
    """
    s1 = (amp.get("V1") or {}).get("s")
    r5 = (amp.get("V5") or {}).get("r")
    r6 = (amp.get("V6") or {}).get("r")
    lat = [(v, n) for v, n in ((r5, "V5"), (r6, "V6")) if v is not None]
    if s1 is None or not lat:
        return None
    r, lead = max(lat)
    mm = (abs(s1) + max(r, 0.0)) * 10.0            # мВ -> мм при 10 мм/мВ
    return {"mm": mm, "lead": lead, "over": mm >= SOKOLOW_MM,
            "s_v1_mm": abs(s1) * 10.0, "r_mm": max(r, 0.0) * 10.0}


def measure(sig: np.ndarray, names: list[str], fs: int,
            simultaneous: bool = False) -> dict | None:
    """Полный набор измерений по оцифрованному сигналу.

    Возвращает и сами числа, и представительные комплексы с метками границ —
    чтобы измерение можно было увидеть глазами, а не принимать на веру.
    """
    sig, names = complete_leads(np.asarray(sig, float), list(names))
    got = _lead_beats(sig, names, fs, simultaneous)
    if got is None:
        return None
    beats, r_idx, rr_ms, n_beats = got
    if not simultaneous:
        beats = _align_by_energy(beats, r_idx, fs)

    # Недостающие фронтальные достраиваем уже по СОВМЕЩЁННЫМ комплексам I и II.
    # Иначе одно плохо прочитанное отведение от конечностей забирало бы с собой
    # электрическую ось целиком — а она считается по I и aVF, и aVF выражается
    # через первые два точно.
    if "I" in beats and "II" in beats:
        a, b = beats["I"], beats["II"]
        for name, w in (("III", b - a), ("aVR", -(a + b) / 2),
                        ("aVL", a - b / 2), ("aVF", b - a / 2)):
            beats.setdefault(name, w)

    sv = spatial_velocity(beats, fs)
    if sv is None:
        return None
    qrs_on, qrs_off = qrs_bounds(sv, r_idx, fs)

    # Второй проход: выбрасываем отведения, у которых самое быстрое отклонение
    # оказалось вообще не в комплексе. Комплекс — самое быстрое событие на ЭКГ,
    # и если в отведении это не так, значит его удар встал мимо остальных: линия
    # прочиталась с разрывом, или в клетку попал кусок соседнего отведения.
    # Такое отведение тянет пространственную скорость на себя и портит границы
    # СРАЗУ ВСЕМ, поэтому границы пересчитываем без него.
    # Спрашиваем не «где самое быстрое отклонение», а «есть ли в комплексе
    # заметная доля быстрого движения». Первое сбивается одиночным выбросом
    # оцифровки: узкий шип в конце окна круче, чем невысокий комплекс в этом
    # отведении, — и совершенно нормальное отведение отбраковывалось бы.
    # Края окна не смотрим: там кривая достроена плоским продолжением, и стык
    # сам по себе даёт всплеск производной.
    m0, m1 = _ms(50, fs), len(sv) - _ms(50, fs)
    bad = set()
    for n, w in beats.items():
        env = _smooth(np.abs(np.diff(w, prepend=w[0])), _ms(16, fs))
        peak = float(np.max(env[m0:m1]))
        here = float(np.max(env[max(m0, qrs_on):max(m0 + 1, qrs_off)]))
        if peak > 1e-12 and here < MIN_QRS_SHARE * peak:
            bad.add(n)
    good = {n: w for n, w in beats.items() if n not in bad}
    if bad and len(good) >= 4:
        sv2 = spatial_velocity(good, fs)
        if sv2 is not None:
            sv = sv2
            qrs_on, qrs_off = qrs_bounds(sv, r_idx, fs)
    else:
        bad = set()

    if qrs_off - qrs_on < _ms(40, fs) or qrs_off - qrs_on > _ms(220, fs):
        return None

    base = {n: _baseline(w, qrs_on, fs) for n, w in beats.items()}
    mag = p_magnitude(beats, base, fs, bad)
    p = p_bounds(mag, qrs_on, fs, rr_ms) if mag is not None else None
    if p and not p_confirmed(beats, base, p, bad):
        p = None

    # Конец T ищем по каждому отведению и берём медиану: в отдельном отведении
    # зубец T может быть почти плоским, и касательная там ловит шум. Медиана по
    # восьми независимым устойчива к паре таких промахов.
    tw = t_window(sv, qrs_off, fs, rr_ms)
    found = []
    if tw:
        for n in INDEP:
            if n in beats and n not in bad:
                r = t_offset(beats[n], tw[0], tw[1], fs, base[n])
                if r:
                    found.append((n, *r))
    # Отведения с мелким зубцом T выбрасываем: касательная к почти плоской
    # кривой задаётся шумом, и такие отведения тянут медиану вразнос. Порог
    # относительный, потому что общая величина T у разных людей разная.
    t_end = t_peak = None
    if found:
        big = max(abs(a) for *_, a in found)
        keep = [f for f in found if abs(f[3]) >= max(0.10, 0.35 * big)]
        if len(keep) >= 2:
            t_peak = int(np.median([f[1] for f in keep]))
            t_end = int(np.median([f[2] for f in keep])) + _ms(T_END_MS, fs)
            t_end = min(t_end, len(sv) - 1)

    to_ms = lambda n: n * 1000.0 / fs                       # noqa: E731
    qrs_ms = to_ms(qrs_off - qrs_on)
    p_ms = to_ms(p[1] - p[0]) if p else None
    pr_ms = to_ms(qrs_on - p[0]) if p else None
    qt_ms = to_ms(t_end - qrs_on) if t_end else None

    # Ось: алгебраическая сумма зубцов в I и aVF — по комплексу, по зубцу P и
    # по зубцу T отдельно.
    ax = {}
    for tag, (a, b) in {"qrs": (qrs_on, qrs_off),
                        "p": p[:2] if p else (0, 0),
                        "t": ((qrs_off + _ms(60, fs), t_end) if t_end else (0, 0))}.items():
        if b <= a or "I" not in beats or "aVF" not in beats:
            ax[tag] = None
            continue
        ax[tag] = axis_deg(_net(beats["I"], a, b, base["I"]),
                           _net(beats["aVF"], a, b, base["aVF"]))

    # ST: уровень через 60 и 80 мс после конца комплекса, от изолинии PQ.
    # Две точки, потому что при тахикардии в J+80 уже начинается зубец T.
    # Величины по несовмещённым отведениям не считаем: их пришлось бы снимать
    # в точках, которые к их же кривой отношения не имеют.
    st = {}
    for n, w in beats.items():
        row = {}
        for tag, off in (("j", 0), ("j60", 60), ("j80", 80)):
            i = qrs_off + _ms(off, fs)
            row[tag] = (float(w[i] - base[n])
                        if n not in bad and 0 <= i < len(w) else None)
        st[n] = row

    amp = {n: ({"r": None, "s": None, "t": None} if n in bad else
               {"r": float(np.max(w[qrs_on:qrs_off] - base[n])),
                "s": float(np.min(w[qrs_on:qrs_off] - base[n])),
                "t": (float(w[t_peak] - base[n])
                      if t_peak is not None and t_peak < len(w) else None)})
           for n, w in beats.items()}

    # Отметки «этому числу верить нельзя». Показать сомнительное измерение с
    # пометкой честнее, чем спрятать его или выдать за надёжное.
    qc = qtc(qt_ms, rr_ms) if qt_ms else None
    flags = []
    if qt_ms is None:
        flags.append("t")
    elif not (QTC_RANGE[0] <= qc["bazett"] <= QTC_RANGE[1]):
        flags.append("qt")
    if pr_ms is None:
        flags.append("p")
    if n_beats < 3:
        flags.append("beats")
    if len(beats) < 6:
        flags.append("leads")

    return {
        "fs": fs, "rr_ms": rr_ms, "hr": 60000.0 / rr_ms,
        "p_ms": p_ms, "pr_ms": pr_ms, "qrs_ms": qrs_ms, "qt_ms": qt_ms,
        "qtc": qc, "axis": ax, "st": st, "amp": amp, "flags": flags,
        "n_beats": n_beats, "misaligned": sorted(bad),
        "sokolow": sokolow_lyon(amp),
        "marks": {"p_on": p[0] if p else None, "p_off": p[1] if p else None,
                  "qrs_on": qrs_on, "qrs_off": qrs_off,
                  "t_peak": t_peak, "t_off": t_end, "r": r_idx},
        "beats": beats, "baseline": base, "sv": sv,
        "simultaneous": simultaneous,
        "n_leads": len(beats),
    }


def numbers(m: dict | None) -> dict | None:
    """Только числа, без массивов — для result.json и внешних потребителей."""
    if not m:
        return None
    ms = lambda i: (None if i is None else i * 1000.0 / m["fs"])   # noqa: E731
    return {k: m[k] for k in ("hr", "rr_ms", "p_ms", "pr_ms", "qrs_ms", "qt_ms", "qtc",
                              "axis", "st", "amp", "flags", "n_beats",
                              "n_leads", "simultaneous")} | {
        "marks_ms": {k: ms(v) for k, v in m["marks"].items()},
    }
