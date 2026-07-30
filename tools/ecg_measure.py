"""Основа измерений: ЧСС и представительный (усреднённый) комплекс.

Отдельный модуль, а не кусок стенда: этим же кодом считает и сайт, и проверка
на PTB-XL, иначе они разъедутся и проверка перестанет что-либо значить.
Разметка комплекса и сами интервалы — в tools/ecg_delineate.py, здесь только
то, на чём они стоят.

Проверено стендом tools/validate_ptbxl.py на 40 записях PTB-XL: медиана ошибки
ЧСС 0.1 уд/мин, все 40 в пределах 5 уд/мин от эталона.

Demo-инструмент, НЕ медизделие.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

MIN_BPM, MAX_BPM = 30.0, 200.0


def longest_run(col: np.ndarray) -> tuple[int, int] | None:
    """Самый длинный непрерывный кусок без пропусков."""
    ok = ~np.isnan(np.asarray(col, float))
    if not ok.any():
        return None
    d = np.diff(np.concatenate([[0], ok.view(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def r_peaks(seg: np.ndarray, fs: int) -> np.ndarray:
    """Положения комплексов, схема Пана-Томпкинса.

    Модуль сигнала брать нельзя: у комплекса с глубоким зубцом S получилось бы
    два пика вместо одного и частота удвоилась бы. Поэтому полоса 5-15 Гц (там
    живёт QRS и почти нет ни зубца T, ни дрейфа изолинии), производная, квадрат
    и скользящее окно — остаются ровно всплески комплексов.
    """
    # 1.2 с достаточно: в раскладке 3x4 отведение видно всего 2.5 с, и требовать
    # больше значило бы не измерять ничего, кроме ритм-строки.
    if len(seg) < int(1.2 * fs):
        return np.array([], int)
    b, a = butter(3, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype="band")
    y = filtfilt(b, a, seg - np.median(seg))
    y = np.diff(y, prepend=y[0]) ** 2
    w = max(1, int(0.15 * fs))
    y = np.convolve(y, np.ones(w) / w, mode="same")
    pk, _ = find_peaks(y, height=0.3 * np.percentile(y, 98), distance=int(0.28 * fs))
    return pk


def heart_rate(x: np.ndarray, fs: int) -> dict | None:
    """ЧСС по самому длинному непрерывному куску отведения.

    Возвращает частоту, число ударов, длину использованного куска и признак
    ровности ритма. Ровность важна: при мерцательной аритмии одно среднее
    число вводит в заблуждение, интервалы там гуляют.
    """
    span = longest_run(x)
    if span is None:
        return None
    seg = np.asarray(x, float)[span[0]:span[1]]
    pk = r_peaks(seg, fs)
    if len(pk) < 3:
        return None
    rr = np.diff(pk) / fs
    rr = rr[(rr > 60.0 / MAX_BPM) & (rr < 60.0 / MIN_BPM)]
    if len(rr) < 2:
        return None
    med = float(np.median(rr))
    # разброс интервалов относительно среднего: у ровного синусового ритма
    # единицы процентов, при мерцании — десятки
    spread = float(np.median(np.abs(rr - med)) / med)
    return {"bpm": round(60.0 / med), "beats": int(len(pk)),
            "seconds": round(len(seg) / fs, 1),
            "spread": round(100 * spread), "regular": spread < 0.12}


PRE_MS, POST_MS = 300, 500          # окно комплекса вокруг зубца R

# Полоса у края клетки, комплексы из которой берём в последнюю очередь.
# 200 мс — примерно длительность комплекса вместе с зубцом P: если центр удара
# ближе к краю, часть его почти наверняка обрезана или досталась от соседа.
EDGE_MS = 200


def _highpass(x: np.ndarray, fs: int, win_ms: float = 120) -> np.ndarray:
    """Убрать медленную составляющую: сигнал минус своё скользящее среднее.

    Окно 120 мс подобрано по ширине событий: комплекс QRS в него не влезает и
    остаётся почти целиком, а зубец T и сегмент ST — события в три-четыре раза
    более широкие — срезаются.
    """
    n = max(1, int(win_ms * fs / 1000) | 1)
    ok = ~np.isnan(x)
    filled = np.where(ok, x, 0.0)
    k = np.ones(n)
    s = np.convolve(filled, k, mode="same")
    c = np.convolve(ok.astype(float), k, mode="same")
    return x - np.divide(s, c, out=np.zeros_like(s), where=c > 0)


def refine_peaks(seg: np.ndarray, peaks: np.ndarray, fs: int,
                 half_ms: float = 50) -> np.ndarray:
    """Уточнить метки до настоящей вершины комплекса.

    Пан-Томпкинс ставит метку по огибающей, а она отстаёт и гуляет на десятки
    миллисекунд. Для усреднения этого мало: рассинхрон в 20 мс размажет ту
    самую вершину, которую мы отдельно восстанавливали.

    Вершину ищем по сигналу БЕЗ медленной составляющей, а не по самому большому
    отклонению от изолинии. При высоком зубце T или выраженной элевации ST
    именно они и оказываются самым большим отклонением: в грудных отведениях с
    мелким зубцом R метка уезжала на 50 мс — ровно на край окна поиска, — и
    комплекс в этом отведении вставал в среднем удар на сто миллисекунд мимо
    остальных. На картинке это видно сразу: отметки границ стоят не на том
    месте кривой.
    """
    half = int(half_ms * fs / 1000)
    near = max(1, int(25 * fs / 1000))
    fast = _highpass(seg, fs)
    base = np.nanmedian(seg)
    out = []
    for p in peaks:
        a, b = max(0, p - half), min(len(seg), p + half + 1)
        w = fast[a:b]
        if not len(w) or np.all(np.isnan(w)):
            continue
        # Сначала по «быстрому» сигналу находим, ГДЕ комплекс, — сюда зубец T
        # уже не дотягивается. А вершину берём на исходной кривой рядом с этим
        # местом: усреднять удары надо по настоящей вершине, иначе теряется
        # ровно та амплитуда, которую отдельно восстанавливали.
        j = a + int(np.nanargmax(np.abs(w)))
        c, d = max(0, j - near), min(len(seg), j + near + 1)
        v = seg[c:d]
        if np.all(np.isnan(v)):
            continue
        out.append(c + int(np.nanargmax(np.abs(v - base))))
    return np.array(sorted(set(out)), dtype=int)


def _detrend_beat(w: np.ndarray, fs: int, edge_ms: float = 40) -> np.ndarray:
    """Убрать дрейф изолинии внутри окна удара.

    Без этого удары не сравнить между собой: за 800 мс изолиния уезжает, и
    корреляция падает до 0.86-0.92 на совершенно нормальных ударах — из-за
    дрейфа, а не из-за разной формы.
    """
    k = max(2, int(edge_ms * fs / 1000))
    ok = np.flatnonzero(~np.isnan(w))
    if len(ok) < 2 * k:
        return w
    # Опираемся на КРАЯ ИМЕЮЩИХСЯ данных, а не на края массива: у окна, часть
    # которого вышла за границу куска, там пусто, и наклон считался бы по
    # пустоте.
    a, b = int(ok[0]), int(ok[-1])
    y0, y1 = float(np.nanmedian(w[a:a + k])), float(np.nanmedian(w[b - k + 1:b + 1]))
    ramp = np.interp(np.arange(len(w)), [a, b], [y0, y1])
    return w - ramp


def _shift(x: np.ndarray, s: int) -> np.ndarray:
    """Сдвиг БЕЗ заворачивания.

    np.roll переносит хвост в начало и делает шов на стыке: дальше этот скачок
    даёт ложный всплеск производной, а он идёт прямо в кривую пространственной
    скорости, по которой ищутся границы комплекса. Края добираем краевым
    значением — они всё равно исключаются из измерений.
    """
    if s == 0:
        return x.copy()
    out = np.empty_like(x)
    if s > 0:
        out[:s] = x[0]
        out[s:] = x[:-s]
    else:
        out[s:] = x[-1]
        out[:s] = x[-s:]
    return out


def _align_shift(x: np.ndarray, ref: np.ndarray, max_shift: int,
                 core: slice | None = None) -> int:
    """Подгонка удара к шаблону по НОРМИРОВАННОЙ корреляции комплекса.

    Считаем по узкому окну вокруг QRS, а не по всему удару: совмещать надо
    именно комплексы, а зубцы P и T только тянут подгонку на себя.
    """
    best, best_score = 0, -np.inf
    for s in range(-max_shift, max_shift + 1):
        a = _shift(x, s)
        u, v = (a[core], ref[core]) if core else (a, ref)
        d = np.std(u) * np.std(v)
        score = float(np.mean((u - u.mean()) * (v - v.mean())) / d) if d > 1e-9 else -np.inf
        if score > best_score:
            best_score, best = score, s
    return best


def representative_beat(
    seg: np.ndarray, peaks: np.ndarray, fs: int, min_corr: float = 0.9,
    pre_ms: float = PRE_MS, post_ms: float = POST_MS,
) -> tuple[np.ndarray, int, np.ndarray] | None:
    """Представительный комплекс: удары одного семейства, усреднённые СРЕДНИМ.

    Возвращает вместе со средним и САМИ удары — стопкой, уже совмещённые и
    отобранные. Нужны они для одной проверки, которую по среднему сделать
    нельзя: связан ли найденный зубец P с комплексом. Зубец P приходит перед
    каждым ударом в одно и то же время, поэтому усреднение его сохраняет; волны
    мерцания предсердий с ударами не связаны и при усреднении гаснут как корень
    из числа ударов. По одному только среднему эти два случая неразличимы — там
    и там виден горб.

    Именно средним, а не медианой: шум в отсчёте гасится как корень из числа
    ударов, а медиана такого выигрыша не даёт и вдобавок рвёт гладкость формы.
    Так же поступает Philips DXL. Плата за среднее — чувствительность к
    выбросам, поэтому чужеродные удары (экстрасистолы, артефакты) сначала
    отбрасываются по несхожести с предварительным средним.
    """
    pre, post = int(pre_ms * fs / 1000), int(post_ms * fs / 1000)
    peaks = refine_peaks(seg, peaks, fs)

    # Комплексы у самых краёв клетки берём только если других нет. Границы
    # клеток на плёнке режутся неточно, и с краю запросто оказывается кусок
    # комплекса из СОСЕДНЕГО отведения — а он попадёт в среднее как свой и
    # испортит и форму, и все границы. Середина клетки такого риска не несёт.
    edge = int(EDGE_MS * fs / 1000)
    inner = peaks[(peaks >= edge) & (peaks <= len(seg) - edge)]
    if len(inner) >= 2:
        peaks = inner

    beats = []
    for p in peaks:
        # Окно за краем не выбрасываем, а добираем пустотой: в клетке 3x4 всего
        # 2.5 с, и целых окон там от силы одно — усреднять было бы нечего.
        # Требуем только, чтобы сам комплекс (±120 мс) был на месте.
        if p - int(0.12 * fs) < 0 or p + int(0.12 * fs) > len(seg):
            continue
        w = np.full(pre + post, np.nan)
        a, b = p - pre, p + post
        src = seg[max(a, 0):min(b, len(seg))]
        w[max(0, -a):max(0, -a) + len(src)] = src
        if np.any(np.isnan(w[pre - int(0.12 * fs):pre + int(0.12 * fs)])):
            continue
        # Пустоту так и оставляем пустотой. Заполнять её числом (медианой окна)
        # нельзя: на стыке заполнения с настоящим сигналом получается СТУПЕНЬКА,
        # а производная от ступеньки — это всплеск в пространственной скорости
        # ровно там, где мы потом ищем зубец P. На плёнке 3x4, где в клетке
        # 2.5 с и почти каждое окно обрезано, интервал PR из-за этого вылезал
        # за 300 мс при настоящих 150.
        beats.append(_detrend_beat(w, fs))
    if not beats:
        return None
    if len(beats) == 1:                       # 2.5 с могут дать всего один удар
        return _fill_edges(beats[0]), 1, np.asarray(beats, float)

    beats = np.array(beats)
    core = slice(pre - int(0.06 * fs), pre + int(0.06 * fs))    # ±60 мс вокруг R
    tmpl = _mean_ignoring_gaps(beats)
    for _ in range(2):                        # шаблон уточняем: сдвиги влияют на него
        aligned = [_shift(w, _align_shift(w, tmpl, int(0.03 * fs), core)) for w in beats]
        tmpl = _mean_ignoring_gaps(aligned)

    keep = []
    for w in aligned:
        c = _nancorr(w, tmpl)
        if c is not None and c >= min_corr:
            keep.append(w)
    if len(keep) < 2:                         # семейство не сложилось — берём лучший
        cors = [_nancorr(w, tmpl) or -1.0 for w in aligned]
        keep = [aligned[int(np.argmax(cors))]]
    return (_fill_edges(_mean_ignoring_gaps(keep)), len(keep),
            np.asarray(keep, float))


def _nancorr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Корреляция по общим непустым отсчётам."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 8:
        return None
    u, v = a[ok], b[ok]
    d = np.std(u) * np.std(v)
    return float(np.corrcoef(u, v)[0, 1]) if d > 1e-9 else None


def _mean_ignoring_gaps(rows: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Среднее по ударам там, где данные есть; где ни у кого нет — пусто.

    Через np.nanmean напрямую нельзя: на полностью пустом столбце он не просто
    вернёт пустоту, а ещё и напечатает предупреждение — а пустые столбцы здесь
    норма, в клетке 3x4 края окна почти всегда обрезаны.
    """
    a = np.asarray(rows, float)
    ok = ~np.isnan(a)
    n = ok.sum(0)
    out = np.full(a.shape[1], np.nan)
    np.divide(np.where(ok, a, 0).sum(0), n, out=out, where=n > 0)
    return out


def _fill_edges(w: np.ndarray) -> np.ndarray:
    """Досыпать оставшуюся пустоту КРАЕВЫМ значением.

    Плоское продолжение, а не число со стороны: у плоского участка производная
    ноль, и он не создаёт ложной границы. Если пусто вообще всё — сдаёмся.
    """
    ok = np.flatnonzero(~np.isnan(w))
    if not len(ok):
        return w
    out = w.copy()
    out[:ok[0]] = w[ok[0]]
    out[ok[-1] + 1:] = w[ok[-1]]
    # Дырки внутри (их быть не должно, но пусть) закрываем линейно.
    bad = np.isnan(out)
    if bad.any():
        out[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), out[~bad])
    return out


def best_lead(sig: np.ndarray, leads: list[str]) -> int | None:
    """Отведение с самым длинным непрерывным куском — по нему и считаем.

    ЧСС не зависит от отведения, поэтому берём то, где сигнала больше всего:
    в раскладке 3x4 это ритм-строка на все 10 с.
    """
    best, best_len = None, 0
    for i in range(sig.shape[1]):
        span = longest_run(sig[:, i])
        if span and span[1] - span[0] > best_len:
            best, best_len = i, span[1] - span[0]
    return best
