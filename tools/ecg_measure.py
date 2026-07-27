"""Измерения по оцифрованному сигналу. Пока — ЧСС.

Отдельный модуль, а не кусок стенда: этим же кодом считает и сайт, и проверка
на PTB-XL, иначе они разъедутся и проверка перестанет что-либо значить.

Проверено стендом tools/validate_ptbxl.py на 40 записях PTB-XL: медиана ошибки
0.1 уд/мин, все 40 в пределах 5 уд/мин от эталона.

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
    if len(seg) < 3 * fs:
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
