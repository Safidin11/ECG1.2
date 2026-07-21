"""Синтетическая 12-канальная WFDB-запись ЭКГ (10с @ 500Гц) для смоук-теста ядра.

Нужна как ВХОД для ecg-image-generator (felixkrones), который из неё рисует
картинку в «родном» формате (3 ряда × 4 отведения 2.5с + ритм-строка 10с),
именно под такой формат обучены веса сегментации.

Запуск (в окружении с wfdb, напр. ecgdig):
    /opt/anaconda3/envs/ecgdig/bin/python tools/make_native_wfdb.py -o data/samples/native_src
"""
import argparse
import os

import numpy as np
import wfdb

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
DURATION = 10.0


def _g(x, c, a, w):
    return a * np.exp(-((x - c) ** 2) / (2 * w ** 2))


def one_beat(phase):
    """Упрощённый P-QRS-T комплекс (в мВ) внутри одного сердечного цикла."""
    return (
        _g(phase, 0.18, 0.12, 0.022)   # P
        + _g(phase, 0.30, -0.10, 0.010)  # Q
        + _g(phase, 0.33, 1.10, 0.009)   # R
        + _g(phase, 0.37, -0.28, 0.010)  # S
        + _g(phase, 0.55, 0.32, 0.030)   # T
    )


def make_signals(seed=7):
    rng = np.random.default_rng(seed)
    n = int(FS * DURATION)
    t = np.arange(n) / FS
    rr = 0.8  # ~75 уд/мин
    phase = (t % rr) / rr
    base = one_beat(phase)

    sig = np.zeros((n, len(LEADS)), dtype=np.float64)
    # Разные амплитуды/полярности по отведениям, чтобы они отличались.
    lead_gain = {
        "I": 0.9, "II": 1.0, "III": 0.5, "aVR": -0.8, "aVL": 0.4, "aVF": 0.7,
        "V1": -0.6, "V2": 1.2, "V3": 1.5, "V4": 1.6, "V5": 1.1, "V6": 0.8,
    }
    for j, lead in enumerate(LEADS):
        g = lead_gain[lead]
        wander = 0.03 * np.sin(2 * np.pi * 0.15 * t + rng.uniform(0, 6))
        noise = 0.008 * rng.standard_normal(n)
        sig[:, j] = g * base + wander + noise
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out_dir", default="data/samples/native_src")
    ap.add_argument("-n", "--name", default="synthetic12")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sig = make_signals(args.seed)
    wfdb.wrsamp(
        args.name,
        fs=FS,
        units=["mV"] * len(LEADS),
        sig_name=LEADS,
        p_signal=sig,
        fmt=["16"] * len(LEADS),
        adc_gain=[1000.0] * len(LEADS),
        baseline=[0] * len(LEADS),
        write_dir=args.out_dir,
    )
    print(f"written WFDB: {os.path.join(args.out_dir, args.name)}.dat/.hea  shape={sig.shape}")


if __name__ == "__main__":
    main()
