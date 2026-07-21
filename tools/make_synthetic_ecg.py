"""Генератор простой синтетической картинки ЭКГ-распечатки для смоук-тестов.

ВНИМАНИЕ: это грубая имитация «бумажной» ЭКГ (красная сетка + кривые),
нужная только чтобы прогнать каркас пайплайна. Это НЕ формат, который ждёт
ядро felixkrones — настоящую синтетику в его формате поднимем в Фазе 1.

Запуск:
    /opt/anaconda3/bin/python tools/make_synthetic_ecg.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "data" / "samples" / "synthetic_ecg.png"


def ecg_beat(t: np.ndarray) -> np.ndarray:
    """Очень упрощённый QRS-подобный сигнал (сумма гауссиан P-Q-R-S-T)."""
    def g(c, a, w):
        return a * np.exp(-((t - c) ** 2) / (2 * w ** 2))

    period = 0.8
    phase = t % period
    return (
        g_wave(phase, 0.15, 0.10, 0.025)   # P
        + g_wave(phase, 0.35, -0.12, 0.012)  # Q
        + g_wave(phase, 0.40, 1.00, 0.010)   # R
        + g_wave(phase, 0.45, -0.25, 0.012)  # S
        + g_wave(phase, 0.60, 0.25, 0.035)   # T
    )


def g_wave(x, c, a, w):
    return a * np.exp(-((x - c) ** 2) / (2 * w ** 2))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fs = 500
    duration = 2.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    leads = ["I", "II", "III", "aVR", "aVL", "aVF",
             "V1", "V2", "V3", "V4", "V5", "V6"]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(3, 4, figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor("white")

    for ax, name in zip(axes.ravel(), leads):
        sig = ecg_beat(t) * (0.8 + 0.4 * rng.random()) + 0.01 * rng.standard_normal(t.size)
        # «миллиметровая» сетка
        ax.set_xticks(np.arange(0, duration + 0.04, 0.04), minor=True)
        ax.set_yticks(np.arange(-1.5, 1.6, 0.1), minor=True)
        ax.grid(which="minor", color="#f4b8b8", linewidth=0.4)
        ax.grid(which="major", color="#e57373", linewidth=0.8)
        ax.set_xticks(np.arange(0, duration + 0.2, 0.2))
        ax.set_yticks(np.arange(-1.5, 1.6, 0.5))
        ax.plot(t, sig, color="black", linewidth=0.8)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, duration)
        ax.text(0.02, 1.15, name, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=0, length=0)

    fig.suptitle("SYNTHETIC ECG (demo only — not a medical record)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, facecolor="white")
    plt.close(fig)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
