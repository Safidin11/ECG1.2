"""Сгенерировать по одному чистому образцу ЭКГ на КАЖДЫЙ формат раскладки.

Синтезирует 12-канальный сигнал (10с, 500Гц) и рисует его нашим рендером в
каждой раскладке из configs/lead_layouts.yml. Кладёт в data/samples/formats/
<имя_формата>/sample.png + README.md. Так у каждого формата — своя папка со
всем необходимым для проверки пайплайна.

Запуск:  ./.venv/bin/python tools/make_format_samples.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from render_digital_ecg import render, LEAD_ORDER  # noqa: E402

FS = 500
DUR = 10.0


def _g(x, c, a, w):
    return a * np.exp(-((x - c) ** 2) / (2 * w ** 2))


def _beat(phase):
    return (_g(phase, 0.18, 0.12, 0.022) + _g(phase, 0.30, -0.10, 0.010)
            + _g(phase, 0.33, 1.10, 0.009) + _g(phase, 0.37, -0.28, 0.010)
            + _g(phase, 0.55, 0.32, 0.030))


def synth_signal(seed=7):
    rng = np.random.default_rng(seed)
    n = int(FS * DUR)
    t = np.arange(n) / FS
    phase = (t % 0.8) / 0.8
    base = _beat(phase)
    gains = {"I": 0.9, "II": 1.0, "III": 0.5, "aVR": -0.8, "aVL": 0.4, "aVF": 0.7,
             "V1": -0.6, "V2": 1.2, "V3": 1.5, "V4": 1.6, "V5": 1.1, "V6": 0.8}
    mat = np.zeros((n, 12), dtype=np.float32)
    for j, lead in enumerate(LEAD_ORDER):
        mat[:, j] = gains[lead] * base + 0.008 * rng.standard_normal(n)
    return mat


def main():
    with open(ROOT / "configs" / "lead_layouts.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    layouts = cfg["layouts"]
    mat = synth_signal()
    base_dir = ROOT / "data" / "samples" / "formats"

    for name, tpl in layouts.items():
        d = base_dir / name
        d.mkdir(parents=True, exist_ok=True)
        out = d / "sample.png"
        render(mat, str(out), fs=FS, grid=tpl["grid"], cols=tpl["cols"],
               title=f"ECG1.2 sample — {name}")
        with open(d / "README.md", "w", encoding="utf-8") as f:
            rows = len(tpl["grid"])
            f.write(f"# Формат {name}\n\n{tpl['description']}\n\n"
                    f"- строк: {rows}, колонок: {tpl['cols']}\n"
                    f"- `sample.png` — синтетический пример этой раскладки\n\n"
                    f"Проверка:\n```\n./run_ecg.sh data/samples/formats/{name}/sample.png\n```\n"
                    f"(или задать формат явно: template: \"{name}\" в configs/pipeline.yml)\n")
        print(f"  {name}: {out}")


if __name__ == "__main__":
    main()
