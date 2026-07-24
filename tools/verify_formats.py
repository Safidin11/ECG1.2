"""Round-trip проверка КАЖДОГО формата: sample.png -> layout/vectorize/output.

Прогоняет каждый образец формата с ЯВНО заданным шаблоном (без nnU-Net —
layout/vectorize работают по цветовым чернилам). Кладёт реконструированную
цифровую ЭКГ в папку формата (reconstructed.png) и печатает покрытие.

Запуск:  ./.venv/bin/python tools/verify_formats.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import layout.stage as L          # noqa: E402
import vectorize.stage as V       # noqa: E402
import output.stage as O          # noqa: E402


def run_one(name, sample):
    wd = tempfile.mkdtemp()
    base = {"_run_dir": wd, "_repo_root": str(ROOT), "fs": 500}
    manifest = {"core_ready_image": str(sample), "fs": 500}
    inp = os.path.join(wd, "in.json")
    with open(inp, "w") as f:
        json.dump(manifest, f)

    o1 = L.run(inp, {**base, "_stage_params": {"template": name}})
    o2 = V.run(o1, {**base, "_stage_params": {"clip_mV": 3.0}})
    o3 = O.run(o2, {**base, "_stage_params": {}})
    d = json.load(open(o3))
    cov = d.get("coverage", {})
    avg = sum(cov.values()) / len(cov) if cov else 0
    dst = sample.parent / "reconstructed.png"
    if d.get("digital_ecg") and Path(d["digital_ecg"]).exists():
        shutil.copy2(d["digital_ecg"], dst)
    lay = d.get("layout", {})
    return lay.get("template"), len(lay.get("cells", {})), len(lay.get("rhythm_strips", [])), avg


def main():
    layouts = yaml.safe_load(open(ROOT / "configs" / "lead_layouts.yml"))["layouts"]
    print(f"{'формат':10} {'клеток':>7} {'ритм':>5} {'ср.покрытие':>12}")
    for name in layouts:
        sample = ROOT / "data" / "samples" / "formats" / name / "sample.png"
        if not sample.exists():
            print(f"{name:10}  нет sample.png"); continue
        tpl, ncells, nrhythm, avg = run_one(name, sample)
        ok = "OK" if avg > 0.9 else "!!"
        print(f"{name:10} {ncells:7d} {nrhythm:5d} {avg:11.0%}  {ok}")


if __name__ == "__main__":
    main()
