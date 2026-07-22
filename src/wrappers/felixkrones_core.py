"""Обёртка над ядром felixkrones/ECG-Digitiser как ИЗОЛИРОВАННЫМ подпроцессом.

Источник: https://github.com/felixkrones/ECG-Digitiser  (PhysioNet Challenge 2024)

Его скрипт `python -m src.run.digitize -d <img_dir> -m <model> -o <out>` —
монолит: он делает поворот (Hough), сегментацию (nnU-Net), векторизацию и
запись сигнала в WFDB + превью-график (с --show_image). Мы НЕ импортируем его
код в наш процесс (конфликт зависимостей: нужен python3.11 + torch + nnU-Net),
а вызываем его CLI в его собственном окружении (ecgdig) и читаем результат из
файлов (WFDB) — ровно как требует архитектура.

Возвращает dict-манифест с путями к WFDB, сигналу (.npy), превью и метаданными.
Результат кэшируется по (путь+mtime+модель), чтобы повторные прогоны были
мгновенными (nnU-Net на CPU ~8 мин на картинку).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger  # noqa: E402

log = get_logger("felixkrones")

SOURCE_REPO = "https://github.com/felixkrones/ECG-Digitiser"


def _cache_key(image_path: Path, model_path: str) -> str:
    # Content-addressed: одинаковое содержимое картинки -> один ключ,
    # независимо от пути (passthrough копирует байты между стадиями).
    h = hashlib.sha1()
    h.update(image_path.read_bytes())
    h.update(f"|{Path(model_path).name}".encode())
    return h.hexdigest()[:16]


def _run_digitize(interpreter: str, repo_dir: Path, model_abs: Path,
                  input_dir: Path, out_dir: Path, log_file: Path,
                  driver: Path) -> None:
    """Запустить ядро как подпроцесс в окружении ecgdig, cwd=repo_dir.

    Используем наш безопасный драйвер (tools/felixkrones_safe_digitize.py),
    который переиспользует функции ядра, но не обнуляет картинку при NaN-угле
    Hough (частая беда на реальных сканах). Их код при этом не изменён.
    """
    cmd = [
        interpreter, str(driver),
        "-d", str(input_dir),
        "-m", str(model_abs),
        "-o", str(out_dir),
    ]
    env = dict(os.environ)
    # nnUNetv2_predict должен быть на PATH (лежит рядом с интерпретатором ecgdig)
    env["PATH"] = str(Path(interpreter).parent) + os.pathsep + env.get("PATH", "")
    log.info("felixkrones subprocess: %s (cwd=%s)", " ".join(cmd), repo_dir)
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(repo_dir), env=env,
                              stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "\n".join(Path(log_file).read_text().splitlines()[-15:])
        raise RuntimeError(f"digitize.py вернул код {proc.returncode}. Хвост лога:\n{tail}")


def run_core(image_path: str, out_stage_dir: str, config: dict) -> dict:
    """Прогнать ядро felixkrones на одной картинке. -> манифест (dict).

    Читает параметры из config['_stage_params']:
      repo_dir, model_path (относительно repo_root), reuse_cache(bool)
    и интерпретатор из config['interpreters']['segment_calibrate'].
    """
    repo_root = Path(config["_repo_root"])
    params = config.get("_stage_params", {})
    interpreter = config["interpreters"]["segment_calibrate"]
    repo_dir = (repo_root / params.get("repo_dir", "external/ECG-Digitiser")).resolve()
    model_abs = (repo_root / params.get("model_path", "external/ECG-Digitiser/models/M3")).resolve()
    reuse_cache = bool(params.get("reuse_cache", True))

    image_path = Path(image_path).resolve()
    out_stage_dir = Path(out_stage_dir)
    out_stage_dir.mkdir(parents=True, exist_ok=True)
    record = image_path.stem

    # --- Кэш ---
    key = _cache_key(image_path, str(model_abs))
    cache_dir = repo_root / "output" / "cache" / key
    core_out = cache_dir / "core_out"

    need_run = True
    if reuse_cache and (core_out / f"{record}.hea").exists():
        log.info("felixkrones: cache HIT (%s) — пропускаю nnU-Net", key)
        need_run = False

    if need_run:
        input_dir = cache_dir / "input"
        shutil.rmtree(input_dir, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        core_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, input_dir / image_path.name)
        driver = repo_root / "tools" / "felixkrones_safe_digitize.py"
        _run_digitize(interpreter, repo_dir, model_abs, input_dir, core_out,
                      cache_dir / "digitize.log", driver)

    # --- Чтение результата (WFDB) ---
    rec = wfdb.rdrecord(str(core_out / record))
    signal = rec.p_signal.astype(np.float32)          # (N, num_leads)
    leads = list(rec.sig_name)
    fs = int(rec.fs)

    # Сохраняем артефакты в папку стадии
    signal_npy = out_stage_dir / "signal.npy"
    np.save(signal_npy, signal)
    preview_src = core_out / f"{record}.png"
    preview_dst = out_stage_dir / "preview.png"
    if preview_src.exists():
        shutil.copy2(preview_src, preview_dst)
    # сырая маска nnU-Net (метки 0..12) — для нашей раскладки/векторизации
    mask_src = core_out / f"{record}_mask.png"
    mask_dst = out_stage_dir / "mask.png"
    has_mask = mask_src.exists()
    if has_mask:
        shutil.copy2(mask_src, mask_dst)
    # копия WFDB рядом
    for ext in (".hea", ".dat"):
        src = core_out / f"{record}{ext}"
        if src.exists():
            shutil.copy2(src, out_stage_dir / f"{record}{ext}")

    manifest = {
        "engine": "felixkrones/ECG-Digitiser",
        "source_repo": SOURCE_REPO,
        "image": str(image_path),
        "wfdb_record": str(out_stage_dir / record),
        "signal_npy": str(signal_npy),
        "preview": str(preview_dst) if preview_src.exists() else None,
        "mask_png": str(mask_dst) if has_mask else None,
        "core_ready_image": str(image_path),
        "leads": leads,
        "fs": fs,
        "n_samples": int(signal.shape[0]),
        "n_leads": int(signal.shape[1]),
        "cache_key": key,
    }
    log.info("felixkrones: сигнал %s, отведения=%s, fs=%d",
             signal.shape, leads, fs)
    return manifest
