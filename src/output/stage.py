"""Стадия output: сигнал -> WFDB + итоговый превью-график.

Берёт финальный сигнал (наша векторизация) из манифеста и пишет его в WFDB
(.dat/.hea), а превью-график копирует как итоговый артефакт. NaN (короткие
отведения, не дополненные до 10с) заменяются нулём для записи.

Вход:  манифест vectorize.json (signal_npy, leads, fs).
Выход: output.json + record.dat/.hea + preview.png.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import get_logger, stage_dir  # noqa: E402

STAGE = "output"
log = get_logger(STAGE)


def run(input_path: str, config: dict) -> str:
    out_dir = stage_dir(config, STAGE)
    with open(input_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sig = np.load(manifest["signal_npy"])          # (N, 12), возможны NaN
    leads = manifest.get("leads")
    fs = manifest.get("fs", 500)

    # WFDB не принимает NaN — пишем 0 вместо пропусков.
    p_signal = np.nan_to_num(sig).astype(np.float64)
    record = "ecg_reconstructed"
    wfdb.wrsamp(
        record, fs=fs, units=["mV"] * p_signal.shape[1], sig_name=leads,
        p_signal=p_signal, write_dir=str(out_dir), fmt=["16"] * p_signal.shape[1],
        adc_gain=[1000.0] * p_signal.shape[1], baseline=[0] * p_signal.shape[1],
    )

    # финальное превью
    final_preview = out_dir / "preview.png"
    if manifest.get("preview") and Path(manifest["preview"]).exists():
        shutil.copy2(manifest["preview"], final_preview)

    manifest["wfdb_record"] = str(out_dir / record)
    manifest["final_preview"] = str(final_preview)
    out_path = out_dir / "output.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("STAGE %s: WFDB=%s.dat + превью=%s", STAGE, record, final_preview)
    return str(out_path)
