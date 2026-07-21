"""ECG1.2 — оркестратор пайплайна дигитизации ЭКГ.

Читает configs/pipeline.yml и по очереди вызывает активные стадии.
Каждая стадия — модуль в src/<name>/ с функцией:

    run(input_path: str, config: dict) -> output_path: str

Выход одной стадии становится входом следующей. Неактивные стадии
(enabled: false) пропускаются. Падение одной стадии НЕ роняет весь
пайплайн — вход прокидывается дальше без изменений (мягкая деградация).

Learning/demo-инструмент. НЕ медицинское изделие.

Запуск:
    python src/pipeline.py --input data/samples/<img> [--config configs/pipeline.yml]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import sys
from pathlib import Path

import yaml

# src/ на пути импорта, чтобы стадии импортировались как пакеты (preprocess, ...)
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import get_logger  # noqa: E402

log = get_logger("pipeline")

REPO_ROOT = SRC_DIR.parent


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_run_dir(cfg: dict) -> Path:
    work_dir = REPO_ROOT / cfg.get("project", {}).get("work_dir", "output/runs")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = work_dir / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_pipeline(input_path: str, config_path: str) -> str:
    cfg = load_config(config_path)
    run_dir = make_run_dir(cfg)
    cfg["_run_dir"] = str(run_dir)
    cfg["_repo_root"] = str(REPO_ROOT)

    log.info("=" * 64)
    log.info("ECG1.2 pipeline | вход: %s", input_path)
    log.info("run-каталог: %s", run_dir)
    log.info("=" * 64)

    current = str(Path(input_path).resolve())
    stages = cfg.get("stages", [])

    for i, stage in enumerate(stages, start=1):
        name = stage["name"]
        module_name = stage.get("module", name)

        if not stage.get("enabled", True):
            log.info("STAGE %s: DISABLED — пропуск (вход прокинут дальше)", name)
            continue

        # Параметры конкретной стадии кладём в конфиг под ключ _stage_params.
        stage_cfg = dict(cfg)
        stage_cfg["_stage_params"] = stage.get("params", {}) or {}

        try:
            mod = importlib.import_module(module_name)
            out = mod.run(current, stage_cfg)
            if not out or not Path(out).exists():
                raise RuntimeError(f"стадия {name} вернула несуществующий путь: {out!r}")
            current = out
        except Exception as exc:  # мягкая деградация: логируем и идём дальше
            log.error("STAGE %s: ОШИБКА — %s: %s", name, type(exc).__name__, exc)
            log.error("STAGE %s: деградация — вход прокинут без изменений", name)
            # current остаётся прежним

    log.info("=" * 64)
    log.info("ГОТОВО. Итоговый артефакт: %s", current)
    log.info("=" * 64)
    return current


def main() -> None:
    ap = argparse.ArgumentParser(description="ECG1.2 — оркестратор пайплайна дигитизации ЭКГ")
    ap.add_argument("--input", "-i", required=True, help="путь к входной картинке ЭКГ")
    ap.add_argument(
        "--config",
        "-c",
        default=str(REPO_ROOT / "configs" / "pipeline.yml"),
        help="путь к pipeline.yml",
    )
    args = ap.parse_args()

    if not Path(args.input).exists():
        log.error("Входной файл не найден: %s", args.input)
        sys.exit(1)

    run_pipeline(args.input, args.config)


if __name__ == "__main__":
    main()
