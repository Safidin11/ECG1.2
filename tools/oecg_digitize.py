"""Обёртка над Open-ECG-Digitizer (внешний движок, свой venv, свои веса).

Мы НЕ правим их код: готовим вход (8-бит + разумный размер), пишем временный
конфиг и запускаем их `python -m src.digitize` как отдельный процесс.

Зачем подготовка входа:
  * 16-битные PNG у них падают ("min_all" not implemented for 'UInt16');
  * фото 12 Мп считается ~10 мин против ~12 с для ~2 Мп при том же качестве.

Использование:
    python tools/oecg_digitize.py -i фото.jpg -o папка_результата
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

OECG_DIR = Path(__file__).resolve().parent.parent / "external" / "Open-ECG-Digitizer"
OECG_PY = OECG_DIR / ".venv" / "bin" / "python"
TARGET_W = 2000          # ширина входа
RESAMPLE = 1800          # внутренний размер U-Net: главный рычаг ПАМЯТИ (их дефолт 3000 под GPU)


def prepare_image(src: str, dst: Path, target_w: int = TARGET_W) -> tuple[int, int]:
    """Привести любое фото к тому, что движок точно съест: 8-бит BGR, ~target_w."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"не удалось прочитать картинку: {src}")
    if img.dtype == np.uint16:                       # 16-бит -> 8-бит
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[1] > target_w:                      # ужать крупное фото
        h = int(round(img.shape[0] * target_w / img.shape[1]))
        img = cv2.resize(img, (target_w, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img)
    return img.shape[1], img.shape[0]


CONFIG_TEMPLATE = """MODEL:
  class_path: 'src.model.inference_wrapper.InferenceWrapper'
  KWARGS:
    config:
      SIGNAL_EXTRACTOR:
        class_path: 'src.model.signal_extractor.SignalExtractor'
        KWARGS: {{}}
      PERSPECTIVE_DETECTOR:
        class_path: 'src.model.perspective_detector.PerspectiveDetector'
        KWARGS:
          num_thetas: 250
      DEWARPER:
        class_path: 'src.model.dewarper.Dewarper'
        KWARGS:
          abs_peak_threshold: 0.1
      SEGMENTATION_MODEL:
        class_path: 'src.model.unet.UNet'
        weight_path: './weights/unet_weights_07072025.pt'
        KWARGS:
          num_in_channels: 3
          num_out_channels: 4
          dims: [32, 64, 128, 256, 320, 320, 320, 320]
          depth: 2
      CROPPER:
        class_path: 'src.model.cropper.Cropper'
        KWARGS:
          granularity: 80
          percentiles: [0.02, 0.98]
          alpha: 0.85
      PIXEL_SIZE_FINDER:
        class_path: 'src.model.pixel_size_finder.PixelSizeFinder'
        KWARGS:
          min_number_of_grid_lines: 30
          max_number_of_grid_lines: 70
          lower_grid_line_factor: 0.5
      LAYOUT_IDENTIFIER:
        class_path: 'src.model.lead_identifier.LeadIdentifier'
        config_path: '{layouts}'
        unet_config_path: 'src/config/lead_name_unet.yml'
        unet_weight_path: './weights/lead_name_unet_weights_07072025.pt'
        KWARGS:
          debug: false
          device: '{device}'
          possibly_flipped: false
          target_num_samples: 10000
          required_valid_samples: 2
    device: '{device}'
    resample_size: {resample}
    rotate_on_resample: true
    enable_timing: false
    apply_dewarping: false
DATA:
  images_path: '{in_dir}'
  image_extensions: ['.png', '.jpg', '.jpeg', '.JPG']
  save_mode: 'all'
  layout_should_include_substring: false
  output_path: '{out_dir}'
  clear_output_dir_if_exists: false
"""


def digitize(input_path: str, out_dir: str,
             layouts: str = str(Path(__file__).resolve().parent.parent / "configs" / "oecg_layouts.yml"),
             resample: int = RESAMPLE, target_w: int = TARGET_W,
             threads: int = 4, device: str = "auto") -> Path:
    """Оцифровать одно фото внешним движком. Возвращает папку с результатом.

    resample — внутренний размер для U-Net. Главный рычаг ПАМЯТИ: 3000 (их
    дефолт под GPU) на 8 ГБ ОЗУ уходит в своп; 1800 держится в памяти.
    """
    if not OECG_PY.exists():
        raise RuntimeError(f"нет venv движка: {OECG_PY}\nСм. external/Open-ECG-Digitizer")
    if device == "auto":                     # Apple GPU в 7 раз быстрее и в 4 раза экономнее CPU
        probe = subprocess.run([str(OECG_PY), "-c",
                                "import torch;print('mps' if torch.backends.mps.is_available() else 'cpu')"],
                               capture_output=True, text=True)
        device = probe.stdout.strip() or "cpu"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_dir = tmp / "in"
        in_dir.mkdir()
        name = Path(input_path).stem
        w, h = prepare_image(input_path, in_dir / f"{name}.png", target_w)
        print(f"[oecg] вход {w}x{h}, внутренний размер {resample}, устройство {device}")
        cfg = tmp / "cfg.yml"
        cfg.write_text(CONFIG_TEMPLATE.format(layouts=layouts, in_dir=in_dir,
                                              out_dir=out, resample=resample, device=device))
        env = {**os.environ, "OMP_NUM_THREADS": str(threads),
               "MKL_NUM_THREADS": str(threads)}
        proc = subprocess.run(
            [str(OECG_PY), "-m", "src.digitize", "--config", str(cfg)],
            cwd=str(OECG_DIR), capture_output=True, text=True, env=env,
        )
        for line in proc.stdout.splitlines():
            if "Layout" in line or "layout" in line or "Error" in line:
                print(f"[oecg] {line.strip()}")
        if proc.returncode != 0:
            raise RuntimeError(f"движок упал:\n{proc.stderr[-1500:]}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Оцифровка ЭКГ новым движком (Open-ECG-Digitizer)")
    ap.add_argument("-i", "--input", required=True, help="фото ЭКГ (jpg/png)")
    ap.add_argument("-o", "--out_dir", default=None, help="куда положить результат")
    ap.add_argument("--resample", type=int, default=RESAMPLE,
                    help=f"внутренний размер U-Net (память!), по умолч. {RESAMPLE}")
    ap.add_argument("--width", type=int, default=TARGET_W, help="ширина входа")
    ap.add_argument("--device", default="auto", help="auto (по умолч.) / mps (Apple GPU) / cpu")
    a = ap.parse_args()
    out_dir = a.out_dir or str(Path.cwd() / "output" / "oecg" / Path(a.input).stem)
    res = digitize(a.input, out_dir, resample=a.resample, target_w=a.width, device=a.device)
    print(f"\n[oecg] готово -> {res}")
    for f in sorted(res.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
