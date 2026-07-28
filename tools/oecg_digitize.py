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

# Пороги их SignalExtractor. Их значения по умолчанию (10 / 0.1 / 0.95) заточены
# под чистые сканы: короткие обрывки трассы удаляются целиком, а линия должна
# на 95% лежать внутри маски. На фото трасса рвётся на плоских участках (сеть
# принимает ровную линию за фон), и эти правила добивают то, что она всё-таки
# нашла. Замер на 12x1: средняя потеря сигнала 17.9% -> 12.6%, на чистых
# снимках без регрессий (IMG_4074: 1.3% -> 0.5%).
#
# min_line_width=30 (их умолчание) — отдельная и куда более коварная проблема:
# слабая строка выбрасывается ЦЕЛИКОМ, движок возвращает 11 линий вместо 12, а
# метки отведений присваиваются по порядку — и всё после пропущенной строки
# СДВИГАЕТСЯ. На реальном 12x1 терялась aVF, из-за чего 'V1' на деле был V2,
# 'V5' был V6, а V6 оставался пустым. Порог 10 возвращает все 12 строк.
EXTRACTOR = {"threshold_sum": 1.0, "label_thresh": 0.05, "threshold_line_in_mask": 0.75,
             "min_line_width": 10}


def prepare_image(src: str, dst: Path, target_w: int = TARGET_W) -> tuple[int, int, float]:
    """Привести любое фото к тому, что движок точно съест: 8-бит BGR, ~target_w.

    Третьим значением возвращает КОЭФФИЦИЕНТ РАСТЯЖЕНИЯ (>1, если снимок
    пришлось увеличивать). Он нужен дальше по конвейеру: растяжение делает
    картинку больше, но деталей не добавляет, и без этого числа мелкий снимок
    выглядит полноценным — именно на этом ломался разбор.
    """
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
    # Привести ширину к target_w в ОБЕ стороны. Увеличение мелких снимков —
    # не косметика: на 1000 px линия отведения толщиной в пиксель, и сеть
    # относит слабые/плоские отведения к фону (замерено: aVF/V3/V6 = 0%
    # против 24-25% после увеличения до 2000 px).
    scale = 1.0
    if abs(img.shape[1] - target_w) > 2:
        scale = target_w / img.shape[1]
        h = int(round(img.shape[0] * scale))
        interp = cv2.INTER_CUBIC if img.shape[1] < target_w else cv2.INTER_AREA
        img = cv2.resize(img, (target_w, h), interpolation=interp)
    cv2.imwrite(str(dst), img)
    return img.shape[1], img.shape[0], scale


CONFIG_TEMPLATE = """MODEL:
  class_path: 'src.model.inference_wrapper.InferenceWrapper'
  KWARGS:
    config:
      SIGNAL_EXTRACTOR:
        class_path: 'src.model.signal_extractor.SignalExtractor'
        KWARGS: {extractor_kwargs}
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
    rotate_on_resample: {rotate}
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


LAYOUTS_YML = Path(__file__).resolve().parent.parent / "configs" / "oecg_layouts.yml"


def _yaml_inline(d: dict | None) -> str:
    """dict -> inline-YAML для KWARGS ({} если пусто)."""
    if not d:
        return "{}"
    return "{" + ", ".join(f"{k}: {v}" for k, v in d.items()) + "}"


def layout_names() -> list[str]:
    """Имена доступных раскладок (в порядке из configs/oecg_layouts.yml)."""
    import re
    return re.findall(r"^([A-Za-z_0-9]+):\s*$", LAYOUTS_YML.read_text(encoding="utf-8"), re.M)


def _single_layout_yml(name: str, dst: Path) -> Path:
    """Файл ровно с одной раскладкой — так движок вынужден взять именно её."""
    import re
    src = LAYOUTS_YML.read_text(encoding="utf-8")
    blocks, cur, key = {}, [], None
    for line in src.splitlines():
        m = re.match(r"^([A-Za-z_0-9]+):\s*$", line)
        if m:
            if key:
                blocks[key] = "\n".join(cur).rstrip()
            key, cur = m.group(1), [line]
        elif key:
            cur.append(line)
    if key:
        blocks[key] = "\n".join(cur).rstrip()
    if name not in blocks:
        raise RuntimeError(f"неизвестная раскладка: {name}")
    dst.write_text(blocks[name] + "\n", encoding="utf-8")
    return dst


def digitize(input_path: str, out_dir: str,
             layouts: str | None = None,
             resample: int = RESAMPLE, target_w: int = TARGET_W,
             threads: int = 4, device: str = "auto",
             layout: str | None = None, rotate: bool = False,
             extractor: dict | None = None, sharpen: bool = True) -> Path:
    """Оцифровать одно фото внешним движком. Возвращает папку с результатом.

    layout — задать формат ЖЁСТКО (имя из layout_names()); None = пусть
    определяет сам по всему списку.
    extractor — параметры их SignalExtractor; None = наши EXTRACTOR (мягче их
    умолчаний), {} = их умолчания. Их загрузчик делает SignalExtractor(**KWARGS),
    поэтому пороги задаются из НАШЕГО конфига без правки их кода.
    rotate — их rotate_on_resample: разворачивает КАЖДОЕ фото, где высота
    больше ширины. Для вертикальных ЭКГ (12x1 — 12 строк) это ломает разбор,
    поэтому по умолчанию выключено: доверяем ориентации снимка.
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
    # Абсолютный путь обязателен: обработчик запускается с рабочим каталогом
    # ДВИЖКА, и относительный путь у него разрешится не туда.
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_dir = tmp / "in"
        in_dir.mkdir()
        name = Path(input_path).stem
        w, h, scale = prepare_image(input_path, in_dir / f"{name}.png", target_w)
        lay_path = layouts or str(LAYOUTS_YML)
        if layout:                            # формат задан вручную -> список из одной раскладки
            lay_path = str(_single_layout_yml(layout, tmp / "layout.yml"))
        print(f"[oecg] вход {w}x{h}, внутренний размер {resample}, устройство {device}, "
              f"формат {layout or 'авто'}"
              + (f", снимок растянут в {scale:.2f}x" if scale > 1.02 else ""))
        cfg = tmp / "cfg.yml"
        cfg.write_text(CONFIG_TEMPLATE.format(layouts=lay_path, in_dir=in_dir, out_dir=out,
                                              resample=resample, device=device,
                                              rotate=str(bool(rotate)).lower(),
                                              extractor_kwargs=_yaml_inline(
                                                  EXTRACTOR if extractor is None else extractor)))
        env = {**os.environ, "OMP_NUM_THREADS": str(threads),
               "MKL_NUM_THREADS": str(threads)}
        # Свой обработчик вместо их src.digitize: тот же сигнал, плюс цифровая
        # копия снимка (геометрия один в один). Сеть прогоняется ОДИН раз.
        worker = Path(__file__).resolve().parent / "oecg_worker.py"
        proc = subprocess.run(
            [str(OECG_PY), str(worker), "-c", str(cfg),
             "-i", str(in_dir / f"{name}.png"), "-o", str(out / name),
             "--layouts", str(LAYOUTS_YML), "--input-scale", f"{scale:.6f}",
             ("--sharpen" if sharpen else "--no-sharpen")],
            cwd=str(OECG_DIR), capture_output=True, text=True, env=env,
        )
        for line in proc.stdout.splitlines():
            if any(k in line for k in ("Layout", "layout", "Error", "[twin]")):
                print(f"[oecg] {line.strip()}")
        if proc.returncode != 0:
            raise RuntimeError(f"движок упал:\n{proc.stderr[-1500:]}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Оцифровка ЭКГ новым движком (Open-ECG-Digitizer)")
    ap.add_argument("-i", "--input", help="фото ЭКГ (jpg/png)")
    ap.add_argument("-o", "--out_dir", default=None, help="куда положить результат")
    ap.add_argument("--resample", type=int, default=RESAMPLE,
                    help=f"внутренний размер U-Net (память!), по умолч. {RESAMPLE}")
    ap.add_argument("--width", type=int, default=TARGET_W, help="ширина входа")
    ap.add_argument("--device", default="auto", help="auto (по умолч.) / mps (Apple GPU) / cpu")
    ap.add_argument("--layout", default=None,
                    help="задать формат жёстко (список: --list-layouts)")
    ap.add_argument("--list-layouts", action="store_true", help="показать доступные форматы")
    a = ap.parse_args()
    if a.list_layouts:
        print("\n".join(layout_names())); return
    out_dir = a.out_dir or str(Path.cwd() / "output" / "oecg" / Path(a.input).stem)
    res = digitize(a.input, out_dir, resample=a.resample, target_w=a.width,
                   device=a.device, layout=a.layout)
    print(f"\n[oecg] готово -> {res}")
    for f in sorted(res.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
