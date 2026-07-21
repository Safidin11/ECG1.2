"""Тонкий драйвер для ecg-image-generator (felixkrones), обходящий баг их CLI.

Их gen_ecg_image_from_data.get_parser() падает на Python 3.11
(`--store_config` объявлен с const без nargs='?'). Мы НЕ правим их код —
собираем argparse.Namespace вручную (все дефолты из их парсера) и вызываем
их же run_single_file(). Генерирует чистую картинку в «родном» формате:
3 ряда × 4 отведения (2.5с) + ритм-строка 10с, 25мм/с, 10мм/мВ.

Запуск (в окружении ecgdig, у него есть tensorflow/wfdb):
    /opt/anaconda3/envs/ecgdig/bin/python tools/gen_native_image.py \
        -i data/samples/native_src/synthetic12.dat \
        -hea data/samples/native_src/synthetic12.hea \
        -o data/samples/native_img
"""
import argparse
import os
import sys

GEN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "ECG-Digitiser", "ecg-image-generator",
)


def build_ns(input_file, header_file, output_directory, seed, start_index,
             resolution, calibration_pulse):
    """Namespace со всеми полями, которые читает run_single_file (дефолты их CLI)."""
    return argparse.Namespace(
        input_file=os.path.abspath(input_file),
        header_file=os.path.abspath(header_file),
        output_directory=os.path.abspath(output_directory),
        seed=seed, start_index=start_index,
        num_leads="twelve", config_file="config.yaml",
        resolution=resolution, pad_inches=0, print_header=False,
        num_columns=-1, full_mode="II",
        mask_unplotted_samples=False, add_qr_code=False,
        link="", num_words=5, x_offset=30, y_offset=30,
        handwriting_size_factor=0.2,
        crease_angle=90, num_creases_vertically=10, num_creases_horizontally=10,
        rotate=0, noise=50, crop=0.01, temperature=40000,
        random_resolution=False, random_padding=False, random_grid_color=False,
        standard_grid_color=5, calibration_pulse=calibration_pulse,
        random_grid_present=1, random_print_header=0, random_bw=0,
        remove_lead_names=True, lead_name_bbox=False, store_config=0,
        deterministic_offset=False, deterministic_num_words=False,
        deterministic_hw_size=False, deterministic_angle=False,
        deterministic_vertical=False, deterministic_horizontal=False,
        deterministic_rot=False, deterministic_noise=False,
        deterministic_crop=False, deterministic_temp=False,
        fully_random=False, hw_text=False, wrinkles=False, augment=False,
        lead_bbox=False, num_images_per_ecg=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", required=True)
    ap.add_argument("-hea", "--header_file", required=True)
    ap.add_argument("-o", "--output_directory", required=True)
    ap.add_argument("-se", "--seed", type=int, default=10)
    ap.add_argument("-st", "--start_index", type=int, default=0)
    ap.add_argument("-r", "--resolution", type=int, default=200)
    ap.add_argument("--calibration_pulse", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(os.path.abspath(args.output_directory), exist_ok=True)
    ns = build_ns(args.input_file, args.header_file, args.output_directory,
                  args.seed, args.start_index, args.resolution, args.calibration_pulse)

    # Их код требует запуска из папки генератора (относительные пути к Fonts/ и т.п.)
    sys.path.insert(0, GEN_DIR)
    os.chdir(GEN_DIR)
    from gen_ecg_image_from_data import run_single_file
    run_single_file(ns)
    print("OK: image(s) written to", ns.output_directory)


if __name__ == "__main__":
    main()
