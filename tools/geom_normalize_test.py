"""Тест геометрической нормализации: посадить строки реального фото на позиции эталона."""
import cv2, numpy as np
from scipy.signal import find_peaks

# Позиции 4 строк эталонной синтетики (letter 1700x2200), доли высоты.
SYNTH_H, SYNTH_W = 1700, 2200
SYNTH_ROWS = [692, 976, 1258, 1519]


def detect_rows(gray):
    H = gray.shape[0]
    ink = (gray < 110).astype(np.float32).sum(1)
    ink = cv2.blur(ink.reshape(-1, 1), (1, max(9, H // 12))).ravel()
    pk, props = find_peaks(ink, distance=H // 12, height=ink.max() * 0.15)
    order = np.argsort(props["peak_heights"])[::-1]
    return sorted(pk[order][:4].tolist())


def geom_normalize(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    src_rows = detect_rows(g)
    if len(src_rows) != 4:
        return None, src_rows
    # ширину подгоняем под эталон
    scale_x = SYNTH_W / bgr.shape[1]
    resized = cv2.resize(bgr, (SYNTH_W, int(round(bgr.shape[0] * scale_x))), interpolation=cv2.INTER_CUBIC)
    g2 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    src_rows = detect_rows(g2)  # пересчёт после resize
    # Обратное отображение с ЛИНЕЙНОЙ ЭКСТРАПОЛЯЦИЕЙ: выше/ниже контента координаты
    # уходят за пределы исходника -> заполняются белым (border), без смаза сетки.
    from scipy.interpolate import interp1d
    src = np.array(src_rows, dtype=np.float32)
    dst = np.array(SYNTH_ROWS, dtype=np.float32)
    inv = interp1d(dst, src, kind="linear", fill_value="extrapolate")
    y_out = np.arange(SYNTH_H, dtype=np.float32)
    y_src = inv(y_out).astype(np.float32)
    map_y = np.repeat(y_src[:, None], SYNTH_W, axis=1).astype(np.float32)
    map_x = np.repeat(np.arange(SYNTH_W, dtype=np.float32)[None, :], SYNTH_H, axis=0)
    out = cv2.remap(resized, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out, src_rows


if __name__ == "__main__":
    import sys
    bgr = cv2.imread(sys.argv[1])
    out, rows = geom_normalize(bgr)
    print("detected rows:", rows)
    if out is not None:
        cv2.imwrite(sys.argv[2], out)
        print("written", sys.argv[2], out.shape)
