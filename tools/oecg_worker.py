"""Обработчик поверх Open-ECG-Digitizer: их пайплайн + цифровой двойник снимка.

Их `src.digitize` сохраняет только сигнал и диагностическую картинку из 4
панелей. Нам нужно ещё одно: ЦИФРОВАЯ КОПИЯ обработанной фотографии — тот же
размер, те же места линий, та же высота зубцов, но линия нарисована по
оцифрованным данным. Такую копию можно наложить на снимок, и всё совпадёт.

Всё берётся из ОДНОГО прогона сети: их `forward()` уже возвращает
  aligned.image   — снимок после исправления перспективы и обрезки,
  signal.raw_lines — линии в ПИКСЕЛЬНЫХ координатах этого снимка,
  pixel_spacing_mm — масштаб.
Поэтому второй прогон (и лишние полторы минуты) не нужен.

Мы НЕ правим их код: импортируем их же классы и повторяем их логику сохранения
CSV дословно.

Запускать интерпретатором движка с cwd = каталог движка.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from src.config.default import get_cfg                          # noqa: E402
from src.utils import find_config_path, import_class_from_path  # noqa: E402
from torchvision.io import decode_image                         # noqa: E402

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Палитра копии — как на бумаге кардиографа GE/Philips. Рисуем через PIL, то
# есть в RGB (в отличие от снимка, который пишется OpenCV в BGR).
PAPER = (255, 253, 253)
GRID_MINOR = (250, 220, 220)     # #F9D8D8 при 90% поверх бумаги, 1 мм
GRID_MAJOR = (240, 170, 170)     # #F0AAAA, 5 мм
TRACE = (17, 17, 17)             # #111111
LABEL = (58, 58, 58)             # #3A3A3A
SEP = (196, 199, 204)            # границы отведений — заметные, но спокойные
SS = 3                           # суперсэмплинг: трасса рисуется крупнее и
                                 # усредняется — отсюда сглаженные края

# Шрифт подписей: сначала системный SF Pro (Semibold), потом Helvetica Neue.
FONTS = (("/System/Library/Fonts/SFNS.ttf", 0, "Semibold"),
         ("/System/Library/Fonts/HelveticaNeue.ttc", 1, None),
         ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0, None),
         ("/System/Library/Fonts/Supplemental/Arial.ttf", 0, None))


def _font(px: int):
    """Медицинский гротеск нужного кегля; вариативному SF задаём начертание."""
    from PIL import ImageFont
    px = max(11, int(round(px)))
    for path, idx, variation in FONTS:
        try:
            f = ImageFont.truetype(path, px, index=idx)
            if variation:
                f.set_variation_by_name(variation)
            return f
        except Exception:
            continue
    return ImageFont.load_default()


def save_csv(canonical, base: str) -> None:
    """Дословно их формат: (отсчёты, отведения), заголовок — имена отведений."""
    if canonical is None:
        return
    data = canonical.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    header = ",".join(LEAD_NAMES[:data.shape[0]])
    np.savetxt(base + "_timeseries_canonical.csv", data.T, delimiter=",",
               header=header, comments="")


def x_offset(arr: np.ndarray, prob: np.ndarray) -> int:
    """На сколько столбцов линии сдвинуты относительно карты вероятностей.

    Их `preprocess_lines` (signal_extractor.py:216-221) обрезает массив линий
    по крайним столбцам, где хоть что-то нашлось: `lines[:, first:last+1]`.
    Поэтому столбец 0 линии — это столбец `first` карты, а не нулевой.
    Само `first` наружу не отдаётся, но его легко найти перебором: при верном
    сдвиге точки линии ложатся на чернила, и сумма вероятностей под ними
    максимальна. Так мы не зависим от их внутренностей.
    """
    w_l, w_p = arr.shape[1], prob.shape[1]
    if w_l >= w_p:
        return 0
    cols = np.flatnonzero(~np.all(np.isnan(arr), axis=0))
    if len(cols) == 0:
        return 0
    sample = cols[:: max(1, len(cols) // 300)]
    best, best_score = 0, -1.0
    for off in range(w_p - w_l + 1):
        score = 0.0
        for k in range(arr.shape[0]):
            v = arr[k, sample]
            m = ~np.isnan(v)
            if not m.any():
                continue
            ys = np.clip(np.round(v[m]).astype(int), 0, prob.shape[0] - 1)
            score += float(prob[ys, sample[m] + off].sum())
        if score > best_score:
            best_score, best = score, off
    return best


def sharpen_apexes(arr: np.ndarray, prob: np.ndarray, thr: float = 0.12,
                   win: int = 2, band: float = 40.0,
                   off: int = 0) -> tuple[np.ndarray, dict]:
    """Вернуть вершины, срезанные усреднением по столбцу.

    Движок берёт координату линии как ЦЕНТР МАСС вероятностей в столбце
    (signal_extractor.py, _extract_line_from_region). На прямом участке это
    несмещённая оценка: чернила лежат симметрично вокруг настоящей кривой.
    А на вершине симметрии нет — выше неё только полутолщина пера, а ниже вся
    восходящая и нисходящая ветви, попавшие в тот же столбец. Центр масс из-за
    этого садится ниже пика, и тем сильнее, чем пик острее.

    Замерено на PTB-XL: амплитуда зубца R занижена на 12.7%, одинаково при любой
    высоте зубца — что и предсказывает эта модель (потеря зависит от ШИРИНЫ
    пика, а не от высоты).

    Лечение: на вершине берём не центр масс, а КРАЙ чернил, и прибавляем
    полутолщину. Полутолщину не задаём, а меряем по самой карте — на ровных
    участках вертикальный размер чернил в столбце равен полной толщине пера
    вместе с ореолом сети.

    Правим ТОЛЬКО вершины: на склонах центр масс верен, и трогать его нельзя,
    иначе вся линия уедет вверх.
    """
    out = arr.copy()
    H, W_prob = prob.shape
    W = arr.shape[1]                       # столбцы считаем в системе ЛИНИЙ
    ink = prob > thr
    stat = {"peaks": 0, "shift_px": []}

    for k in range(arr.shape[0]):
        row = arr[k]
        ok = ~np.isnan(row)
        if ok.sum() < 20:
            continue
        # Полоса вокруг своей линии: иначе в столбце попадутся чернила соседней.
        top = np.full(W, np.nan)
        bot = np.full(W, np.nan)
        for x in np.flatnonzero(ok):
            xp = x + off                       # тот же столбец в системе КАРТЫ
            if not (0 <= xp < W_prob):
                continue
            y0 = int(round(row[x]))
            lo, hi = max(0, y0 - int(band)), min(H, y0 + int(band) + 1)
            col = np.flatnonzero(ink[lo:hi, xp])
            if len(col):
                top[x] = lo + col[0]
                bot[x] = lo + col[-1]

        # полутолщина: по ровным участкам, где центр масс заведомо не смещён
        slope = np.full(W, np.nan)
        idx = np.flatnonzero(ok)
        slope[idx[1:-1]] = (row[idx[2:]] - row[idx[:-2]]) / 2.0
        flat = np.abs(slope) < 0.3
        thick = (bot - top) / 2.0
        half = float(np.nanmedian(thick[flat & ~np.isnan(thick)])) if np.any(
            flat & ~np.isnan(thick)) else 1.0
        if not np.isfinite(half):
            half = 1.0

        # вершины: смена знака наклона
        s = np.where(np.isnan(slope), 0.0, slope)
        for x in idx[1:-1]:
            if np.isnan(row[x]) or np.isnan(top[x]):
                continue
            a, b = s[max(x - 1, 0)], s[min(x + 1, W - 1)]
            lo, hi = max(0, x - win), min(W, x + win + 1)
            if a < 0 and b > 0:                     # вершина вверх (y убывает)
                edge = np.nanmin(top[lo:hi])
                cand = edge + half
                if np.isfinite(cand) and cand < row[x]:
                    stat["shift_px"].append(float(row[x] - cand))
                    out[k, x] = cand
                    stat["peaks"] += 1
            elif a > 0 and b < 0:                   # вершина вниз
                edge = np.nanmax(bot[lo:hi])
                cand = edge - half
                if np.isfinite(cand) and cand > row[x]:
                    stat["shift_px"].append(float(cand - row[x]))
                    out[k, x] = cand
                    stat["peaks"] += 1
    stat["median_shift_px"] = (round(float(np.median(stat["shift_px"])), 2)
                               if stat["shift_px"] else 0.0)
    stat.pop("shift_px")
    return out, stat


def restore_apexes_in_place(wrapper, got: dict, sig: dict) -> dict | None:
    """Вернуть срезанные вершины и пересчитать по ним сигнал.

    Пересчёт обязателен: если поправить только линии, поправленной окажется
    картинка, а в CSV останутся срезанные амплитуды. Привязку к отведениям
    зовём ИХ ЖЕ (`wrapper.identifier`) — это их публичный шаг, их код не тронут.
    """
    lines, prob = sig.get("raw_lines"), got.get("aligned", {}).get("signal_prob")
    if lines is None or prob is None:
        return None
    p = prob.squeeze().cpu().numpy()
    arr = lines.cpu().numpy() if hasattr(lines, "cpu") else np.asarray(lines)
    if arr.ndim == 1:
        arr = arr[None, :]
    if p.ndim != 2 or arr.shape[1] > p.shape[1]:
        print(f"[twin] вершины не правим: линии {arr.shape} против карты {p.shape}")
        return None

    off = x_offset(arr, p)
    fixed, stat = sharpen_apexes(arr, p, off=off)
    stat["x_offset"] = off
    tensor = torch.from_numpy(fixed).float()
    try:
        layout = wrapper.identifier(
            tensor, got["aligned"]["text_prob"],
            got["pixel_spacing_mm"]["average_pixel_per_mm"],
            layout_should_include_substring=None)
    except Exception as exc:
        print(f"[twin] привязка после правки вершин не удалась: {exc}")
        return None

    sig["raw_lines"] = tensor
    if layout.get("canonical_lines") is not None:
        sig["canonical_lines"] = layout["canonical_lines"]
        sig["lines"] = layout.get("lines")
        sig["layout_matching_cost"] = layout.get("cost", 1.0)
        got["layout_name"] = layout.get("layout", got.get("layout_name", ""))
    print(f"[twin] вершин поправлено {stat['peaks']}, "
          f"медианный подъём {stat['median_shift_px']} px")
    return stat


def row_coverage(arr: np.ndarray, mm_y: float) -> list[float]:
    """Доля ширины содержимого, реально прослеженная в каждой СТРОКЕ плёнки.

    Трассировщик отдаёт куски, а не строки: одна строка плёнки может прийти
    двумя-тремя обрывками, и тогда куски надо сначала собрать по вертикальной
    близости. Именно неполная строка — признак той поломки, при которой имена
    отведений съезжают: раздаются они по позициям найденных кусков, поэтому
    один потерянный кусок портит все двенадцать имён разом.
    """
    if arr.size == 0:
        return []
    med = np.array([np.nanmedian(r) if np.any(~np.isnan(r)) else np.nan for r in arr])
    ok = ~np.isnan(med)
    if not ok.any():
        return []
    med, arr = med[ok], arr[ok]
    order = np.argsort(med)
    med, arr = med[order], arr[order]

    # порог «та же строка»: половина типичного расстояния между строками
    gaps = np.diff(med)
    step = float(np.median(gaps)) if len(gaps) else 0.0
    if step <= 0:
        step = 8 * mm_y if mm_y else 40.0
    tol = max(0.4 * step, 3 * mm_y if mm_y else 20.0)

    groups, cur = [], [0]
    for i in range(1, len(med)):
        if med[i] - med[cur[-1]] <= tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    xs = np.flatnonzero(np.any(~np.isnan(arr), axis=0))
    if len(xs) == 0:
        return []
    width = int(xs[-1]) - int(xs[0]) + 1
    out = []
    for g in groups:
        seen = np.any(~np.isnan(arr[g]), axis=0)[int(xs[0]):int(xs[-1]) + 1]
        out.append(float(seen.mean()) if width else 0.0)
    return out


def rhythm_lead(canonical, default: str = "II") -> str:
    """Какое отведение движок положил в ритм-строку.

    В раскладках 3×4+ритм строка помечена как `Any`: конкретное отведение
    определяет сам движок. Оно узнаётся по покрытию — ритм-строка идёт все 10 с,
    остальные клетки только по 2.5 с. Иначе подписи двойника и стандартной
    раскладки расходятся.
    """
    if canonical is None:
        return default
    data = canonical.squeeze().cpu().numpy()
    if data.ndim == 1:
        data = data[None, :]
    cov = [float(np.mean(~np.isnan(row))) for row in data]
    j = int(np.argmax(cov))
    return LEAD_NAMES[j] if cov[j] > 0.6 and j < len(LEAD_NAMES) else default


def lead_grid(layout_name: str, layouts_path: str):
    """Сетка отведений выбранной раскладки: список строк + число колонок."""
    import yaml
    try:
        layouts = yaml.safe_load(open(layouts_path, encoding="utf-8")) or {}
    except Exception:
        return None, 1, 0
    lay = layouts.get(layout_name)
    if not lay:
        return None, 1, 0
    grid = [[r] if isinstance(r, str) else list(r) for r in lay.get("leads", [])]
    cols = int(lay.get("layout", {}).get("cols", 1))
    return grid, cols, len(lay.get("rhythm_leads") or [])


def _sorted_lines(lines) -> np.ndarray:
    """Линии сверху вниз. Экстрактор отдаёт их в произвольном порядке, а подписи
    раскладки идут сверху вниз."""
    arr = lines.cpu().numpy() if hasattr(lines, "cpu") else np.asarray(lines)
    if arr.ndim == 1:
        arr = arr[None, :]
    order = np.argsort([np.nanmedian(r) if np.any(~np.isnan(r)) else np.inf for r in arr])
    return arr[order]


def content_box(arr: np.ndarray, shape, mm_px_x: float, mm_px_y: float):
    """Прямоугольник, который реально занят трассами, плюс поля под оформление.

    Нужен потому, что после исправления перспективы по краям холста остаются
    пустые чёрные поля, и картинка выглядит смещённой. Обрезаем и снимок, и
    копию ОДИНАКОВО — взаимная геометрия не меняется, наложение по-прежнему
    совпадает, просто уходит перекошенная пустота.
    """
    H, W = shape
    xs = np.flatnonzero(np.any(~np.isnan(arr), axis=0))
    ys = arr[~np.isnan(arr)]
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, W, H
    # Поля считаем в СВОИХ миллиметрах по каждой оси: масштаб по x и y разный,
    # и калибр-импульс шириной 9 мм не влезал, когда поле мерили по вертикали.
    mmx = (1.0 / mm_px_x) if mm_px_x > 0 else W / 250.0
    mm = (1.0 / mm_px_y) if mm_px_y > 0 else H / 60.0
    pad_left = int(round(16 * mmx))                       # место под калибр-импульс
    pad_right = int(round(8 * mmx))
    pad_top = int(round(10 * mm))                         # место под подписи
    pad_bottom = int(round(8 * mm))
    # Границы НЕ ограничиваем размером снимка: если содержимое начинается у
    # самого края, поле под калибр-импульс просто добирается пустым — иначе
    # импульс некуда рисовать.
    return (int(xs[0]) - pad_left, int(ys.min()) - pad_top,
            int(xs[-1]) + pad_right, int(ys.max()) + pad_bottom)


def draw_twin(arr: np.ndarray, box, shape, mm_px_x: float, mm_px_y: float,
              out_path: str, lead_rows=None, rhythm_name: str = "II") -> None:
    """Нарисовать цифровую копию: трассы ровно на своих координатах + оформление.

    Координаты трасс НЕ меняются — только вычитается смещение обрезки, которое
    точно так же применяется к снимку. Всё остальное (сетка, подписи, границы
    колонок, калибр-импульс) — оформление поверх.

    Трасса и импульс рисуются на маске в SS раз крупнее и усредняются обратно:
    так получаются сглаженные края и скруглённые стыки без «лесенки». Сетка и
    текст рисуются сразу в конечном размере — им резкость нужнее сглаживания.
    """
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    mm_x = (1.0 / mm_px_x) if mm_px_x > 0 else 0.0        # 1 мм по горизонтали, px
    mm_y = (1.0 / mm_px_y) if mm_px_y > 0 else 0.0

    # Толщины заданы для «эталонных» 8 px/мм и масштабируются под разрешение.
    dens = ((mm_x + mm_y) / 2.0 / 8.0) if (mm_x and mm_y) else 1.0
    k_res = max(1.0, dens)
    w_minor = max(1, int(round(0.5 * k_res)))
    w_major = max(1, int(round(1.0 * k_res)))
    w_trace = max(2, int(round(2.0 * k_res)))

    base = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(base)

    # --- миллиметровка: сначала мелкая, поверх — жирная (5 мм) ---
    if 2 <= mm_x <= 60 and 2 <= mm_y <= 60:
        for major in (False, True):
            color = GRID_MAJOR if major else GRID_MINOR
            wid = w_major if major else w_minor
            for k in range(int(W / mm_x) + 1):
                if (k % 5 == 0) != major:
                    continue
                x = int(round(k * mm_x))
                d.line([(x, 0), (x, H)], fill=color, width=wid)
            for k in range(int(H / mm_y) + 1):
                if (k % 5 == 0) != major:
                    continue
                y = int(round(k * mm_y))
                d.line([(0, y), (W, y)], fill=color, width=wid)

    # --- где начинается и кончается содержимое (для колонок и импульса) ---
    xs = np.flatnonzero(np.any(~np.isnan(arr), axis=0))
    cx0, cx1 = (int(xs[0]) - x0, int(xs[-1]) - x0) if len(xs) else (0, W)

    ncols = max((len(r) for r in lead_rows), default=1) if lead_rows else 1

    def runs_of(row):
        """Непрерывные куски линии: на плёнке между отведениями есть разрыв,
        поэтому куски и есть колонки. Мелкие обрывки от сбоев распознавания
        отбрасываем по длине."""
        ok = ~np.isnan(row)
        if not ok.any():
            return []
        edge = np.diff(ok.astype(np.int8))       # не `d`: снаружи это холст PIL
        starts = list(np.flatnonzero(edge == 1) + 1)
        ends = list(np.flatnonzero(edge == -1) + 1)
        if ok[0]:
            starts = [0] + starts
        if ok[-1]:
            ends = ends + [len(row)]
        span = max(1, len(row) // (ncols * 6))
        return [(a, b) for a, b in zip(starts, ends) if b - a >= span]

    col_w = (cx1 - cx0) / max(ncols, 1)

    # --- границы между отведениями ---
    # Ищем по разрывам линий: на плёнке между отведениями всегда есть промежуток.
    # Разрывы бывают и внутри отведения (сеть потеряла кусок), поэтому берём не
    # все подряд, а те, что легли рядом с ожидаемой границей колонки, и по ним
    # считаем медиану — случайные обрывы её не сдвигают. Если рядом ничего нет,
    # граница остаётся ровно по расчёту.
    seps = []
    if ncols > 1:
        cand = []
        for r in arr:
            rr = runs_of(r)
            cand += [(rr[k - 1][1] + rr[k][0]) / 2.0 - x0 for k in range(1, len(rr))]
        for c in range(1, ncols):
            want = cx0 + c * col_w
            near = [m for m in cand if abs(m - want) < 0.35 * col_w]
            seps.append(int(round(np.median(near) if near else want)))
        top, bot = int(0.015 * H), H - int(0.015 * H)
        # Не тоньше 2 px: на странице картинка ужимается по ширине, и волосяная
        # линия при этом просто исчезает.
        for x in seps:
            d.line([(x, top), (x, bot)], fill=SEP, width=max(2, int(round(1.5 * k_res))))

    # --- трасса и калибр-импульс: маска в SS раз крупнее -> усреднение ---
    ink = Image.new("L", (W * SS, H * SS), 0)
    di = ImageDraw.Draw(ink)
    pen = max(2, int(round(w_trace * SS)))

    def stroke(points, closed_caps=True):
        """Полилиния со скруглёнными стыками и торцами (в координатах маски)."""
        if len(points) < 2:
            return
        di.line(points, fill=255, width=pen, joint="curve")
        if closed_caps:
            r = pen / 2.0
            for px, py in (points[0], points[-1]):
                di.ellipse([px - r, py - r, px + r, py + r], fill=255)

    for row in arr:
        pts, run = [], []
        for x in range(len(row)):
            xc = x - x0
            if not (0 <= xc < W):
                continue
            y = row[x]
            if np.isnan(y):
                if len(run) > 1:
                    pts.append(run)
                run = []
            else:
                run.append(((xc + 0.5) * SS, (float(y) - y0 + 0.5) * SS))
        if len(run) > 1:
            pts.append(run)
        for seg in pts:                                    # разрывы не соединяем
            stroke(seg)

    # --- калибр-импульс: 10 мм (1 мВ) в высоту, 5 мм в ширину, углы 90° ---
    if mm_x > 0 and mm_y > 0:
        for row in arr:
            good = row[~np.isnan(row)]
            if len(good) < 20:
                continue
            b = (float(np.median(good)) - y0) * SS
            top = b - 10 * mm_y * SS                       # 1 мВ при 10 мм/мВ
            xb = (cx0 - 3 * mm_x) * SS
            xa = xb - 5 * mm_x * SS
            if xa < pen or not (0 < top < H * SS):
                continue
            stroke([(xa - 2 * mm_x * SS, b), (xa, b), (xa, top),
                    (xb, top), (xb, b), (xb + 2 * mm_x * SS, b)])

    base.paste(Image.new("RGB", (W, H), TRACE), (0, 0), ink.reduce(SS))

    # --- подписи отведений над началом своего отрезка ---
    if lead_rows:
        font = _font(3.6 * mm_y if mm_y else H / 40)
        gap = mm_y if mm_y else 0.01 * H
        for i, row in enumerate(arr):
            names = lead_rows[i] if i < len(lead_rows) else [rhythm_name]
            n = len(names)
            # Подписи делим ТЕМИ ЖЕ границами, что нарисованы, иначе имя и линия
            # разъезжаются. Ритм-строка (n = 1) занимает всю ширину.
            edges = [cx0] + seps + [cx1] if len(seps) == n - 1 else \
                [int(round(cx0 + k * (cx1 - cx0) / n)) for k in range(n + 1)]
            bounds = [(edges[k], edges[k + 1]) for k in range(n)]
            base_row = np.nanmedian(row)           # общая базовая линия строки
            for c, nm in enumerate(names):
                xs0, xs1 = bounds[c]
                a_, b_ = max(xs0 + x0, 0), min(xs1 + x0, len(row))
                seg = row[a_:b_]
                ok = np.flatnonzero(~np.isnan(seg))
                if len(ok) < 5:
                    continue
                # Подпись — у НАЧАЛА самого отведения, а не у границы колонки:
                # край распознаётся не всегда, и подписи разъезжались.
                tx = a_ + int(ok[0]) - x0 + max(3, int(round(1.5 * mm_x)))
                wt = d.textlength(nm, font=font)
                # Подпись не должна залезать на трассу: берём верх сигнала в её
                # же полосе по x и поднимаемся ещё на миллиметр.
                near = row[max(tx + x0, 0):min(int(tx + wt + 2 * mm_x) + x0, len(row))]
                near = near[~np.isnan(near)]
                y = base_row - y0 - 6 * mm_y
                if len(near):
                    y = min(y, float(near.min()) - y0 - 1.5 * gap)
                y = max(font.size * 1.05, min(H - 2.0, y))
                d.text((tx, y), nm, font=font, fill=LABEL, anchor="ls")

    base.save(out_path)


def crop_pad(img: np.ndarray, box, fill=(250, 250, 250)) -> np.ndarray:
    """Вырезать прямоугольник, который может выходить за границы картинки:
    недостающие поля добираются заливкой. Так снимок и копия остаются одного
    размера и совмещаются."""
    x0, y0, x1, y1 = box
    out = np.full((y1 - y0, x1 - x0, 3), fill, np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def aligned_bgr(img_tensor) -> np.ndarray:
    """Обработанный снимок (после исправления перспективы) в BGR."""
    import cv2
    arr = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    if arr.max() <= 1.001:
        arr = arr * 255.0
    return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-i", "--image", required=True)
    ap.add_argument("-o", "--out_base", required=True, help="префикс выходных файлов")
    ap.add_argument("--layouts", default=None, help="наш configs/oecg_layouts.yml для подписей")
    ap.add_argument("--sharpen", action=argparse.BooleanOptionalAction, default=True,
                    help="возвращать срезанные вершины по карте вероятностей")
    ap.add_argument("--input-scale", dest="input_scale", type=float, default=1.0,
                    help="во сколько раз конвейер растянул исходный снимок "
                         "(нужно, чтобы честно посчитать детализацию оригинала)")
    a = ap.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(find_config_path(a.config))
    wrapper = import_class_from_path(cfg.MODEL.class_path)(**cfg.MODEL.KWARGS)

    image = decode_image(a.image, mode="RGB").unsqueeze(0)
    got = wrapper(image, layout_should_include_substring=None)

    sig = got.get("signal", {})
    # Вершины возвращаем ДО сохранения CSV: иначе в сигнал уйдут срезанные
    # амплитуды, а поправлен окажется только рисунок.
    apex_stat = None
    if a.sharpen:
        apex_stat = restore_apexes_in_place(wrapper, got, sig)
    save_csv(sig.get("canonical_lines"), a.out_base)

    # метаданные раскладки — в их же формате
    name = os.path.basename(a.out_base)
    with open(os.path.join(os.path.dirname(a.out_base), "digitization_metadata.csv"),
              "w", encoding="utf-8") as f:
        f.write("file_path,matching_cost,is_flipped,lead_layout\n")
        f.write(f'{name},{sig.get("layout_matching_cost", 1.0)},'
                f'{sig.get("layout_is_flipped", "False")},{got.get("layout_name", "Unknown layout")}\n')

    photo = aligned_bgr(got["aligned"]["image"])
    H, W = photo.shape[:2]

    # Их же диагностическая картинка из 4 панелей — «все этапы движка».
    try:
        from src.digitize import save_png_plot
        save_png_plot(got, sig.get("canonical_lines"), a.out_base + "_stages")
    except Exception as exc:
        print(f"[twin] панель этапов не построена: {exc}")

    # ВАЖНО: берём raw_lines — это выход экстрактора в ПИКСЕЛЬНЫХ координатах
    # обработанного снимка. Поле lines после привязки к отведениям уже
    # пересчитано в другие единицы и для геометрии не годится.
    import cv2
    lines = sig.get("raw_lines")
    if lines is not None:
        ps = got.get("pixel_spacing_mm", {})
        mmx, mmy = float(ps.get("x", 0) or 0), float(ps.get("y", 0) or 0)
        rows, _, _ = lead_grid(got.get("layout_name", ""), a.layouts) if a.layouts \
            else (None, 1, 0)
        arr = _sorted_lines(lines)
        # Сырой выход трассировщика — для разбора поломок: по нему видно, какие
        # куски он вообще нашёл, до всякой привязки к отведениям.
        np.save(a.out_base + "_lines.npy", arr)
        box = content_box(arr, (H, W), mmx, mmy)
        draw_twin(arr, box, (H, W), mmx, mmy, a.out_base + "_twin.png",
                  lead_rows=rows, rhythm_name=rhythm_lead(sig.get("canonical_lines")))
        cv2.imwrite(a.out_base + "_aligned.png", crop_pad(photo, box))
        print(f"[twin] копия {box[2]-box[0]}x{box[3]-box[1]}, линий={len(arr)}, "
              f"подписи={'да' if rows else 'нет'}")
        # Масштаб определяет амплитуды в мВ: ошибка здесь линейно уходит в
        # милливольты. Печатаем, чтобы её можно было сверить со стендом.
        print(f"[twin] масштаб сетки: {1/mmx:.3f} px/мм по x, {1/mmy:.3f} px/мм по y"
              if mmx > 0 and mmy > 0 else "[twin] масштаб сетки не определён")

        # Отчёт о качестве разбора — по нему сайт решает, можно ли доверять
        # результату. Масштаб пересчитываем на ИСХОДНЫЙ снимок: конвейер мелкие
        # фото растягивает, и на растянутой картинке px/мм выглядят приличными,
        # хотя деталей столько же, сколько было.
        # ДЕЛИМ на коэффициент: если снимок растянули в 1.56 раза, то на каждый
        # миллиметр приходится в 1.56 раза меньше НАСТОЯЩИХ пикселей, чем
        # намерено на обработанной картинке.
        detail_x = (1 / mmx) / a.input_scale if mmx > 0 else 0.0
        detail_y = (1 / mmy) / a.input_scale if mmy > 0 else 0.0
        cov = row_coverage(arr, mmy and 1 / mmy)

        # Калибровочный импульс — независимая проверка масштаба. Движок берёт
        # масштаб из миллиметровки; если её на плёнке нет, число всё равно
        # выдаётся, и оно вымышленное. Импульс по определению 1 мВ = 10 мм, и
        # когда он расходится с сеткой — верить надо ему.
        pulse = None
        try:
            import calibration as cal
            g = cv2.cvtColor(crop_pad(photo, box), cv2.COLOR_BGR2GRAY)
            found = cal.find_pulse(g)
            if found:
                pulse = {"px": found["px"], "rows": found["n_rows"],
                         "spread_px": found["spread_px"]}
                sc = cal.scale_from_pulse(found["px"], 1 / mmy if mmy > 0 else 0)
                if sc:
                    pulse |= {"px_per_mm": round(sc["px_per_mm"], 2),
                              "ratio": round(sc.get("ratio") or 0, 3),
                              "agrees": bool(sc.get("agrees"))}
                print(f"[twin] калибр-импульс {found['px']:.0f} px по "
                      f"{found['n_rows']} строкам -> {found['px'] / 10:.2f} px/мм"
                      + ("" if pulse.get("agrees", True)
                         else f"  РАСХОДИТСЯ С СЕТКОЙ в {pulse['ratio']:.2f} раза"))
        except Exception as exc:
            print(f"[twin] импульс не прочитан: {exc}")
        json.dump({"px_per_mm_processed": [round(1 / mmx, 2) if mmx else 0,
                                           round(1 / mmy, 2) if mmy else 0],
                   "px_per_mm_source": [round(detail_x, 2), round(detail_y, 2)],
                   "input_scale": round(a.input_scale, 3),
                   "n_lines": int(len(arr)),
                   "row_coverage": [round(c, 3) for c in cov],
                   "layout": got.get("layout_name", ""),
                   "layout_cost": float(sig.get("layout_matching_cost", 1.0)),
                   "pulse": pulse},
                  open(a.out_base + "_quality.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[twin] детализация исходника {detail_y:.1f} px/мм, "
              f"покрытие строк {[round(c, 2) for c in cov]}")
    else:
        cv2.imwrite(a.out_base + "_aligned.png", photo)
        print("[twin] линий нет — копия не построена")


if __name__ == "__main__":
    main()
