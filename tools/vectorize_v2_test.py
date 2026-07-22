"""Переработанная векторизация: каждое отведение анализируется НЕЗАВИСИМО.

Ключевые отличия от v1 (которая брала общую полосу-ROI на всю строку):
  * baseline — СВОЙ для каждого отведения (мода гистограммы y его трассы);
  * ROI/окно поиска — СВОЁ для каждого отведения (центр строки ± доля до
    соседней строки), соседи не влияют;
  * трассировка «следованием» — в каждом столбце берём кластер пикселей,
    ближайший к предыдущей точке (устойчиво к смещению соседей, к глубоким
    S-зубцам и толщине штриха), а не среднюю по всему окну;
  * разрывы не останавливают трассировку — интерполируем и идём дальше;
  * подхват из полутонового изображения, где маска дырявая (тёмные пиксели
    рядом с текущей траекторией);
  * снятие дрейфа базовой линии скользящей медианой — чинит длинную ритм-полоску
    и остаточные смещения.
"""
import cv2, numpy as np, json, glob
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

LEAD_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
GRID = [["I","aVR","V1","V4"],["II","aVL","V2","V5"],["III","aVF","V3","V6"]]
FS = 500
DARK = 110   # порог «чернил» в полутоне (после deshadow сетка светлее)


def detect_rows(trace):
    H,W = trace.shape
    prof = cv2.blur(trace.sum(1).astype(np.float32).reshape(-1,1),(1,max(9,H//40))).ravel()
    pk,_ = find_peaks(prof, distance=H//8, height=prof.max()*0.15)
    if len(pk)>4: pk = pk[np.argsort(prof[pk])[::-1][:4]]
    return sorted(int(p) for p in pk)


def mm_per_px(gray):
    H,W = gray.shape
    row = 255 - gray[H//2].astype(np.float32); row -= row.mean()
    ac = np.correlate(row,row,'full')[W-1:]; ac[:3]=0
    pk,_ = find_peaks(ac[:60], height=ac.max()*0.2)
    return float(pk[0]) if len(pk) else 8.0


def lead_baseline(src, x0, x1, lo, hi):
    """Мода y трассы отведения в его окне -> базовая линия (изолиния)."""
    sub = src[lo:hi, x0:x1]
    ys = np.where(sub>0)[0]
    if len(ys)==0: return (lo+hi)//2
    h,_ = np.histogram(ys, bins=np.arange(0, hi-lo+2, 2))
    return lo + int(np.argmax(h)*2)


def clusters(colpix):
    """Группы подряд идущих ненулевых y -> список центроидов."""
    ys = np.where(colpix>0)[0]
    if len(ys)==0: return []
    out=[]; start=ys[0]; prev=ys[0]
    for y in ys[1:]:
        if y-prev>3:
            out.append((start+prev)/2); start=y
        prev=y
    out.append((start+prev)/2)
    return out


def trace_follow(mask, gray, x0, x1, lo, hi, baseline):
    """Трассировка отведения следованием за ближайшим кластером.

    Источник: маска; где маски нет — тёмные пиксели полутона рядом с текущей y.
    """
    n = x1-x0
    ys = np.full(n, np.nan)
    prev = baseline - lo
    winh = hi-lo
    for i in range(n):
        col = mask[lo:hi, x0+i]
        cl = clusters(col)
        if not cl and gray is not None:
            # подхват из полутона: тёмные пиксели рядом с prev (±40px)
            g = gray[lo:hi, x0+i]
            near_lo = max(0,int(prev)-40); near_hi = min(winh,int(prev)+40)
            dark = np.where(g[near_lo:near_hi] < DARK)[0]
            if len(dark): cl = [near_lo + dark.mean()]
        if cl:
            y = min(cl, key=lambda c: abs(c-prev))   # ближайший к траектории
            ys[i] = y; prev = y
    # интерполяция разрывов (трассировка не «останавливается»)
    idx = np.arange(n); good = ~np.isnan(ys)
    if good.sum() < 5: return None, 0.0
    cov = float(good.mean())
    ys = np.interp(idx, idx[good], ys[good]) + lo
    return ys, cov


def to_mv(ys, mm_px, seconds, fs, clip=3.0):
    mV = -(ys - np.median(ys)) / (10.0*mm_px)
    # снятие дрейфа базовой линии (скользящая медиана ~0.6с) — чинит ритм-строку
    win = max(11, int(0.6*fs*len(ys)/(seconds*fs)) | 1)
    win = min(win, (len(ys)//2)*2-1) if len(ys)>22 else 11
    if win>=3 and win<len(ys):
        mV = mV - median_filter(mV, size=win)
    mV = np.clip(mV, -clip, clip)
    target = int(fs*seconds)
    return np.interp(np.linspace(0,1,target), np.linspace(0,1,len(mV)), mV)


def run(core_ready, mask_path, out_png):
    gray = cv2.cvtColor(cv2.imread(core_ready), cv2.COLOR_BGR2GRAY)
    mask = (cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)>0).astype(np.uint8)
    H,W = mask.shape
    rows = detect_rows(mask)
    mmpx = mm_per_px(gray)
    xs = np.where(mask.any(0))[0]; xL,xR = int(xs.min()), int(xs.max())
    cal = int(14*mmpx)
    colw = (xR-xL)/4
    cols = [[int(xL+c*colw), int(xL+(c+1)*colw)] for c in range(4)]
    cols[0][0]+=cal

    # окна: центр строки ± 0.7 * расстояние до соседа (СВОЁ для каждой строки)
    centers = rows[:3]; rhy = rows[3]
    signals={}; covs={}
    for r,center in enumerate(centers):
        up = center - (0 if r==0 else centers[r-1]); up = center if r==0 else (center-centers[r-1])
        dn = (centers[r+1]-center) if r<2 else (rhy-center)
        wlo = max(0, int(center - 0.72*up)); whi = min(H, int(center + 0.72*dn))
        for c,(x0,x1) in enumerate(cols):
            lead = GRID[r][c]
            base = lead_baseline(mask, x0, x1, wlo, whi)
            ys,cov = trace_follow(mask, gray, x0, x1, wlo, whi, base)
            if ys is not None:
                signals[lead] = to_mv(ys, mmpx, 2.5, FS); covs[lead]=round(cov,2)
    # ритм — независимо, окно вниз до края листа
    wlo = max(0,int(rhy-0.72*(rhy-centers[2]))); whi=H
    base = lead_baseline(mask, xL+cal, xR, wlo, whi)
    ys,cov = trace_follow(mask, gray, xL+cal, xR, wlo, whi, base)
    signals["II_rhythm"] = to_mv(ys, mmpx, 10.0, FS); covs["II_rhythm"]=round(cov,2)

    order=[l for l in LEAD_ORDER+["II_rhythm"] if l in signals]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axs=plt.subplots(len(order),1,figsize=(12,1.4*len(order)))
    for ax,l in zip(axs,order):
        ax.plot(np.arange(len(signals[l]))/FS, signals[l], lw=0.8, color='black')
        ax.set_ylabel(f"{l}\ncov={covs[l]:.0%}",rotation=0,labelpad=30,fontsize=9,va='center')
        ax.set_ylim(-2,2.5); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png,dpi=110); plt.close()
    print("written",out_png,"| covs:",covs)


if __name__=="__main__":
    import sys
    run(sys.argv[1], sys.argv[2], sys.argv[3])
