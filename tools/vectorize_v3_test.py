"""v3: трасса из ЦВЕТОВЫХ чернил (сетка убрана) + ограничение скачка."""
import cv2, numpy as np, json, glob
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

LEAD_ORDER=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
GRID=[["I","aVR","V1","V4"],["II","aVL","V2","V5"],["III","aVF","V3","V6"]]
FS=500; INK_T=130; SLEW=70; SEP_TRIM=50


def color_ink(bgr,T=INK_T):
    B,G,R=cv2.split(bgr)
    return ((B<T)&(G<T)&(R<T)).astype(np.uint8)


def detect_rows(mask):
    H,W=mask.shape
    prof=cv2.blur(mask.sum(1).astype(np.float32).reshape(-1,1),(1,max(9,H//40))).ravel()
    pk,_=find_peaks(prof,distance=H//8,height=prof.max()*0.15)
    if len(pk)>4: pk=pk[np.argsort(prof[pk])[::-1][:4]]
    return sorted(int(p) for p in pk)


def mmpx(gray):
    H,W=gray.shape; row=255-gray[H//2].astype(np.float32); row-=row.mean()
    ac=np.correlate(row,row,'full')[W-1:]; ac[:3]=0
    pk,_=find_peaks(ac[:60],height=ac.max()*0.2); return float(pk[0]) if len(pk) else 8.0


def baseline(ink,x0,x1,lo,hi):
    ys=np.where(ink[lo:hi,x0:x1]>0)[0]
    if len(ys)==0: return (lo+hi)//2
    h,_=np.histogram(ys,bins=np.arange(0,hi-lo+2,2)); return lo+int(np.argmax(h)*2)


def clusters(col):
    ys=np.where(col>0)[0]
    if len(ys)==0: return []
    out=[]; s=ys[0]; p=ys[0]
    for y in ys[1:]:
        if y-p>3: out.append(((s+p)/2, p-s+1)); s=y
        p=y
    out.append(((s+p)/2, p-s+1)); return out  # (центроид, высота)


def trace(ink,x0,x1,lo,hi,base):
    n=x1-x0; ys=np.full(n,np.nan); prev=base-lo
    for i in range(n):
        cl=clusters(ink[lo:hi,x0+i])
        if not cl: continue
        # ближайший к траектории; при равенстве — тоньше (не разделитель/текст)
        c,_=min(cl,key=lambda c:(abs(c[0]-prev), c[1]))
        if abs(c-prev)<=SLEW:            # ограничение скачка -> нет «прямоугольников»
            ys[i]=c; prev=c
    idx=np.arange(n); good=~np.isnan(ys)
    if good.sum()<5: return None,0.0
    return np.interp(idx,idx[good],ys[good])+lo, float(good.mean())


def to_mv(ys,mm,sec,fs,clip=3.0):
    mV=-(ys-np.median(ys))/(10*mm)
    win=int(0.6*fs); win=min(win if win%2 else win+1,(len(mV)//2)*2-1)
    if 3<=win<len(mV): mV=mV-median_filter(mV,size=win)
    return np.interp(np.linspace(0,1,int(fs*sec)),np.linspace(0,1,len(mV)),np.clip(mV,-clip,clip))


def run(core,maskp,out):
    bgr=cv2.imread(core); gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    ink=color_ink(bgr); mask=(cv2.imread(maskp,cv2.IMREAD_UNCHANGED)>0).astype(np.uint8)
    H,W=ink.shape; rows=detect_rows(mask); mm=mmpx(gray)
    xs=np.where(mask.any(0))[0]; xL,xR=int(xs.min()),int(xs.max()); cal=int(14*mm)
    colw=(xR-xL)/4; cols=[[int(xL+c*colw),int(xL+(c+1)*colw)] for c in range(4)]
    cols[0][0]+=cal
    for c in (1,2,3): cols[c][0]+=SEP_TRIM     # срезать пунктирный разделитель
    cen=rows[:3]; rhy=rows[3]; sig={}; cov={}
    for r,ct in enumerate(cen):
        up=ct-(cen[r-1] if r>0 else 0); dn=(cen[r+1] if r<2 else rhy)-ct
        lo=max(0,int(ct-0.72*up)); hi=min(H,int(ct+0.72*dn))
        for c,(x0,x1) in enumerate(cols):
            l=GRID[r][c]; b=baseline(ink,x0,x1,lo,hi)
            ys,cv=trace(ink,x0,x1,lo,hi,b)
            if ys is not None: sig[l]=to_mv(ys,mm,2.5,FS); cov[l]=round(cv,2)
    lo=max(0,int(rhy-0.72*(rhy-cen[2]))); b=baseline(ink,xL+cal,xR,lo,H)
    ys,cv=trace(ink,xL+cal,xR,lo,H,b); sig["II_rhythm"]=to_mv(ys,mm,10,FS); cov["II_rhythm"]=round(cv,2)
    order=[l for l in LEAD_ORDER+["II_rhythm"] if l in sig]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,axs=plt.subplots(len(order),1,figsize=(12,1.4*len(order)))
    for ax,l in zip(axs,order):
        ax.plot(np.arange(len(sig[l]))/FS,sig[l],lw=0.8,color='black')
        ax.set_ylabel(f"{l}\ncov={cov[l]:.0%}",rotation=0,labelpad=30,fontsize=9,va='center')
        ax.set_ylim(-2,2.5); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out,dpi=110); plt.close(); print("written",out,"covs:",cov)


if __name__=="__main__":
    import sys; run(sys.argv[1],sys.argv[2],sys.argv[3])
