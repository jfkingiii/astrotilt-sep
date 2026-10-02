import numpy as np, sep
from common import make_field, match
rng=np.random.default_rng(3)
def ecc(a,b): return np.sqrt(1-(b/a)**2)
for etrue in [0.0,0.6]:
  for snr in [50,1000]:
    img,truth=make_field(rng,14,40,3.5,etrue,snr)
    bkg=sep.Background(img); d=img-bkg.back(); rms=bkg.rms()
    print(f"e_true={etrue} snr={snr}  global rms={bkg.globalrms:.2f} (true 10)")
    for label,kw in [("default matched 5",dict(thresh=5)),
                     ("no filter 5",dict(thresh=5,filter_kernel=None)),
                     ("matched 10",dict(thresh=10)),("matched 20",dict(thresh=20)),
                     ("no filter 2",dict(thresh=2,filter_kernel=None))]:
        t=kw.pop('thresh'); o=sep.extract(d,t,err=rms,**kw)
        e=ecc(o['a'],o['b'])
        # full (untruncated) flux-weighted moments in fixed 6-sigma window: true intrinsic check
        print(f"  {label:20s} n={len(o):4d} med e={np.median(e):.3f}  med npix={np.median(o['npix']):.0f}")
    # unthresholded moments in a big circular window (reference)
    es=[]
    for x,y in zip(truth['x'],truth['y']):
        xi,yi=int(round(x)),int(round(y)); r=10
        st=d[yi-r:yi+r+1,xi-r:xi+r+1].astype(float); yy,xx=np.mgrid[-r:r+1,-r:r+1]
        w=st*((xx**2+yy**2)<=r*r); f=w.sum(); cx=(w*xx).sum()/f; cy=(w*yy).sum()/f
        X=(w*(xx-cx)**2).sum()/f; Y=(w*(yy-cy)**2).sum()/f; XY=(w*(xx-cx)*(yy-cy)).sum()/f
        tr=X+Y; dt=np.sqrt(((X-Y)/2)**2+XY**2); l1=tr/2+dt; l2=tr/2-dt; es.append(np.sqrt(max(0,1-l2/l1)))
    print(f"  {'untruncated r=10 moments':20s} med e={np.median(es):.3f}")
