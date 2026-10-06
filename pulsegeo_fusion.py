# Does the Riemannian packaging (SPD/Jet/Scattering) on the PROPER 3-region pulse add on top of
# rich-pulse (morphology+HRV)? Re-extract 3-region pulses, compute geometry, ablate vs appearance.
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd, cv2, time
np.mat=np.asmatrix
sys.path.insert(0,"D:/mrrpg/mcd_rppg/mvbp_release"); import features as FT
sys.path.insert(0,"D:/mrrpg/baselines/rPPG-Toolbox")
from unsupervised_methods.methods.POS_WANG import POS_WANG
import mediapipe as mp
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision
from scipy.signal import detrend
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
R="D:/mrrpg/mcd_rppg/"; MODEL=R+"face_landmarker.task"; FS=30.0; NF=1350; GEOC=R+"ModelA/_pulsegeo.npz"
CL={"fh":[10,67,69,109,108,151,9,107,336,338,297,299],"lc":[123,50,36,137,234,117,118,101,205,206,187],
    "rc":[352,280,266,366,454,346,347,330,425,426,411]}
d=pd.read_csv(R+"mvbp_release/cache/features_facebp.csv"); d["subject"]=d["subject"].astype(str)
if os.path.exists(GEOC):
    GEO=np.load(GEOC)["geo"]; print("loaded geometry cache",GEO.shape)
else:
    lm=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpy.BaseOptions(model_asset_path=MODEL),running_mode=vision.RunningMode.IMAGE,num_faces=1))
    GEO=np.full((len(d), 0),0.0); rows=[]; t0=time.time(); c=0
    for k,(_,r) in enumerate(d.iterrows()):
        vid=R+f"video/{r['subject']}_FullHDwebcam_before.avi"
        if not os.path.exists(vid): rows.append(None); continue
        cap=cv2.VideoCapture(vid); ok,f0=cap.read()
        if not ok: cap.release(); rows.append(None); continue
        h,w=f0.shape[:2]; res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(f0,cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks: cap.release(); rows.append(None); continue
        Lm=res.face_landmarks[0]; rect={nm:(min(int(Lm[i].x*w) for i in idc),min(int(Lm[i].y*h) for i in idc),max(int(Lm[i].x*w) for i in idc),max(int(Lm[i].y*h) for i in idc)) for nm,idc in CL.items()}
        seq={nm:[] for nm in rect}; n=0; cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        while n<NF:
            ok,fr=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
            for nm,(x0,y0,x1,y1) in rect.items(): seq[nm].append(rgb[y0:y1,x0:x1].reshape(-1,3).mean(0))
            n+=1
        cap.release()
        if n<450: rows.append(None); continue
        pul={nm:detrend(np.nan_to_num(np.asarray(POS_WANG(np.array(s)[:,None,None,:].astype(float),FS)).ravel())) for nm,s in seq.items()}
        comb=np.mean([pul[nm] for nm in pul],0); stack=np.stack([pul["fh"],pul["lc"],pul["rc"]],1)
        rows.append(np.concatenate([FT.spd_tan(comb),FT.jet_feats(stack),FT.scattering(comb,fs=FS)]))
        c+=1
        if c%50==0: print(f"  {c} ({(time.time()-t0)/c:.1f}s/vid)",flush=True)
    dim=max(len(x) for x in rows if x is not None); GEO=np.array([x if x is not None else np.full(dim,np.nan) for x in rows])
    np.savez(GEOC,geo=GEO); print("cached",GEOC)
# load rich-pulse + appearance, align
PR=np.load(R+"ModelA/_pulserich.npz")["pr"]
ok=(~np.all(np.isnan(PR),1)) & (~np.all(np.isnan(GEO),1)); PR=np.nan_to_num(PR[ok]); GEO=np.nan_to_num(GEO[ok])
EMB={n:np.load(R+f"ModelA/_emb_{n}.npy")[ok] for n in ["resnet50","dinov2","vit_b_16","convnext_tiny","efficientnet_b0"]}
SPDCOV=np.load(R+"ModelA/_emb_spdcov.npy")[ok]
sbp=d["sbp"].to_numpy(float)[ok]; dbp=d["dbp"].to_numpy(float)[ok]; groups=d["subject"].to_numpy()[ok]; PA=48
MOD={"RN50":(EMB["resnet50"],PA),"DINO":(EMB["dinov2"],PA),"VIT":(EMB["vit_b_16"],PA),"CNX":(EMB["convnext_tiny"],PA),
     "EFF":(EMB["efficientnet_b0"],PA),"SPDcov":(SPDCOV,PA),"RICH":(PR,0),"GEO":(GEO,0)}
APP=["RN50","DINO","VIT","CNX","EFF","SPDcov"]; gkf=GroupKFold(5); gin=GroupKFold(4)
def base(nm,tr,te,y):
    F,pca=MOD[nm]; a,b=np.nan_to_num(F[tr]),np.nan_to_num(F[te])
    if pca: pc=PCA(min(pca,a.shape[1])).fit(a); a,b=pc.transform(a),pc.transform(b)
    mu,sd=a.mean(0),a.std(0)+1e-8; return Ridge(alpha=10).fit((a-mu)/sd,y).predict((b-mu)/sd)
def nstack(names,y):
    p=np.zeros(len(y))
    for tr,te in gkf.split(y,y,groups):
        gt=groups[tr]; io={n:np.zeros(len(tr)) for n in names}
        for itr,ite in gin.split(tr,y[tr],gt):
            for n in names: io[n][ite]=base(n,tr[itr],tr[ite],y[tr][itr])
        A=np.column_stack([io[n] for n in names]); w,_=nnls(A,y[tr]); w=w if w.sum()>1e-8 else np.ones(len(names))/len(names)
        p[te]=np.column_stack([base(n,tr,te,y[tr]) for n in names])@w
    return p
def rr(p,t): return float(pearsonr(p,t)[0]) if np.std(p)>1e-6 else 0.0
def MAE(p,y): return float(np.mean(np.abs(p-y)))
print(f"\n{ok.sum()} subjects | does Riemannian GEO add on top of rich-pulse?\n")
for tgt,y in [("SBP",sbp),("DBP",dbp)]:
    pm=MAE(np.full_like(y,y.mean()),y)
    g=nstack(["GEO"],y); a=nstack(APP,y); ar=nstack(APP+["RICH"],y); arg=nstack(APP+["RICH","GEO"],y); ag=nstack(APP+["GEO"],y)
    print(f"[{tgt}] predict-mean {pm:.2f}")
    print(f"   GEO alone (Riemannian)     {MAE(g,y):.2f} (MASE {MAE(g,y)/pm:.3f})")
    print(f"   appearance                 {MAE(a,y):.2f} ({MAE(a,y)/pm:.3f})")
    print(f"   appearance+GEO             {MAE(ag,y):.2f} ({MAE(ag,y)/pm:.3f})  [GEO adds {MAE(a,y)-MAE(ag,y):+.3f}]")
    print(f"   appearance+RICH            {MAE(ar,y):.2f} ({MAE(ar,y)/pm:.3f})")
    print(f"   appearance+RICH+GEO        {MAE(arg,y):.2f} ({MAE(arg,y)/pm:.3f})  [GEO adds on top of RICH {MAE(ar,y)-MAE(arg,y):+.3f}]")
