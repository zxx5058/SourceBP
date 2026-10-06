# "Rich pulse" branch = 3-region waveform morphology + HRV. Fuse with appearance -> does it LOWER BP MAE?
import warnings; warnings.filterwarnings("ignore")
import os, sys, numpy as np, pandas as pd, cv2, time
np.mat=np.asmatrix; sys.path.insert(0,"ModelA"); sys.path.insert(0,"D:/mrrpg/baselines/rPPG-Toolbox")
from unsupervised_methods.methods.POS_WANG import POS_WANG
import mcd_ppg_bp as M
import mediapipe as mp
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision
from scipy.signal import butter, filtfilt, detrend, find_peaks
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
R="D:/mrrpg/mcd_rppg/"; MODEL=R+"face_landmarker.task"; FS=30.0; NF=1350; CACHE=R+"ModelA/_pulserich.npz"
CL={"fh":[10,67,69,109,108,151,9,107,336,338,297,299],"lc":[123,50,36,137,234,117,118,101,205,206,187],
    "rc":[352,280,266,366,454,346,347,330,425,426,411]}
NOTCH=["crest_time","width10","width25","width50","dia_amp","ri","aug_index","refl_time","ipa","ba","ca","da","ea","aging_index"]
d=pd.read_csv(R+"mvbp_release/cache/features_facebp.csv"); d["subject"]=d["subject"].astype(str)
def band(x,lo,hi): b,a=butter(4,[lo/(FS/2),hi/(FS/2)],"band"); return filtfilt(b,a,detrend(np.nan_to_num(x)))
if os.path.exists(CACHE):
    PR=np.load(CACHE)["pr"]; print("loaded pulse-rich cache",PR.shape)
else:
    lm=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpy.BaseOptions(model_asset_path=MODEL),running_mode=vision.RunningMode.IMAGE,num_faces=1))
    PR=np.full((len(d),len(NOTCH)*3+2),np.nan); t0=time.time()
    for k,(_,r) in enumerate(d.iterrows()):
        vid=R+f"video/{r['subject']}_FullHDwebcam_before.avi"
        if not os.path.exists(vid): continue
        cap=cv2.VideoCapture(vid); ok,f0=cap.read()
        if not ok: cap.release(); continue
        h,w=f0.shape[:2]; res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(f0,cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks: cap.release(); continue
        Lm=res.face_landmarks[0]; rect={nm:(min(int(Lm[i].x*w) for i in idc),min(int(Lm[i].y*h) for i in idc),max(int(Lm[i].x*w) for i in idc),max(int(Lm[i].y*h) for i in idc)) for nm,idc in CL.items()}
        seq={nm:[] for nm in rect}; n=0; cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        while n<NF:
            ok,fr=cap.read()
            if not ok: break
            rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
            for nm,(x0,y0,x1,y1) in rect.items(): seq[nm].append(rgb[y0:y1,x0:x1].reshape(-1,3).mean(0))
            n+=1
        cap.release()
        if n<450: continue
        pul={nm:np.asarray(POS_WANG(np.array(s)[:,None,None,:].astype(float),FS)).ravel() for nm,s in seq.items()}
        feat=[]
        for nm in ["fh","lc","rc"]:
            fd,_=M.extract_ppg_features(pul[nm],FS); feat+=[fd.get(c,np.nan) for c in NOTCH]
        comb=band(np.mean([pul[nm] for nm in pul],0),0.7,3.5); pk,_=find_peaks(comb,distance=int(0.45*FS))
        ibi=np.diff(pk)/FS*1000.0; ibi=ibi[(ibi>400)&(ibi<1500)]
        feat+=[np.std(ibi) if len(ibi)>5 else np.nan, np.sqrt(np.mean(np.diff(ibi)**2)) if len(ibi)>5 else np.nan]  # SDNN,RMSSD
        PR[k]=feat
        if (k+1)%50==0: print(f"  {k+1}/{len(d)} ({(time.time()-t0)/(k+1):.1f}s/vid)",flush=True)
    np.savez(CACHE,pr=PR); print("cached",CACHE)
# appearance + spdcov
EMB={n:np.load(R+f"ModelA/_emb_{n}.npy") for n in ["resnet50","dinov2","vit_b_16","convnext_tiny","efficientnet_b0"]}
SPDCOV=np.load(R+"ModelA/_emb_spdcov.npy")
ok=~np.all(np.isnan(PR),1); PR=np.nan_to_num(PR[ok])
sbp=d["sbp"].to_numpy(float)[ok]; dbp=d["dbp"].to_numpy(float)[ok]; groups=d["subject"].to_numpy()[ok]
for n in EMB: EMB[n]=EMB[n][ok]
SPDCOV=SPDCOV[ok]; PA=48
MOD={"RN50":(EMB["resnet50"],PA),"DINO":(EMB["dinov2"],PA),"VIT":(EMB["vit_b_16"],PA),"CNX":(EMB["convnext_tiny"],PA),
     "EFF":(EMB["efficientnet_b0"],PA),"SPDcov":(SPDCOV,PA),"PULSERICH":(PR,0)}
APP=["RN50","DINO","VIT","CNX","EFF","SPDcov"]
gkf=GroupKFold(5); gin=GroupKFold(4)
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
        A=np.column_stack([io[n] for n in names]); w,_=nnls(A,y[tr])
        if w.sum()<1e-8: w=np.ones(len(names))/len(names)
        p[te]=np.column_stack([base(n,tr,te,y[tr]) for n in names])@w
    return p
def rr(p,t): return float(pearsonr(p,t)[0]) if np.std(p)>1e-6 else 0.0
def MAE(p,y): return float(np.mean(np.abs(p-y)))
print(f"\n{ok.sum()} subjects | does 'rich pulse' (3-region morphology + HRV) LOWER the result?\n")
for tgt,y in [("SBP",sbp),("DBP",dbp)]:
    pm=MAE(np.full_like(y,y.mean()),y)
    pp=nstack(["PULSERICH"],y); pa=nstack(APP,y); pf=nstack(APP+["PULSERICH"],y)
    print(f"[{tgt}] predict-mean {pm:.2f}")
    print(f"   pulse-rich ALONE      {MAE(pp,y):.2f} (MASE {MAE(pp,y)/pm:.3f}, r {rr(pp,y):+.3f})")
    print(f"   appearance ALONE      {MAE(pa,y):.2f} (MASE {MAE(pa,y)/pm:.3f}, r {rr(pa,y):+.3f})")
    print(f"   appearance+pulse-rich {MAE(pf,y):.2f} (MASE {MAE(pf,y)/pm:.3f}, r {rr(pf,y):+.3f})  >> pulse-rich adds {MAE(pa,y)-MAE(pf,y):+.3f} mmHg")
