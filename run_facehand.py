# OUR METHOD on the Face-Hand dataset (V-BPE / PLOS ONE 2024). Appearance fusion (5 frozen backbones
# + 2nd-order covariance pooling) + POS pulse geometry (SPD/JET/SCAT), nested-NNLS, subject 5-fold.
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np
np.mat=np.asmatrix
sys.path.insert(0,"D:/mrrpg/mcd_rppg/mvbp_release"); import features as FT
sys.path.insert(0,"D:/mrrpg/baselines/rPPG-Toolbox")
from unsupervised_methods.methods.POS_WANG import POS_WANG
import torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.optimize import nnls
from scipy.signal import detrend
from scipy.stats import pearsonr
MR="D:/mrrpg/mcd_rppg/"; FS=29.0; PCA_APP=48
def rr(p,t): return float(pearsonr(p,t)[0]) if np.std(p)>1e-6 else 0.0
def MAE(p,y): return float(np.mean(np.abs(p-y)))
z=np.load(MR+"ModelA/_facehand_v2.npz",allow_pickle=True)  # MCD-consistent ROI: fh/lc/rc pulse + wide 128 crop
ids=z["ids"]; FACE=z["face"]; RGB=z["rgb"]; sbp=z["sbp"]; dbp=z["dbp"]; age=z["age"]; bmi=z["bmi"]
N=len(ids); print(f"{N} videos | face {FACE.shape} rgb {RGB.shape} | SBP {sbp.mean():.1f}±{sbp.std():.1f} DBP {dbp.mean():.1f}±{dbp.std():.1f}")
dev="cuda" if torch.cuda.is_available() else "cpu"
faces=[FACE[i].astype(np.float32)/255.0 for i in range(N)]

# ---- appearance: 5 frozen backbones + 2nd-order covariance pooling ----
EMB={}
for nm in ["resnet50","dinov2","vit_b_16","convnext_tiny","efficientnet_b0"]:
    if nm=="dinov2":
        m=torch.hub.load('facebookresearch/dinov2','dinov2_vits14'); EMB[nm]=FT._embed_backbone(m,faces,224,dev)
    else:
        m,sz=FT._backbone(nm); EMB[nm]=FT._embed_backbone(m,faces,sz,dev)
    print("  embedded",nm,EMB[nm].shape,flush=True)
SPDCOV=FT._cov_pool_resnet50(faces,dev); print("  spdcov",SPDCOV.shape,flush=True)

# ---- pulse geometry from POS rPPG (mirror run_1200) ----
SPD=[];JET=[];SCAT=[]
for i in range(N):
    rgb=RGB[i]; sig=np.stack([detrend(np.nan_to_num(rgb[:,c])) for c in range(3)],1)
    sig=(sig-sig.mean(0))/(sig.std(0)+1e-8)
    snr=[FT.pulse_snr_db(sig[:,c],FS) for c in range(3)]; x=sig[:,int(np.argmax(snr))]
    SPD.append(FT.spd_tan(x)); JET.append(FT.jet_feats(sig)); SCAT.append(FT.scattering(x,fs=FS))
SPD=np.array(SPD);JET=np.array(JET);SCAT=np.array(SCAT)

MOD={"RN50":(EMB["resnet50"],PCA_APP),"DINO":(EMB["dinov2"],PCA_APP),"VIT":(EMB["vit_b_16"],PCA_APP),
     "CNX":(EMB["convnext_tiny"],PCA_APP),"EFF":(EMB["efficientnet_b0"],PCA_APP),"SPDcov":(SPDCOV,PCA_APP),
     "PULSE":(np.hstack([SPD,JET,SCAT]),0)}
names=["RN50","DINO","VIT","CNX","EFF","SPDcov","PULSE"]
groups=ids; gkf=GroupKFold(5); gin=GroupKFold(4)
def base(F,pca,tr,te,y):
    a,b=F[tr],F[te]
    if pca: pc=PCA(min(pca,a.shape[1])).fit(a); a,b=pc.transform(a),pc.transform(b)
    mu,sd=a.mean(0),a.std(0)+1e-8; return Ridge(alpha=10).fit((a-mu)/sd,y).predict((b-mu)/sd)
def single(nm,y):
    F,pca=MOD[nm]; p=np.zeros(len(y))
    for tr,te in gkf.split(y,y,groups): p[te]=base(F,pca,tr,te,y[tr])
    return p
def nstack(nm,y):
    p=np.zeros(len(y))
    for tr,te in gkf.split(y,y,groups):
        gt=groups[tr]; io={n:np.zeros(len(tr)) for n in nm}
        for itr,ite in gin.split(tr,y[tr],gt):
            for n in nm: io[n][ite]=base(MOD[n][0],MOD[n][1],tr[itr],tr[ite],y[tr][itr])
        A=np.column_stack([io[n] for n in nm]); w,_=nnls(A,y[tr])
        if w.sum()<1e-8: w=np.ones(len(nm))/len(nm)
        p[te]=np.column_stack([base(MOD[n][0],MOD[n][1],tr,te,y[tr]) for n in nm])@w
    return p
print("\n===== OUR METHOD on Face-Hand (V-BPE) dataset, subject 5-fold =====")
for tgt,y in [("SBP",sbp),("DBP",dbp)]:
    pm=MAE(np.full_like(y,y.mean()),y)
    print(f"\n[{tgt}] predict-mean {pm:.2f}")
    pflat=single("DINO",y); pcov=single("SPDcov",y); ppul=single("PULSE",y)
    print(f"   DINO {MAE(pflat,y):.2f} | 2nd-order {MAE(pcov,y):.2f} | pulse {MAE(ppul,y):.2f} (MASE {MAE(ppul,y)/pm:.3f})")
    best=min(MAE(single(n,y),y) for n in names); pf=nstack(names,y)
    print(f"   best-single {best:.2f} | FULL FUSION {MAE(pf,y):.2f} (MASE {MAE(pf,y)/pm:.3f}, r {rr(pf,y):+.3f}) | V-BPE paper {'17.1' if tgt=='SBP' else '13.2'}")
# mechanism: appearance -> age
pa=single("DINO",age); print(f"\n[mechanism] appearance(DINO)->age MAE {MAE(pa,age):.1f}yr r {rr(pa,age):+.3f} | age->SBP r {rr(age,sbp):+.3f}")
