# Provenance-aware Super Learner: penalize base learners whose OOF preds correlate with the demographic shortcut.
# w* = argmin ||Yw-y||^2/N + lambda * sum_m rho_m w_m,  rho_m=|corr(yhat_m, yhat_demo)|,  w>=0.
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from scipy.optimize import nnls, minimize
from scipy.stats import pearsonr
R="D:/mrrpg/mcd_rppg/"; PA=48
d=pd.read_csv(R+"mvbp_release/cache/features_facebp.csv"); d["subject"]=d["subject"].astype(str)
PR=np.load(R+"ModelA/_pulserich.npz")["pr"]; GEO=np.load(R+"ModelA/_pulsegeo.npz")["geo"]
ok=(~np.all(np.isnan(PR),1))&(~np.all(np.isnan(GEO),1)); PR=np.nan_to_num(PR[ok]); GEO=np.nan_to_num(GEO[ok])
EMB={n:np.nan_to_num(np.load(R+f"ModelA/_emb_{n}.npy")[ok]) for n in ["resnet50","dinov2","vit_b_16","convnext_tiny","efficientnet_b0"]}
sbp=d.sbp.to_numpy(float)[ok]; dbp=d.dbp.to_numpy(float)[ok]; subj=d.subject.to_numpy()[ok]
age=d.age.to_numpy(float)[ok]; bmi=d.bmi.to_numpy(float)[ok]; sx=d.sexn.to_numpy(float)[ok]
def z(c): return (c-c.mean())/(c.std()+1e-8)
DEMO=np.column_stack([z(age),z(bmi),z(sx)])
MOD={**{n:(EMB[n],PA) for n in EMB},"RICH":(PR,0),"GEO":(GEO,0)}
LEARN=["resnet50","dinov2","vit_b_16","convnext_tiny","efficientnet_b0","RICH","GEO"]
def MAE(p,y): return float(np.mean(np.abs(p-y)))
def gf(idx,seed,K):
    u=np.unique(subj[idx]); rng=np.random.RandomState(seed); rng.shuffle(u); f={s:i%K for i,s in enumerate(u)}
    return np.array([f[subj[i]] for i in idx])
def base(nm,tr,te,y):
    F,pca=MOD[nm]; a,b=F[tr],F[te]
    if pca: pc=PCA(min(pca,a.shape[1])).fit(a); a,b=pc.transform(a),pc.transform(b)
    mu,sd=a.mean(0),a.std(0)+1e-8; return Ridge(10).fit((a-mu)/sd,y).predict((b-mu)/sd)
def oof1(nm,y,fa):
    p=np.zeros(len(y))
    for k in range(5):
        tr=np.where(fa!=k)[0]; te=np.where(fa==k)[0]; p[te]=base(nm,tr,te,y[tr])
    return p
def oof_demo(y,fa):
    p=np.zeros(len(y))
    for k in range(5):
        tr=np.where(fa!=k)[0]; te=np.where(fa==k)[0]; p[te]=Ridge(1.0).fit(DEMO[tr],y[tr]).predict(DEMO[te])
    return p
def solve(Y,y,c):  # min ||Yw-y||^2/N + c^T w, w>=0
    N=len(y); H=Y.T@Y/N; g=Y.T@y/N
    if np.all(c==0): w,_=nnls(Y,y); return w
    obj=lambda w: w@H@w-2*g@w+c@w; jac=lambda w: 2*H@w-2*g+c
    r=minimize(obj,np.ones(Y.shape[1])/Y.shape[1],jac=jac,bounds=[(0,None)]*Y.shape[1],method="L-BFGS-B")
    return r.x
def run(seed,lam):
    fa=gf(np.arange(len(sbp)),seed,5); o={}
    for tgt,y in [("SBP",sbp),("DBP",dbp)]:
        P=np.column_stack([oof1(n,y,fa) for n in LEARN]); yd=oof_demo(y,fa)
        pred=np.zeros(len(y))
        for k in range(5):
            tr=np.where(fa!=k)[0]; te=np.where(fa==k)[0]
            rho=np.array([abs(pearsonr(P[tr,m],yd[tr])[0]) for m in range(P.shape[1])])
            w=solve(P[tr],y[tr],lam*rho); pred[te]=P[te]@w
        o[tgt]=MAE(pred,y)
    return o
pmS=MAE(np.full_like(sbp,sbp.mean()),sbp); pmD=MAE(np.full_like(dbp,dbp.mean()),dbp)
print(f"n={ok.sum()} | Provenance-aware Super Learner (penalize demographic-correlated learners) | 5-seed")
print(f"  base learners: 5 backbones + RICH + GEO\n")
print(f"{'config':32s} | {'SBP':>6s} {'MASE':>5s} | {'DBP':>6s} {'MASE':>5s}")
for lam in [0.0,5.0,20.0,80.0]:
    rs=[run(s,lam) for s in range(5)]; S=np.mean([r['SBP'] for r in rs]); D=np.mean([r['DBP'] for r in rs])
    tag="(=standard Super Learner)" if lam==0 else ""
    print(f"{('lambda='+str(lam)+' '+tag):32s} | {S:6.2f} {S/pmS:.3f} | {D:6.2f} {D/pmD:.3f}")
