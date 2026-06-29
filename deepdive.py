#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深挖实验(都在 OOS 下):
A. 因子消融 —— base / +micro / +ofi / +qimb / all,看每组因子各贡献多少
B. 稳健性 —— base vs all 各跑多种子,看 +22% 是否稳定(mean±std)
C. 门控对比 —— 原版"最大概率阈值" vs "signed 净概率(P涨-P跌)"选股
口径:单次收益率 pnl_average(官方模型评分主导项)。
用法:python deepdive.py
"""
import os, glob, warnings
import numpy as np, pandas as pd
import xgboost as xgb
from Predictor import build_features, LABEL_COLS
from improve import extra_features, HORIZ
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))

# extra_features 产出的列分组(用于消融)
GRP={
 "micro":["microprice","micro_minus_mid"],
 "ofi":["ofi","ofi_ma5","ofi_ma20"],
 "qimb":["qimb_weighted","spread_chg"],
}

def make_feats(df):
    sym=int(df["sym"].iloc[0]); sess="pm" if str(df["time"].iloc[0])>="12:00:00" else "am"
    return pd.concat([build_features(df.copy(),sym=sym,session=sess), extra_features(df)],axis=1)

def load(files, stride=1):
    X=[]; Y={l:[] for l in LABEL_COLS}; RET={l:[] for l in LABEL_COLS}
    for fp in files:
        df=pd.read_csv(fp)
        if len(df)<130: continue
        f=make_feats(df); mid=df["n_midprice"].to_numpy(float)
        if stride>1: f=f.iloc[::stride]
        X.append(f)
        idx=f.index
        for l in LABEL_COLS:
            N=HORIZ[l]; y=df[l].to_numpy(float)
            ret=np.full(len(df),np.nan); ret[:len(df)-N]=mid[N:]-mid[:len(df)-N]
            Y[l].append(y[idx]); RET[l].append(ret[idx])
    X=pd.concat(X,ignore_index=True)
    for l in LABEL_COLS: Y[l]=np.concatenate(Y[l]); RET[l]=np.concatenate(RET[l])
    return X.reset_index(drop=True),Y,RET

def cols_for(allcols, cfg):
    extra=sum(GRP.values(),[])
    base=[c for c in allcols if c not in extra]
    if cfg=="base": return base
    if cfg=="all": return list(allcols)
    return base+GRP[cfg]

def gate_maxprob(proba, thr, margin=0.2):
    pred=proba.argmax(1); top=proba.max(1); s=np.sort(proba,1); sec=s[:,-2]
    return np.where((top>=thr)&(pred!=1)&((top-sec)>=margin*((pred==0)|(pred==2))), pred, 1)

def gate_signed(proba, thr, margin=0):
    # 净概率 = P(涨)-P(跌);超阈值才出方向
    net=proba[:,2]-proba[:,0]
    d=np.where(net>=thr,2,np.where(net<=-thr,0,1))
    return d

def pnl_avg(decision, ret):
    act=(decision!=1)&(~np.isnan(ret))
    if act.sum()==0: return 0.0,0
    sign=np.where(decision[act]==2,1.0,-1.0)
    return float((sign*ret[act]).mean()), int(act.sum())

def train_models(Xtr,Ytr,cols,seed):
    M={}
    for l in LABEL_COLS:
        y=Ytr[l]; m=~np.isnan(y)
        M[l]=xgb.train({"objective":"multi:softprob","num_class":3,"max_depth":6,"eta":0.1,
                        "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":seed,
                        "verbosity":0}, xgb.DMatrix(Xtr.loc[m,cols],label=y[m].astype(int)),
                        num_boost_round=150)
    return M

def eval_cfg(Xtr,Ytr,Xva,RETva,Xte,RETte,cols,seed,gate="maxprob"):
    M=train_models(Xtr,Ytr,cols,seed)
    gfn=gate_maxprob if gate=="maxprob" else gate_signed
    grid=np.arange(0.40,0.96,0.02) if gate=="maxprob" else np.arange(0.05,0.85,0.03)
    tavg=[]
    for l in LABEL_COLS:
        dva=xgb.DMatrix(Xva[cols]); dte=xgb.DMatrix(Xte[cols])
        pva=M[l].predict(dva); pte=M[l].predict(dte)
        best=(-9,0)
        for thr in grid:
            pa,n=pnl_avg(gfn(pva,thr),RETva[l])
            if n>=30 and pa>best[0]: best=(pa,thr)
        pa_t,_=pnl_avg(gfn(pte,best[1]),RETte[l])
        tavg.append(pa_t)
    return float(np.mean(tavg))

def main():
    syms=range(10)
    ff=lambda ds:[p for s in syms for d in ds for p in glob.glob(os.path.join(HERE,"data",f"snapshot_sym{s}_date{d}_*.csv"))]
    tr_f=ff(range(0,46,2)); va_f=ff(range(46,56)); te_f=ff(range(56,79))
    print(f"加载数据(train行采样stride=2提速)…",flush=True)
    Xtr,Ytr,_=load(tr_f,stride=2); Xva,_,RETva=load(va_f); Xte,_,RETte=load(te_f)
    allcols=list(Xtr.columns)
    print(f"train {len(Xtr)} / val {len(Xva)} / test {len(Xte)} 行, 特征 {len(allcols)}\n",flush=True)

    print("=== A. 因子消融(seed=0, maxprob门控)===")
    abl={}
    for cfg in ["base","micro","ofi","qimb","all"]:
        cols=cols_for(allcols,cfg)
        v=eval_cfg(Xtr,Ytr,Xva,RETva,Xte,RETte,cols,0)
        abl[cfg]=v; print(f"  {cfg:>6}: test单次收益率 {v:.6f}  (特征{len(cols)})",flush=True)
    b=abl['base']
    print(f"  → 相对 base 提升: " + " ".join(f"{k}+{(abl[k]/b-1)*100:.0f}%" for k in ['micro','ofi','qimb','all']))

    print("\n=== B. 稳健性(base vs all, 多种子)===")
    for cfg in ["base","all"]:
        cols=cols_for(allcols,cfg); vs=[]
        for sd in [0,1,2]:
            vs.append(eval_cfg(Xtr,Ytr,Xva,RETva,Xte,RETte,cols,sd))
        print(f"  {cfg:>5}: {np.mean(vs):.6f} ± {np.std(vs):.6f}  (seeds {[round(x,6) for x in vs]})",flush=True)

    print("\n=== C. 门控对比(all特征, seed=0)===")
    cols=cols_for(allcols,"all")
    for g in ["maxprob","signed"]:
        v=eval_cfg(Xtr,Ytr,Xva,RETva,Xte,RETte,cols,0,gate=g)
        print(f"  {g:>8} 门控: test单次收益率 {v:.6f}",flush=True)

if __name__=="__main__":
    main()
