#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四组进阶实验(OOS):
A. 选股目标:max 单次收益 vs max 夏普 —— 看夏普目标能否在保收益的同时把出手数提上来
B. 因子剪枝:base / +micro / +micro+ofi / all —— 找最优因子子集
C. 换模型:XGBoost vs LightGBM(all 特征)
D. 容量曲线:阈值扫描下 出手数/总收益/单次收益/夏普 的完整权衡(出图)
口径:单次收益率 pnl_avg、出手数 n、夏普 IR=pnl_avg/std*sqrt(n)。
"""
import os, glob, warnings
import numpy as np, pandas as pd
import xgboost as xgb, lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Predictor import build_features, LABEL_COLS
from improve import extra_features, HORIZ
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
GRP={"micro":["microprice","micro_minus_mid"],"ofi":["ofi","ofi_ma5","ofi_ma20"],
     "qimb":["qimb_weighted","spread_chg"]}

def make_feats(df):
    sym=int(df["sym"].iloc[0]); sess="pm" if str(df["time"].iloc[0])>="12:00:00" else "am"
    return pd.concat([build_features(df.copy(),sym=sym,session=sess),extra_features(df)],axis=1)

def load(files,stride=1):
    X=[];Y={l:[] for l in LABEL_COLS};RET={l:[] for l in LABEL_COLS}
    for fp in files:
        df=pd.read_csv(fp)
        if len(df)<130: continue
        f=make_feats(df); mid=df["n_midprice"].to_numpy(float)
        if stride>1: f=f.iloc[::stride]
        idx=f.index; X.append(f)
        for l in LABEL_COLS:
            N=HORIZ[l]; y=df[l].to_numpy(float)
            ret=np.full(len(df),np.nan); ret[:len(df)-N]=mid[N:]-mid[:len(df)-N]
            Y[l].append(y[idx]); RET[l].append(ret[idx])
    X=pd.concat(X,ignore_index=True)
    for l in LABEL_COLS: Y[l]=np.concatenate(Y[l]); RET[l]=np.concatenate(RET[l])
    return X.reset_index(drop=True),Y,RET

def cols_for(allcols,cfg):
    extra=sum(GRP.values(),[]); base=[c for c in allcols if c not in extra]
    return {"base":base,"micro":base+GRP["micro"],"micro_ofi":base+GRP["micro"]+GRP["ofi"],
            "all":list(allcols)}[cfg]

def metrics(proba,ret,thr):
    pred=proba.argmax(1); top=proba.max(1)
    act=(top>=thr)&(pred!=1)&(~np.isnan(ret)); n=int(act.sum())
    if n<1: return 0.0,0,0.0,0.0
    sign=np.where(pred[act]==2,1.0,-1.0); pnl=sign*ret[act]
    avg=pnl.mean(); std=pnl.std()+1e-12
    return avg,n,float(pnl.sum()),avg/std*np.sqrt(n)

def best_thr(proba,ret,grid,obj):
    best=(-1e9,0.5)
    for thr in grid:
        avg,n,tot,shp=metrics(proba,ret,thr)
        if n<30: continue
        score=avg if obj=="pertrade" else shp
        if score>best[0]: best=(score,thr)
    return best[1]

def train_xgb(X,Y,cols,seed=0):
    M={}
    for l in LABEL_COLS:
        y=Y[l]; m=~np.isnan(y)
        M[l]=xgb.train({"objective":"multi:softprob","num_class":3,"max_depth":6,"eta":0.1,
            "subsample":0.8,"colsample_bytree":0.8,"tree_method":"hist","seed":seed,"verbosity":0},
            xgb.DMatrix(X.loc[m,cols],label=y[m].astype(int)),num_boost_round=150)
    return M
def pred_xgb(M,X,cols): return {l:M[l].predict(xgb.DMatrix(X[cols])) for l in LABEL_COLS}

def train_lgb(X,Y,cols,seed=0):
    M={}
    for l in LABEL_COLS:
        y=Y[l]; m=~np.isnan(y)
        M[l]=lgb.train({"objective":"multiclass","num_class":3,"max_depth":7,"num_leaves":63,
            "learning_rate":0.1,"feature_fraction":0.8,"bagging_fraction":0.8,"seed":seed,"verbose":-1},
            lgb.Dataset(X.loc[m,cols],label=y[m].astype(int)),num_boost_round=150)
    return M
def pred_lgb(M,X,cols): return {l:M[l].predict(X[cols]) for l in LABEL_COLS}

def main():
    syms=range(10)
    ff=lambda ds:[p for s in syms for d in ds for p in glob.glob(os.path.join(HERE,"data",f"snapshot_sym{s}_date{d}_*.csv"))]
    print("加载数据…",flush=True)
    Xtr,Ytr,_=load(ff(range(0,46,2)),stride=2)
    Xva,_,RETva=load(ff(range(46,56))); Xte,_,RETte=load(ff(range(56,79)))
    allcols=list(Xtr.columns); grid=np.arange(0.40,0.94,0.02)
    print(f"train {len(Xtr)} / val {len(Xva)} / test {len(Xte)}\n",flush=True)

    # ---- B. 因子剪枝 ----
    print("=== B. 因子剪枝(test 单次收益率均值)===",flush=True)
    xgb_models={}
    for cfg in ["base","micro","micro_ofi","all"]:
        cols=cols_for(allcols,cfg); M=train_xgb(Xtr,Ytr,cols); xgb_models[cfg]=(M,cols)
        pv=pred_xgb(M,Xva,cols); pt=pred_xgb(M,Xte,cols); avgs=[]
        for l in LABEL_COLS:
            thr=best_thr(pv[l],RETva[l],grid,"pertrade")
            avgs.append(metrics(pt[l],RETte[l],thr)[0])
        print(f"  {cfg:>9}: {np.mean(avgs):.6f} (特征{len(cols)})",flush=True)

    # ---- A. 选股目标:单次收益 vs 夏普(用 all 模型)----
    print("\n=== A. 选股目标对比(all特征)===",flush=True)
    M,cols=xgb_models["all"]; pv=pred_xgb(M,Xva,cols); pt=pred_xgb(M,Xte,cols)
    for obj in ["pertrade","sharpe"]:
        pa=[];nn=[];sh=[];to=[]
        for l in LABEL_COLS:
            thr=best_thr(pv[l],RETva[l],grid,obj)
            avg,n,tot,shp=metrics(pt[l],RETte[l],thr); pa.append(avg);nn.append(n);sh.append(shp);to.append(tot)
        print(f"  目标={obj:>8}: 单次收益{np.mean(pa):.6f} | 出手{int(np.mean(nn))} | 总收益{np.mean(to):.3f} | 夏普{np.mean(sh):.2f}",flush=True)

    # ---- C. XGB vs LightGBM(all)----
    print("\n=== C. 模型对比(all特征, test 单次收益率均值)===",flush=True)
    Mx,cols=xgb_models["all"]; px=pred_xgb(Mx,Xte,cols); pxv=pred_xgb(Mx,Xva,cols)
    Ml=train_lgb(Xtr,Ytr,cols); pl=pred_lgb(Ml,Xte,cols); plv=pred_lgb(Ml,Xva,cols)
    for name,pv_,pt_ in [("XGBoost",pxv,px),("LightGBM",plv,pl)]:
        avgs=[metrics(pt_[l],RETte[l],best_thr(pv_[l],RETva[l],grid,"pertrade"))[0] for l in LABEL_COLS]
        print(f"  {name:>9}: {np.mean(avgs):.6f}",flush=True)

    # ---- D. 容量曲线(label_20, all-XGB)----
    print("\n=== D. 容量曲线 → capacity_curve.png ===",flush=True)
    l="label_20"; proba=px[l]; ret=RETte[l]
    ths=np.arange(0.34,0.92,0.02); rows=[]
    for thr in ths:
        avg,n,tot,shp=metrics(proba,ret,thr); rows.append((thr,n,tot,avg,shp))
    cur=pd.DataFrame(rows,columns=["thr","n_trades","total_pnl","pnl_avg","sharpe"])
    cur.to_csv(os.path.join(HERE,"capacity_curve.csv"),index=False)
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    ax[0].plot(cur.thr,cur.pnl_avg,'o-',color="tab:red",label="单次收益(per-trade)")
    a2=ax[0].twinx(); a2.plot(cur.thr,cur.sharpe,'s--',color="tab:blue",label="夏普 IR")
    ax[0].set_xlabel("出手阈值"); ax[0].set_ylabel("单次收益",color="tab:red"); a2.set_ylabel("夏普 IR",color="tab:blue")
    ax[0].set_title("单次收益↑ 随精选;夏普中间见顶")
    ax[1].plot(cur.thr,cur.total_pnl,'o-',color="tab:green",label="总收益")
    a3=ax[1].twinx(); a3.plot(cur.thr,cur.n_trades,'s--',color="tab:gray",label="出手数")
    ax[1].set_xlabel("出手阈值"); ax[1].set_ylabel("总收益",color="tab:green"); a3.set_ylabel("出手数",color="tab:gray")
    ax[1].set_title("总收益 & 出手数 随走量(阈值↓)堆高")
    fig.suptitle(f"容量曲线 (capacity curve) — {l}, OOS test",fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(HERE,"capacity_curve.png"),dpi=110)
    print("  曲线峰值:夏普最高 @阈值",round(float(cur.loc[cur.sharpe.idxmax(),"thr"]),2),
          "| 单次收益最高 @阈值",round(float(cur.loc[cur.pnl_avg.idxmax(),"thr"]),2),
          "| 总收益最高 @阈值",round(float(cur.loc[cur.total_pnl.idxmax(),"thr"]),2),flush=True)

if __name__=="__main__":
    main()
