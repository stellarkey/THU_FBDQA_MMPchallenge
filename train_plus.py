#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练并固化"最优组合"模型:LightGBM + (micro-price + OFI) 因子。
- 训练集:dates 0–60(偶数日,全标的);阈值在 dates 61–78 上按"最大化单次收益"挑选。
- 产物:plus_models/lgb_<label>.txt(5个模型) + plus_models/config.json(特征列 + 各horizon阈值)。
供 PredictorPlus.py 加载。约几分钟。
用法:python train_plus.py
"""
import os, glob, json
import numpy as np, pandas as pd
import lightgbm as lgb
from Predictor import build_features, LABEL_COLS
from improve import extra_features, HORIZ
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"plus_models"); os.makedirs(OUT,exist_ok=True)
# micro_ofi 子集 = build_features 全列 + 这 5 个增量列
PLUS_EXTRA=["microprice","micro_minus_mid","ofi","ofi_ma5","ofi_ma20"]

def feats_plus(df):
    sym=int(df["sym"].iloc[0]); sess="pm" if str(df["time"].iloc[0])>="12:00:00" else "am"
    base=build_features(df.copy(),sym=sym,session=sess)
    ex=extra_features(df)[PLUS_EXTRA]
    return pd.concat([base,ex],axis=1)

def load(files,stride=1):
    X=[];Y={l:[] for l in LABEL_COLS};RET={l:[] for l in LABEL_COLS}
    for fp in files:
        df=pd.read_csv(fp)
        if len(df)<130: continue
        f=feats_plus(df); mid=df["n_midprice"].to_numpy(float)
        if stride>1: f=f.iloc[::stride]
        idx=f.index; X.append(f)
        for l in LABEL_COLS:
            N=HORIZ[l]; y=df[l].to_numpy(float)
            ret=np.full(len(df),np.nan); ret[:len(df)-N]=mid[N:]-mid[:len(df)-N]
            Y[l].append(y[idx]); RET[l].append(ret[idx])
    X=pd.concat(X,ignore_index=True)
    for l in LABEL_COLS: Y[l]=np.concatenate(Y[l]); RET[l]=np.concatenate(RET[l])
    return X.reset_index(drop=True),Y,RET

def pnl_avg(proba,ret,thr,margin=0.2):
    pred=proba.argmax(1); top=proba.max(1); s=np.sort(proba,1); sec=s[:,-2]
    act=(top>=thr)&(pred!=1)&((top-sec)>=margin*((pred==0)|(pred==2)))&(~np.isnan(ret))
    if act.sum()<1: return -9,0
    sign=np.where(pred[act]==2,1.0,-1.0); return float((sign*ret[act]).mean()),int(act.sum())

def main():
    ff=lambda ds:[p for s in range(10) for d in ds for p in glob.glob(os.path.join(HERE,"data",f"snapshot_sym{s}_date{d}_*.csv"))]
    print("加载训练集(dates 0-60 偶数日)…",flush=True)
    Xtr,Ytr,_=load(ff(range(0,61,2)),stride=2)
    print(f"  训练 {len(Xtr)} 行, 特征 {Xtr.shape[1]}",flush=True)
    print("加载阈值调参集(dates 61-78)…",flush=True)
    Xva,_,RETva=load(ff(range(61,79)))
    cols=list(Xtr.columns); grid=np.arange(0.40,0.94,0.02); thr_map={}
    for l in LABEL_COLS:
        y=Ytr[l]; m=~np.isnan(y)
        booster=lgb.train({"objective":"multiclass","num_class":3,"max_depth":7,"num_leaves":63,
            "learning_rate":0.1,"feature_fraction":0.8,"bagging_fraction":0.8,"seed":0,"verbose":-1},
            lgb.Dataset(Xtr.loc[m,cols],label=y[m].astype(int)),num_boost_round=200)
        booster.save_model(os.path.join(OUT,f"lgb_{l}.txt"))
        pv=booster.predict(Xva[cols])
        best=(-9,0.6)
        for thr in grid:
            pa,n=pnl_avg(pv,RETva[l],thr)
            if n>=30 and pa>best[0]: best=(pa,thr)
        thr_map[l]=round(float(best[1]),3)
        print(f"  ✓ {l}: 阈值={thr_map[l]} (val单次收益 {best[0]:.6f})",flush=True)
    json.dump({"cols":cols,"plus_extra":PLUS_EXTRA,"thresholds":thr_map,"margin":0.2},
              open(os.path.join(OUT,"config.json"),"w"),ensure_ascii=False,indent=1)
    print("\n已保存 plus_models/(5个lgb模型 + config.json)",flush=True)

if __name__=="__main__":
    main()
