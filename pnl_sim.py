#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""近似复现官方「模型评分」核心指标 = pnl_average(单次收益率)。
对每个非"平"预测:按方向吃未来 N tick 的中间价变动作为收益。
pnl = Σ 单笔收益; pnl_average = pnl / 出手次数; 并报准确率/精确率/召回(对照)。
用现成模型(in-sample)验证模拟器是否落在冠军 ~0.009 量级。
用法:python pnl_sim.py [文件数]
"""
import sys, os, glob, random
import numpy as np, pandas as pd
import xgboost as xgb
from Predictor import build_features, LABEL_COLS, THRESHOLDS, MIN_DIRECTION_MARGIN

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZ = {"label_5":5,"label_10":10,"label_20":20,"label_40":40,"label_60":60}

def load_models():
    m={}
    for l in LABEL_COLS:
        b=xgb.Booster(); b.load_model(os.path.join(HERE,f"{l}.json")); m[l]=b
    return m

def decide(proba_row, label, thr, margin):
    pred=int(np.argmax(proba_row)); s=np.sort(proba_row)
    top=float(s[-1]); second=float(s[-2]) if len(s)>1 else 0.0
    if top<thr: return 1
    if pred in (0,2) and top-second<margin: return 1
    return pred

def eval_files(files, models, thr_map=None, margin=MIN_DIRECTION_MARGIN):
    thr_map = thr_map or THRESHOLDS
    agg={l:{"pnl":0.0,"acts":0,"hit_dir":0,"n":0} for l in LABEL_COLS}
    for fp in files:
        df=pd.read_csv(fp)
        if len(df)<130: continue
        sym=int(df["sym"].iloc[0])
        sess="pm" if str(df["time"].iloc[0])>="12:00:00" else "am"
        feat=build_features(df.copy(),sym=sym,session=sess)
        mid=df["n_midprice"].to_numpy()
        for l in LABEL_COLS:
            N=HORIZ[l]; proba=models[l].predict(xgb.DMatrix(feat))
            n=len(df)
            for t in range(n-N):   # 需有未来N tick
                d=decide(proba[t],l,thr_map[l],margin)
                if d==1: continue
                ret=mid[t+N]-mid[t]
                pnl=ret if d==2 else -ret
                agg[l]["pnl"]+=pnl; agg[l]["acts"]+=1
                truth=df[l].iloc[t]
                if not pd.isna(truth):
                    agg[l]["n"]+=1
                    if d==int(truth): agg[l]["hit_dir"]+=1
    return agg

def report(agg, title=""):
    print(f"\n=== {title} ===")
    print(f"{'horizon':>9} {'出手数':>7} {'pnl总':>9} {'单次收益pnl_avg':>14} {'方向精确率':>9}")
    tot_avg=[]
    for l in LABEL_COLS:
        a=agg[l]; acts=a["acts"]
        pavg=a["pnl"]/acts if acts else 0.0
        prec=a["hit_dir"]/a["n"] if a["n"] else 0.0
        tot_avg.append(pavg)
        print(f"{l:>9} {acts:7d} {a['pnl']:9.3f} {pavg:14.6f} {prec:9.3f}")
    print(f"  → 5个horizon 单次收益率均值 = {np.mean(tot_avg):.6f}  (冠军私榜 ~0.009)")
    return np.mean(tot_avg)

if __name__=="__main__":
    nf=int(sys.argv[1]) if len(sys.argv)>1 else 20
    random.seed(0)
    files=random.sample(sorted(glob.glob(os.path.join(HERE,"data","snapshot_*.csv"))), nf)
    models=load_models()
    agg=eval_files(files,models)
    report(agg, f"原版高阈值 / {nf}文件 (in-sample)")
