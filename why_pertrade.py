#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""演示:为什么排名用"单次收益率"而非"总收益"。
同一模型,把出手阈值从低(几乎全开枪)扫到高(极度精选),
看 出手数 / 总收益 / 单次收益 / 夏普(信息比率) 各自怎么变。
用现成模型在样本上跑(研究指标行为,in-sample 足够)。
"""
import os, glob, random
import numpy as np, pandas as pd
import xgboost as xgb
from Predictor import build_features, LABEL_COLS
HERE=os.path.dirname(os.path.abspath(__file__))
HORIZ={"label_5":5,"label_10":10,"label_20":20,"label_40":40,"label_60":60}

def main(label="label_20", nf=25):
    random.seed(0)
    files=random.sample(sorted(glob.glob(os.path.join(HERE,"data","snapshot_*.csv"))),nf)
    booster=xgb.Booster(); booster.load_model(os.path.join(HERE,f"{label}.json"))
    N=HORIZ[label]
    P=[]; R=[]  # 每个 tick 的 (proba行, 未来N tick收益)
    for fp in files:
        df=pd.read_csv(fp)
        if len(df)<130: continue
        sym=int(df["sym"].iloc[0]); sess="pm" if str(df["time"].iloc[0])>="12:00:00" else "am"
        feat=build_features(df.copy(),sym=sym,session=sess)
        proba=booster.predict(xgb.DMatrix(feat))
        mid=df["n_midprice"].to_numpy(float)
        ret=np.full(len(df),np.nan); ret[:len(df)-N]=mid[N:]-mid[:len(df)-N]
        P.append(proba); R.append(ret)
    P=np.vstack(P); R=np.concatenate(R)
    ok=~np.isnan(R); P=P[ok]; R=R[ok]
    pred=P.argmax(1); top=P.max(1)
    total_ticks=len(R)
    print(f"=== {label}:阈值扫描(共 {total_ticks} 个可交易 tick)===")
    print(f"{'阈值':>5} {'出手数':>7} {'出手占比':>7} {'总收益':>9} {'单次收益':>10} {'夏普(IR)':>9}")
    for thr in [0.34,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
        act=(top>=thr)&(pred!=1)
        n=int(act.sum())
        if n==0:
            print(f"{thr:5.2f} {0:7d}      —          —          —         —"); continue
        sign=np.where(pred[act]==2,1.0,-1.0)
        pnl=sign*R[act]
        total=pnl.sum(); avg=pnl.mean(); std=pnl.std()+1e-12
        sharpe=avg/std*np.sqrt(n)   # 信息比率(规模无关的风险调整)
        print(f"{thr:5.2f} {n:7d} {n/total_ticks:7.1%} {total:9.3f} {avg:10.6f} {sharpe:9.2f}")
    print("\n要点:阈值↓→出手多→【总收益】堆高(靠走量);阈值↑→出手少→【单次收益】才高(靠真本事)。")
    print("     【夏普/IR】在中间见顶——兼顾边际胜率与规模,才是教科书级口径。")

if __name__=="__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "label_20")
