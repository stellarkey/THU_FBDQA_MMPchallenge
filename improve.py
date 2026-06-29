#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""样本外(OOS)实验:能否超越冠军的"单次收益率(pnl_average)"。
- 按 date 切分:train=靠前日期,test=靠后日期(模型没见过 test)。
- 两套特征:base=原版 build_features;plus=原版+微观结构因子(micro-price/OFI/多档队列不平衡等)。
- 各训 5 个 XGBoost(multi:softprob),在 val 上为每个 horizon 选最大化 pnl_average 的阈值,再上 test。
- 对照口径:pnl_average(单次收益率,= 官方模型评分的主导项)。
用法:python improve.py
"""
import os, glob, random, warnings
import numpy as np, pandas as pd
import xgboost as xgb
from Predictor import build_features, LABEL_COLS
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
HORIZ={"label_5":5,"label_10":10,"label_20":20,"label_40":40,"label_60":60}
random.seed(0); np.random.seed(0)

def extra_features(df):
    """微观结构增量因子(在 build_features 之外补充)。"""
    e={}
    b1=df["n_bid1"].to_numpy(float); a1=df["n_ask1"].to_numpy(float)
    bs1=df["n_bsize1"].to_numpy(float); as1=df["n_asize1"].to_numpy(float)
    tot1=bs1+as1+1e-12
    # micro-price:量加权中间价(对侧量权重)——短线强预测
    e["microprice"]=(a1*bs1+b1*as1)/tot1
    mid=df["n_midprice"].to_numpy(float)
    e["micro_minus_mid"]=e["microprice"]-mid
    # 多档加权队列不平衡
    bsz=np.column_stack([df[f"n_bsize{i}"].to_numpy(float) for i in range(1,6)])
    asz=np.column_stack([df[f"n_asize{i}"].to_numpy(float) for i in range(1,6)])
    w=np.array([1.0,0.8,0.6,0.4,0.2])
    wb=(bsz*w).sum(1); wa=(asz*w).sum(1)
    e["qimb_weighted"]=(wb-wa)/(wb+wa+1e-12)
    # OFI 订单流不平衡(买一/卖一价量变动近似)
    bp=pd.Series(b1); ap=pd.Series(a1); bsz1=pd.Series(bs1); asz1=pd.Series(as1)
    dbp=bp.diff().fillna(0); dap=ap.diff().fillna(0)
    ofi=(np.where(dbp>=0,bsz1,0)-np.where(dbp<0,bsz1.shift(1).fillna(0),0)
         -np.where(dap<=0,asz1,0)+np.where(dap>0,asz1.shift(1).fillna(0),0))
    e["ofi"]=ofi
    e["ofi_ma5"]=pd.Series(ofi).rolling(5,min_periods=1).mean().to_numpy()
    e["ofi_ma20"]=pd.Series(ofi).rolling(20,min_periods=1).mean().to_numpy()
    # 价差动量
    spr=(a1-b1)
    e["spread_chg"]=pd.Series(spr).diff().fillna(0).to_numpy()
    out=pd.DataFrame(e,index=df.index).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return out.astype(np.float32)

def make_feats(df, plus):
    sym=int(df["sym"].iloc[0]); sess="pm" if str(df["time"].iloc[0])>="12:00:00" else "am"
    f=build_features(df.copy(),sym=sym,session=sess)
    if plus: f=pd.concat([f, extra_features(df)],axis=1)
    return f

def load_split(files, plus):
    X=[]; Y={l:[] for l in LABEL_COLS}; RET={l:[] for l in LABEL_COLS}
    for fp in files:
        df=pd.read_csv(fp)
        if len(df)<130: continue
        f=make_feats(df,plus); mid=df["n_midprice"].to_numpy(float)
        X.append(f)
        for l in LABEL_COLS:
            N=HORIZ[l]; y=df[l].to_numpy(float)
            ret=np.full(len(df),np.nan); ret[:len(df)-N]=mid[N:]-mid[:len(df)-N]
            Y[l].append(y); RET[l].append(ret)
    X=pd.concat(X,ignore_index=True)
    for l in LABEL_COLS: Y[l]=np.concatenate(Y[l]); RET[l]=np.concatenate(RET[l])
    return X,Y,RET

def pnl_avg_at(proba, ret, thr, margin=0.2):
    pred=proba.argmax(1); top=proba.max(1)
    s=np.sort(proba,1); second=s[:,-2]
    act=(top>=thr)&(pred!=1)&((top-second)>=margin*((pred==0)|(pred==2)))
    act&=~np.isnan(ret)
    if act.sum()==0: return 0.0,0
    sign=np.where(pred[act]==2,1.0,-1.0)
    pnl=(sign*ret[act])
    return pnl.mean(), int(act.sum())

def best_thr(proba, ret, grid):
    best=(-1e9,0.5,0)
    for thr in grid:
        pa,n=pnl_avg_at(proba,ret,thr)
        if n>=30 and pa>best[0]: best=(pa,thr,n)
    return best

def run(plus, train_files, val_files, test_files):
    Xtr,Ytr,_=load_split(train_files,plus)
    Xva,_,RETva=load_split(val_files,plus)
    Xte,_,RETte=load_split(test_files,plus)
    dtr=xgb.DMatrix(Xtr); dva=xgb.DMatrix(Xva); dte=xgb.DMatrix(Xte)
    grid=np.arange(0.40,0.96,0.02)
    res={}
    for l in LABEL_COLS:
        ytr=Ytr[l]; m=~np.isnan(ytr)
        booster=xgb.train({"objective":"multi:softprob","num_class":3,"max_depth":6,
                           "eta":0.1,"subsample":0.8,"colsample_bytree":0.8,
                           "tree_method":"hist","verbosity":0},
                          xgb.DMatrix(Xtr[m],label=ytr[m].astype(int)),num_boost_round=200)
        pva=booster.predict(dva); pte=booster.predict(dte)
        pa_v,thr,nv=best_thr(pva,RETva[l],grid)
        pa_t,nt=pnl_avg_at(pte,RETte[l],thr)
        res[l]=dict(thr=thr,val_pa=pa_v,test_pa=pa_t,test_acts=nt)
    return res

def main():
    syms=list(range(10))
    def files_for(dates):
        out=[]
        for s in syms:
            for d in dates:
                out+=glob.glob(os.path.join(HERE,"data",f"snapshot_sym{s}_date{d}_*.csv"))
        return out
    train_d=list(range(0,46,2))     # 训练:0..44 偶数日(~23天)
    val_d=list(range(46,56))        # 验证:46..55
    test_d=list(range(56,79))       # 测试:56..78(模型没见过)
    tr=files_for(train_d); va=files_for(val_d); te=files_for(test_d)
    print(f"训练文件{len(tr)} / 验证{len(va)} / 测试{len(te)}",flush=True)
    for plus,name in [(False,"base 原版特征(冠军同款)"),(True,"plus 加微观结构因子")]:
        print(f"\n>>> 训练+评测:{name} …",flush=True)
        res=run(plus,tr,va,te)
        print(f"{'horizon':>9} {'阈值':>5} {'val单次收益':>11} {'test单次收益':>12} {'test出手':>7}")
        tavg=[]
        for l in LABEL_COLS:
            r=res[l]; tavg.append(r['test_pa'])
            print(f"{l:>9} {r['thr']:5.2f} {r['val_pa']:11.6f} {r['test_pa']:12.6f} {r['test_acts']:7d}")
        print(f"  ★ test 单次收益率均值 = {np.mean(tavg):.6f}")

if __name__=="__main__":
    main()
