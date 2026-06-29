#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑通 + 评测 MMPchallenge:加载现成 XGBoost 模型(label_*.json),
按官方 Predictor.predict 接口(传"截至某 tick 的历史 df"列表)推理,
对照数据 csv 自带的真实 label,算各 horizon 准确率。
注:模型即用此数据训练,本评测为 in-sample(偏乐观),仅用于跑通+看量级。
用法:python eval_harness.py [每文件采样点数] [测试文件数]
"""
import sys, os, glob, random
import numpy as np
import pandas as pd
from Predictor import Predictor, LABEL_COLS

HERE = os.path.dirname(os.path.abspath(__file__))
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 40   # 每文件采样预测点
N_FILES  = int(sys.argv[2]) if len(sys.argv) > 2 else 8    # 测试文件数
MIN_HIST = 110   # 滚动窗口最大 100,留够历史

def main():
    random.seed(0)
    files = sorted(glob.glob(os.path.join(HERE, "data", "snapshot_*.csv")))
    # 取"日期靠后"的文件当测试(尽量像样本外)
    files = sorted(files, key=lambda p: int(p.split("date")[1].split("_")[0]))[-N_FILES*3:]
    files = random.sample(files, min(N_FILES, len(files)))
    print(f"加载模型…", flush=True)
    pred = Predictor()
    print(f"测试 {len(files)} 个文件,每文件采样 {N_SAMPLE} 点\n", flush=True)

    hit = {l: 0 for l in LABEL_COLS}; tot = {l: 0 for l in LABEL_COLS}
    # 混淆:统计预测分布(看是否一股脑判"1不变")
    pred_dist = {l: [0,0,0] for l in LABEL_COLS}
    for fp in files:
        df = pd.read_csv(fp)
        n = len(df)
        if n < MIN_HIST + 5: continue
        pts = sorted(random.sample(range(MIN_HIST, n), min(N_SAMPLE, n - MIN_HIST)))
        batch = [df.iloc[:t+1].copy() for t in pts]
        out = pred.predict(batch)   # List[List[int]]
        for t, row in zip(pts, out):
            for j, l in enumerate(LABEL_COLS):
                truth = df[l].iloc[t]
                if pd.isna(truth): continue
                p = int(row[j]); tot[l] += 1; pred_dist[l][p] += 1
                if p == int(truth): hit[l] += 1
        print(f"  ✓ {os.path.basename(fp)} ({len(pts)}点)", flush=True)

    print("\n=== 各 horizon 准确率(in-sample)===")
    print(f"{'horizon':>10} {'acc':>7} {'n':>6}   预测分布[跌/平/涨]   基线(全判平)")
    for l in LABEL_COLS:
        if tot[l] == 0: continue
        acc = hit[l] / tot[l]
        d = pred_dist[l]
        # 基线:全判"1不变"的准确率 = 真实为1的占比(此处用预测分布近似不准,单独算)
        print(f"{l:>10} {acc:7.3f} {tot[l]:6d}   {d}")
    # 真实标签分布(基线参照)
    print("\n=== 真实标签分布(全判某类的基线)===")
    allf = pd.concat([pd.read_csv(f, usecols=LABEL_COLS) for f in files], ignore_index=True)
    for l in LABEL_COLS:
        vc = allf[l].value_counts(normalize=True).sort_index()
        base = vc.max()
        print(f"  {l:>10}: 跌{vc.get(0.0,0):.2%} 平{vc.get(1.0,0):.2%} 涨{vc.get(2.0,0):.2%}  → 多数类基线 {base:.3f}")

if __name__ == "__main__":
    main()
