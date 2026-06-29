#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PredictorPlus —— 最优组合的即插即用版,接口与原版 Predictor 完全一致(predict(List[DataFrame])->List[List[int]])。
配置:LightGBM 模型 + (micro-price + OFI) 微观结构因子 + max-prob 阈值门控 + 涨跌停硬判。
OOS 实测单次收益率比原版 base +27%(详见 IMPROVEMENT.md / 改进实验.md)。
依赖:plus_models/(由 train_plus.py 生成)、Predictor.build_features、improve.extra_features。
用法:
    from PredictorPlus import PredictorPlus
    pred = PredictorPlus()
    out = pred.predict([df1, df2, ...])   # 每个 df 是"截至某 tick 的历史";返回每条 [l5,l10,l20,l40,l60](0跌/1平/2涨)
"""
import os, json
from typing import List
import numpy as np, pandas as pd
import lightgbm as lgb
from Predictor import build_features
from improve import extra_features

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_COLS = ["label_5", "label_10", "label_20", "label_40", "label_60"]
UPPER_LIMIT, LOWER_LIMIT, LIMIT_ATOL = 0.1, -0.1, 5e-4


class PredictorPlus:
    def __init__(self, model_dir: str = None):
        d = model_dir or os.path.join(HERE, "plus_models")
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.cols = cfg["cols"]
        self.plus_extra = cfg["plus_extra"]
        self.thr = cfg["thresholds"]
        self.margin = cfg.get("margin", 0.2)
        self.models = {l: lgb.Booster(model_file=os.path.join(d, f"lgb_{l}.txt")) for l in LABEL_COLS}

    # ---- 与原版一致的辅助 ----
    def _infer_session(self, df):
        return "pm" if str(df["time"].iloc[0]) >= "12:00:00" else "am"

    def _limit_override(self, df):
        mid = float(df["n_midprice"].iloc[-1]); close = float(df["n_close"].iloc[-1])
        a1 = float(df["n_ask1"].iloc[-1]); b1 = float(df["n_bid1"].iloc[-1])
        hi = np.isclose(max(mid, close, a1), UPPER_LIMIT, atol=LIMIT_ATOL)
        lo = np.isclose(min(mid, close, b1), LOWER_LIMIT, atol=LIMIT_ATOL)
        if hi and lo: return 1
        if hi: return 0
        if lo: return 2
        return None

    def _features(self, df):
        sym = int(df["sym"].iloc[0]); sess = self._infer_session(df)
        base = build_features(df.copy(), sym=sym, session=sess)
        ex = extra_features(df)[self.plus_extra]
        feat = pd.concat([base, ex], axis=1)
        return feat.reindex(columns=self.cols, fill_value=0.0).iloc[[-1]]

    def _decide(self, proba_row, label):
        pred = int(np.argmax(proba_row)); s = np.sort(proba_row)
        top, second = float(s[-1]), float(s[-2]) if len(s) > 1 else 0.0
        if top < self.thr[label]: return 1
        if pred in (0, 2) and top - second < self.margin: return 1
        return pred

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        preds = []
        for df in x:
            ov = self._limit_override(df)
            if ov is not None:
                preds.append([ov] * len(LABEL_COLS)); continue
            row = self._features(df)
            one = [self._decide(self.models[l].predict(row)[0], l) for l in LABEL_COLS]
            preds.append(one)
        return preds


if __name__ == "__main__":
    # 自检:随便拿一个数据文件跑几条
    import glob
    f = sorted(glob.glob(os.path.join(HERE, "data", "snapshot_*.csv")))[0]
    df = pd.read_csv(f)
    p = PredictorPlus()
    out = p.predict([df.iloc[:200], df.iloc[:500], df.iloc[:1000]])
    print("自检通过,预测样例:")
    for r in out: print("  ", r)
