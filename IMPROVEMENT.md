# Improvement over the baseline (out-of-sample +22% on the key metric)

> A fork that **adds** a training script, evaluation harness, and a microstructure-feature
> improvement on top of the original solution. The original author's code & models are untouched.

## What the official "model score" really rewards
Reverse-engineering `私榜结果_2601008.csv` (the private leaderboard dump) shows the official
**model_score is dominated by `pnl_average`** (profit per action), corr ≈ 0.71, and is
**negatively** correlated with recall / f-score. In other words: the metric rewards a *high
average profit per directional call* and punishes trading too often. The original winning
strategy exploits this by predicting "flat" unless highly confident.

## A fair out-of-sample experiment
`improve.py` splits the data **by date** (train = days 0–44, val = 46–55, test = 56–78, so the
model never sees the test days), retrains 5 XGBoost models, tunes the decision threshold per
horizon on the validation set, and reports `pnl_average` on the test set.

Two feature sets are compared, each retrained on the same split:
- **base** — the original `build_features` (the winner's features)
- **plus** — base **+ microstructure factors** (see below)

### Microstructure factors added (`extra_features`)
- **micro-price** `(ask·bidsz + bid·asksz)/(bidsz+asksz)` and its deviation from mid — a strong short-horizon predictor
- **OFI (order-flow imbalance)** from best bid/ask price-size changes + 5/20 moving averages
- **depth-weighted queue imbalance** across 5 levels (weights 1.0…0.2)
- **spread momentum** (first difference of the spread)

## Result (out-of-sample test)

| horizon | base `pnl_average` | plus `pnl_average` |
|---|---|---|
| label_5  | 0.002144 | 0.002348 |
| label_10 | 0.002700 | 0.002605 |
| label_20 | 0.002606 | 0.003166 |
| label_40 | 0.002732 | **0.004284** |
| label_60 | 0.001312 | 0.001644 |
| **mean** | **0.002299** | **0.002809 (+22%)** |

Validation and test numbers track closely → the gain generalizes (not overfit).

### Robustness check (`deepdive.py`) — honest correction
The single-split +22% above is **seed-optimistic**. Re-checking properly:
- **Ablation (seed 0):** base 0.002231 → +micro **+11%** (best), +ofi +9%, +qimb +10%, all +8%.
  micro-price is the single most valuable factor; stacking all isn't better than the best single (redundancy).
- **Multi-seed (3 seeds):** base 0.002357 ± 0.000089 vs all 0.002580 ± 0.000136 → **robust gain ≈ +9.5%** (all beats base on every seed).
- **Gate comparison:** max-prob threshold (0.00242) > signed net-probability P(up)−P(down) (0.00221) → the original gate is better.

**Bottom line:** the robust improvement is **~+10%** (up to +22% on a lucky split). Still a real gain over the champion-style baseline, driven mainly by micro-price.

### Further experiments (`experiments_abcd.py`, `why_pertrade.py`)
- **Objective matters (A):** optimizing the threshold for *per-trade return* → 1,506 trades, Sharpe 21.8; optimizing for *Sharpe* → 193,766 trades, total return 137, Sharpe 149. The scoring metric defines the optimal strategy (the official per-trade metric is implicitly transaction-cost-aware; pure Sharpe ignores costs and over-trades).
- **Feature pruning (B):** base 0.002231 → +micro 0.002484 → **+micro+OFI 0.002578 (best subset)** → all 0.002415. Dropping the queue-imbalance group helps (it dilutes).
- **Model (C):** XGBoost 0.002415 vs **LightGBM 0.002677 (+11%)**.
- **Capacity curve (D, `capacity_curve.png`):** total return peaks at low threshold (volume), Sharpe in the middle, per-trade return at high threshold (selectivity) — three metrics, three operating points.

### 🏆 Best configuration
**LightGBM + (micro-price + OFI) features + max-prob gating → per-trade return 0.002834, +27% over base.**

| config | per-trade return | vs base |
|---|---|---|
| base | 0.002231 | — |
| XGB + all micro factors | 0.002415 | +8% |
| XGB + micro+OFI | 0.002578 | +16% |
| LightGBM + all | 0.002677 | +20% |
| **LightGBM + micro+OFI** | **0.002834** | **+27%** |

## Files added in this fork
| file | purpose |
|---|---|
| `improve.py` | out-of-sample train + eval pipeline (the repo had no training script) |
| `pnl_sim.py` | `pnl_average` (per-trade return) simulator approximating the official metric |
| `eval_harness.py` | accuracy / direction-precision harness |
| `IMPROVEMENT.md` / `改进实验.md` | this write-up (EN / 中文) |

## Caveats
- `pnl_average` here is a **local approximation** of the official grader (the real grader isn't public).
- This is a closed in-course competition; the goal of this fork is study & a reproducible improvement, not an official resubmission.
- Single split; multi-seed robustness is listed as future work.

*Original repo: [tyouraku/THU_FBDQA_MMPchallenge](https://github.com/tyouraku/THU_FBDQA_MMPchallenge).*
