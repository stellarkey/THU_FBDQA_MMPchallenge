# THU_FBDQA_MMPchallenge
college project of FBDQA course in THU

---

## 🚀 Fork additions (out-of-sample +27%)

This fork **adds** a training script, evaluation harness, and a microstructure-feature improvement on top of the original solution (original code/models untouched).

- 📈 best config **LightGBM + (micro-price + OFI) + max-prob gating** → per-trade return **+27%** over base (out-of-sample)
- The official score is driven by **per-trade return (pnl_average)**, not total return
- Writeups: **[IMPROVEMENT.md](IMPROVEMENT.md)** (EN) / **[改进实验.md](改进实验.md)** / **[经验总结.md](经验总结.md)**
- `improve.py` · `experiments_abcd.py` · `pnl_sim.py` · `PredictorPlus.py`