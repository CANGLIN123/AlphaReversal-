"""
因子相关性分析 — 多因子模型的共线性检验
计算因子间的 Pearson/Spearman 相关系数矩阵

面试必问: "你的因子之间独立性如何？"
标准回答: "我做了相关性矩阵，相关性 >0.7 的因子做了正交化或剔除"
"""
import sys, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import *
from factor_engine import *

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

init_tushare()

print("Loading top300 stocks...")
pool = pd.read_csv(CACHE_DIR / 'stock_pool_top300_proxy.csv', dtype=str)
stock_data = load_multi_stock_data(pool['ts_code'].tolist(), '20200101', '20251201')
stock_data = filter_by_liquidity(stock_data, min_daily_amount=20_000)

print("Computing factors...")
valid = {}
for i, (code, df) in enumerate(stock_data.items()):
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j','bbi','atr14'])) >= 120:
            valid[code] = df_f
    except: pass

print(f"Valid: {len(valid)} stocks")

# Build factor panel: sample every 20 days to reduce time-series correlation
FACTOR_DEFS = [
    ('超卖深度',      'kdj_j_60d_min'),
    ('KDJ_J值',       'kdj_j'),
    ('KDJ_K值',       'kdj_k'),
    ('缩量比',        'volume_ratio'),
    ('BBI偏离',       'bbi_gap'),
    ('ATR波动',       'atr14'),
    ('ADX趋势',       'adx'),
    ('20日动量',      'ret_20d'),
    ('反弹弹性',      'rebound_elasticity'),
    ('Amihud流动性',  'amihud'),
    ('换手率变化',    'turnover_change'),
]

rows = []
for code, df in valid.items():
    sampled = df.iloc[::20]  # sample every 20 days
    for idx in sampled.index:
        row = sampled.loc[idx]
        r = {}
        # Compute derived factors
        bbi_v = row.get('bbi', np.nan)
        ma60_v = row.get('ma60', np.nan)
        r['code'] = code
        r['kdj_j_60d_min'] = row.get('kdj_j_60d_min', np.nan)
        r['kdj_j'] = row.get('kdj_j', np.nan)
        r['kdj_k'] = row.get('kdj_k', np.nan)
        r['volume_ratio'] = row.get('volume_ratio', np.nan)
        r['bbi_gap'] = (bbi_v - ma60_v) / ma60_v if (not pd.isna(bbi_v) and not pd.isna(ma60_v) and ma60_v > 0) else np.nan
        r['atr14'] = row.get('atr14', np.nan)
        r['adx'] = row.get('adx', np.nan)
        r['ret_20d'] = row.get('ret_20d', np.nan)
        r['rebound_elasticity'] = row.get('rebound_elasticity', np.nan)
        r['amihud'] = row.get('amihud', np.nan)
        r['turnover_change'] = row.get('turnover_change', np.nan)
        rows.append(r)

df_panel = pd.DataFrame(rows).dropna()
print(f"Panel: {len(df_panel)} observations")

# Correlation matrix
factor_cols = [c for _, c in FACTOR_DEFS]
corr = df_panel[factor_cols].corr()

# Print
print("\n" + "=" * 70)
print("  因子相关性矩阵 (Pearson)")
print("=" * 70)

labels = [n for n, _ in FACTOR_DEFS]
print(f"  {'':>14}", end='')
for l in labels:
    print(f"{l:>8}", end='')
print()

for i, (name, col) in enumerate(FACTOR_DEFS):
    print(f"  {name:>14}", end='')
    for j, (n2, c2) in enumerate(FACTOR_DEFS):
        v = corr.loc[col, c2]
        if i == j:
            print(f"  {'1.00':>6}", end='  ')
        elif abs(v) > 0.5:
            print(f"\033[91m{v:>8.2f}\033[0m", end='')  # red for high corr
        elif abs(v) > 0.3:
            print(f"\033[93m{v:>8.2f}\033[0m", end='')  # yellow for medium
        else:
            print(f"{v:>8.2f}", end='')
    print()

# Find high-correlation pairs
print(f"\n  高相关对 (|r| > 0.5):")
high_pairs = []
for i in range(len(FACTOR_DEFS)):
    for j in range(i + 1, len(FACTOR_DEFS)):
        v = corr.loc[factor_cols[i], factor_cols[j]]
        if abs(v) > 0.5:
            high_pairs.append((labels[i], labels[j], v))
            print(f"    {labels[i]} <-> {labels[j]}: r = {v:.3f}")

if not high_pairs:
    print("    无！各因子独立性良好 ✅")

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            xticklabels=labels, yticklabels=labels,
            ax=ax, annot_kws={'fontsize': 8})
ax.set_title('因子相关性矩阵', fontsize=14, fontweight='bold')
plt.tight_layout()
output_dir = Path(__file__).parent.parent / "output"
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "factor_correlation.png", dpi=150, bbox_inches='tight')
print(f"\n图表已保存: {output_dir / 'factor_correlation.png'}")
