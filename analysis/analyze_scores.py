"""选股分值统计 — 新评分体系 (min=50, 反转因子, 新权重)"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from factor_engine import compute_all_factors

cache_dir = Path(__file__).parent.parent / "cache"
START, END = '20200101', '20251201'

print("=" * 65)
print("  Score Analysis — New Scoring System")
print("=" * 65)
print(f"  J-threshold=20  min_score=50  weights: J=15% Vol=30% BBI=25% Rev=20% Qual=10%")
print()

# Load
print("[1/3] Loading...")
stock_data = {}
for f in sorted(cache_dir.glob(f"*_{START}_{END}_qfq.csv"))[:200]:
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if not df.empty and 'close' in df.columns:
            stock_data[f.name[:9]] = df
    except: pass
print(f"      {len(stock_data)} stocks")

print("[2/3] Factors...")
valid = {}
for c, df in stock_data.items():
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j','bbi','atr14'])) >= 60:
            valid[c] = df_f
    except: pass
print(f"      {len(valid)} valid")

# Index
idx_file = cache_dir / f"IDX_000001.SH_{START}_{END}.csv"
index_df = pd.read_csv(idx_file, index_col=0, parse_dates=True) if idx_file.exists() else None
dates = sorted(index_df.index) if index_df is not None else sorted(valid[list(valid.keys())[0]].index)

# Score
print("[3/3] Scoring...")
cfg = {'j_threshold': 20, 'min_score': 50,
       'score_weights': {'j_value': 0.15, 'volume_pattern': 0.30,
                         'bbi_trend': 0.25, 'reversal': 0.20, 'quality': 0.10}}

all_selected, all_passed, all_max = [], [], []
sub_scores = {'J值': [], '量价': [], 'BBI': [], '反转': []}

for i, date in enumerate(dates):
    if i < 60: continue

    rows = []
    for code, df in valid.items():
        if date not in df.index: continue
        r = df.loc[date]
        rows.append({'code': code, 'close': r.get('close', np.nan),
                     'kdj_j': r.get('kdj_j', np.nan), 'kdj_j_prev': r.get('kdj_j_prev', np.nan),
                     'bbi': r.get('bbi', np.nan), 'ma5': r.get('ma5', np.nan),
                     'ma60': r.get('ma60', np.nan),
                     'vol_pattern_valid': r.get('vol_pattern_valid', 0),
                     'vol_pattern_score': r.get('vol_pattern_score', 0),
                     'volume_ratio': r.get('volume_ratio', np.nan)})
    if not rows: continue

    df_s = pd.DataFrame(rows)
    mask = (df_s['kdj_j'].notna() & df_s['bbi'].notna() & df_s['ma60'].notna() &
            df_s['close'].notna() & (df_s['kdj_j'] < cfg['j_threshold']) &
            (df_s['vol_pattern_valid'] == 1) &
            (df_s['bbi'] > df_s['ma60']) & (df_s['close'] > df_s['ma60']))
    df_v = df_s.loc[mask].copy()
    if len(df_v) == 0: continue

    w = cfg['score_weights']
    df_v['score_j'] = np.maximum(0, (cfg['j_threshold'] - df_v['kdj_j']) / cfg['j_threshold'] * 100)
    df_v['score_vol'] = np.minimum(100, df_v['vol_pattern_score'])
    df_v['score_bbi'] = np.minimum(100, (df_v['bbi'] - df_v['ma60']) / df_v['ma60'] * 500)

    # 新反转因子
    j_turn = (df_v['kdj_j_prev'].notna() & (df_v['kdj_j'] > df_v['kdj_j_prev'])).astype(float)
    above5 = (df_v['ma5'].notna() & (df_v['close'] > df_v['ma5'])).astype(float)
    shrink = (df_v['volume_ratio'].notna() & (df_v['volume_ratio'] < 0.8)).astype(float)
    df_v['score_rev'] = np.clip(j_turn * 40 + above5 * 35 + shrink * 25, 0, 100)

    df_v['score'] = (df_v['score_j'] * w['j_value'] + df_v['score_vol'] * w['volume_pattern'] +
                     df_v['score_bbi'] * w['bbi_trend'] + df_v['score_rev'] * w['reversal'] +
                     50 * w['quality'])

    passed = df_v[df_v['score'] >= cfg['min_score']]
    all_passed.extend(passed['score'].tolist())
    all_max.append(df_v['score'].max())

    top5 = passed.nlargest(5, 'score')
    all_selected.extend(top5['score'].tolist())
    sub_scores['J值'].extend(top5['score_j'].tolist())
    sub_scores['量价'].extend(top5['score_vol'].tolist())
    sub_scores['BBI'].extend(top5['score_bbi'].tolist())
    sub_scores['反转'].extend(top5['score_rev'].tolist())

# Output
print(f"\n{'=' * 65}")
print(f"  COMPREHENSIVE SCORE — TOP-5 SELECTED")
print(f"{'=' * 65}")
s = pd.Series(all_selected)
print(f"  Count : {len(s):,}")
print(f"  Mean  : {s.mean():.1f}")
print(f"  Median: {s.median():.1f}")
print(f"  Std   : {s.std():.1f}")
print(f"  Min   : {s.min():.1f}")
print(f"  Max   : {s.max():.1f}")
print(f"\n  Percentiles: P10={s.quantile(0.10):.1f}  P25={s.quantile(0.25):.1f}  "
      f"P50={s.quantile(0.50):.1f}  P75={s.quantile(0.75):.1f}  "
      f"P90={s.quantile(0.90):.1f}  P95={s.quantile(0.95):.1f}")

print(f"\n  Buckets:")
for lo, hi in [(0,50),(50,52),(52,54),(54,56),(56,58),(58,60),(60,65),(65,70),(70,100)]:
    cnt = ((s >= lo) & (s < hi)).sum()
    bar = '#' * int(cnt / max(1, len(s)) * 50)
    print(f"    [{lo:3d}-{hi:3d}): {cnt:6d} ({cnt/len(s)*100:5.1f}%) {bar}")

# All passed
print(f"\n{'=' * 65}")
print(f"  ALL PASSED (score >= 50)")
print(f"{'=' * 65}")
ap = pd.Series(all_passed)
print(f"  Count : {len(ap):,}")
print(f"  Mean  : {ap.mean():.1f}  Median: {ap.median():.1f}  Max: {ap.max():.1f}")

# Sub scores
print(f"\n{'=' * 65}")
print(f"  SUB-SCORE BREAKDOWN — Top-5 Selected")
print(f"{'=' * 65}")
for name, data in sub_scores.items():
    ss = pd.Series(data)
    print(f"  {name:6s} | weight={cfg['score_weights'].get('j_value' if name=='J值' else 'volume_pattern' if name=='量价' else 'bbi_trend' if name=='BBI' else 'reversal',0)*100:.0f}% | "
          f"mean={ss.mean():5.1f}  median={ss.median():5.1f}  "
          f"P10={ss.quantile(0.10):5.1f}  P25={ss.quantile(0.25):5.1f}  "
          f"P75={ss.quantile(0.75):5.1f}  P90={ss.quantile(0.90):5.1f}")

# Sub-score contribution to total (weighted)
print(f"\n  Weighted contribution to total score:")
total_mean = s.mean()
for name, data in sub_scores.items():
    ss = pd.Series(data)
    wgt = {'J值':0.15,'量价':0.30,'BBI':0.25,'反转':0.20}[name]
    contrib = ss.mean() * wgt
    print(f"    {name}: {ss.mean():.1f} * {wgt:.2f} = {contrib:.1f}  ({contrib/total_mean*100:.1f}%)")

print(f"\nDone!")
