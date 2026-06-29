"""选股行为深度分析：因子贡献、市场环境、选股偏好"""
import sys, pandas as pd, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from factor_engine import compute_all_factors, calc_market_regime

cache_dir = Path(__file__).parent.parent / "cache"
START, END = '20200101', '20251201'

print("=" * 65)
print("  Selection Behavior Analysis")
print("=" * 65)

# Load
stock_data = {}
for f in sorted(cache_dir.glob(f"*_{START}_{END}_qfq.csv"))[:200]:
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if not df.empty and 'close' in df.columns:
            stock_data[f.name[:9]] = df
    except: pass

valid = {}
for c, df in stock_data.items():
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j','bbi','atr14'])) >= 60:
            valid[c] = df_f
    except: pass

idx_file = cache_dir / f"IDX_000001.SH_{START}_{END}.csv"
index_df = pd.read_csv(idx_file, index_col=0, parse_dates=True) if idx_file.exists() else None
dates = sorted(index_df.index)

# Industry
ind_file = cache_dir / "stock_industry.csv"
stock_sectors = {}
if ind_file.exists():
    ind_df = pd.read_csv(ind_file, dtype=str)
    stock_sectors = dict(zip(ind_df['ts_code'], ind_df['industry']))

cfg = {'j_threshold': 20, 'min_score': 50,
       'score_weights': {'j_value': 0.15, 'volume_pattern': 0.23,
                         'bbi_trend': 0.21, 'reversal': 0.16,
                         'hot_industry': 0.15, 'quality': 0.10}}

# Track selection history
selection_log = []      # per day: {date, top_codes, top_scores, regime}
factor_contrib = []     # per day: avg sub-scores
industry_picks = Counter()  # which industries picked

for i, date in enumerate(dates):
    if i < 60: continue

    # Regime
    regime_info = calc_market_regime(index_df.loc[:date])

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
                     'volume_ratio': r.get('volume_ratio', np.nan),
                     'ret_20d': r.get('ret_20d', np.nan)})
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
    j_turn = (df_v['kdj_j_prev'].notna() & (df_v['kdj_j'] > df_v['kdj_j_prev'])).astype(float)
    above5 = (df_v['ma5'].notna() & (df_v['close'] > df_v['ma5'])).astype(float)
    shrink = (df_v['volume_ratio'].notna() & (df_v['volume_ratio'] < 0.8)).astype(float)
    df_v['score_rev'] = np.clip(j_turn * 40 + above5 * 35 + shrink * 25, 0, 100)

    # Hot industry
    if stock_sectors:
        df_v['sector'] = df_v['code'].map(stock_sectors)
        sector_ret = df_v.groupby('sector')['ret_20d'].mean()
        sector_rank = sector_ret.rank(pct=True)
        sector_score = 30 + sector_rank * 70
        df_v['score_hot'] = df_v['sector'].map(sector_score).fillna(50)
    else:
        df_v['score_hot'] = np.clip(df_v['ret_20d'].fillna(0).rank(pct=True) * 100, 0, 100)

    df_v['score'] = (df_v['score_j']*w['j_value'] + df_v['score_vol']*w['volume_pattern'] +
                     df_v['score_bbi']*w['bbi_trend'] + df_v['score_rev']*w['reversal'] +
                     df_v['score_hot']*w['hot_industry'] + 50*w['quality'])

    passed = df_v[df_v['score'] >= cfg['min_score']]
    if len(passed) == 0: continue

    top5 = passed.nlargest(5, 'score')

    selection_log.append({
        'date': date,
        'regime': regime_info['regime'],
        'n_passed': len(passed),
        'top5_codes': top5['code'].tolist(),
        'top5_score': top5['score'].mean(),
    })

    factor_contrib.append({
        'date': date,
        'regime': regime_info['regime'],
        'j': top5['score_j'].mean(),
        'vol': top5['score_vol'].mean(),
        'bbi': top5['score_bbi'].mean(),
        'rev': top5['score_rev'].mean(),
        'hot': top5['score_hot'].mean(),
        'total': top5['score'].mean(),
    })

    for code in top5['code']:
        ind = stock_sectors.get(code, 'unknown')
        industry_picks[ind] += 1

# ==== REPORT ====
df_log = pd.DataFrame(selection_log)
df_fac = pd.DataFrame(factor_contrib)

print(f"\n{'=' * 65}")
print(f"  1. MARKET REGIME BREAKDOWN")
print(f"{'=' * 65}")
for regime, grp in df_log.groupby('regime'):
    print(f"\n  [{regime}]")
    print(f"    Days active: {len(grp)}")
    print(f"    Avg candidates passed filter: {grp['n_passed'].mean():.1f}")
    print(f"    Avg top5 score: {grp['top5_score'].mean():.1f}")

print(f"\n{'=' * 65}")
print(f"  2. FACTOR CONTRIBUTION BY REGIME")
print(f"{'=' * 65}")
print(f"  {'Regime':<20} {'J值':>6} {'量价':>6} {'BBI':>6} {'反转':>6} {'热度':>6} {'总分':>6}")
num_cols = ['j','vol','bbi','rev','hot','total']
for regime, grp in df_fac.groupby('regime'):
    m = grp[num_cols].mean()
    print(f"  {regime:<20} {m['j']:6.1f} {m['vol']:6.1f} {m['bbi']:6.1f} {m['rev']:6.1f} {m['hot']:6.1f} {m['total']:6.1f}")

print(f"\n{'=' * 65}")
print(f"  3. INDUSTRY SELECTION FREQUENCY (Top-5 picks)")
print(f"{'=' * 65}")
total_picks = sum(industry_picks.values())
for ind, cnt in industry_picks.most_common(10):
    bar = '#' * int(cnt / max(1, total_picks) * 50)
    print(f"  {ind:<15} {cnt:>6} ({cnt/total_picks*100:5.1f}%) {bar}")

print(f"\n{'=' * 65}")
print(f"  4. YEARLY PERFORMANCE SNAPSHOT")
print(f"{'=' * 65}")
df_log['year'] = df_log['date'].dt.year
for year, grp in df_log.groupby('year'):
    regimes = Counter(grp['regime'])
    top_regime = regimes.most_common(1)[0][0]
    print(f"  {year}: {len(grp)} trading days | avg candidates={grp['n_passed'].mean():.0f} | "
          f"avg score={grp['top5_score'].mean():.1f} | dominant regime={top_regime}")

print(f"\nDone!")
