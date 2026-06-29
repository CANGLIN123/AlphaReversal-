"""Simple IC analysis"""
import sys, pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import *
from factor_engine import *
init_tushare()

pool = pd.read_csv(CACHE_DIR / 'stock_pool_top300_proxy.csv', dtype=str)
stock_data = load_multi_stock_data(pool['ts_code'].tolist(), '20200101', '20251201')
stock_data = filter_by_liquidity(stock_data, min_daily_amount=20_000)

valid = {}
for i, (code, df) in enumerate(stock_data.items()):
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j', 'bbi', 'atr14'])) >= 120:
            valid[code] = df_f
    except: pass

print('Stocks: {}'.format(len(valid)))

factors = {
    'oversold': 'kdj_j_60d_min',
    'shrink': 'volume_ratio',
    'elasticity': 'rebound_elasticity',
    'amihud': 'amihud',
}

fwd_days = 10
results = {}
for fname, fcol in factors.items():
    all_ics = []
    for code, df in valid.items():
        vals = df[fcol].dropna()
        rets = df['close'].pct_change(fwd_days).shift(-fwd_days)
        mask = vals.index.intersection(rets.dropna().index)
        if len(mask) >= 60:
            ic, _ = spearmanr(vals.loc[mask], rets.loc[mask])
            if not np.isnan(ic):
                all_ics.append(ic)
    if all_ics:
        mean_ic = np.mean(all_ics)
        std_ic = np.std(all_ics)
        ir = mean_ic / std_ic if std_ic > 0 else 0
        results[fname] = {'ic': mean_ic, 'ir': ir, 'n': len(all_ics)}
        print('{}: IC={:.4f} IR={:.3f} (N={})'.format(fname, mean_ic, ir, len(all_ics)))

print()
print('=' * 50)
print('  Factor IC/IR (forward 10-day)')
print('=' * 50)
for n, r in sorted(results.items(), key=lambda x: abs(x[1]['ir']), reverse=True):
    grade = 'STRONG' if abs(r['ir']) > 0.2 else ('MODERATE' if abs(r['ir']) > 0.1 else 'WEAK')
    print('  {:>12}  IC={:>7.4f}  IR={:>7.3f}  [{}]'.format(n, r['ic'], r['ir'], grade))
