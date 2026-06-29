"""Run backtest with rebound elasticity factor"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import *
from factor_engine import *
from backtest_engine import BacktestEngine

init_tushare()
pool = pd.read_csv('cache/stock_pool_top300_proxy.csv', dtype=str)
stock_data = load_multi_stock_data(pool['ts_code'].tolist(), '20200101', '20240601')
stock_data = filter_by_liquidity(stock_data, min_daily_amount=20_000)
index_df = download_index_daily('000001.SH', '20200101', '20240601')
print('Factors (now with rebound elasticity)...')
valid = {}
for i, (code, df) in enumerate(stock_data.items()):
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j','bbi','atr14','rebound_elasticity'])) >= 60:
            valid[code] = df_f
    except: pass
    if i % 50 == 49: print('  {}/{}'.format(i+1, len(stock_data)))
print('Valid: {}'.format(len(valid)))

# Elasticity distribution check
elast_vals = []
for df in list(valid.values())[:50]:
    elast_vals.extend(df['rebound_elasticity'].dropna().tail(60).tolist())
elast_vals = np.array(elast_vals)
print('Elasticity stats: mean={:.1f} median={:.1f} P25={:.1f} P75={:.1f}'.format(
    elast_vals.mean(), np.median(elast_vals),
    np.percentile(elast_vals,25), np.percentile(elast_vals,75)))

stock_sectors = get_stock_sectors()
config = {
    'initial_capital': 1000000,
    'max_positions': 5, 'max_positions_low': 3,
    'risk_per_trade': 0.02,
    'execution_model': 'next_open',
    'atr_stop_multiplier': 2.0, 'trailing_stop_multiplier': 2.0,
    'trailing_activation': 0.04, 'partial_profit_taking': True,
    'partial_profit_levels': [0.10, 0.20], 'partial_exit_ratios': [0.25, 0.25],
    'max_hold_days': 25, 'profit_deadline_days': 10,
    'profit_deadline_threshold': -0.02,
    'bbi_profit_protect_pct': 0.05, 'bbi_profit_exit_ratio': 0.3,
    'portfolio_stop_loss': 0.20, 'adx_trend_threshold': 25,
    'adaptive_params_enabled': True,
    'j_threshold': 20, 'j_60d_min_threshold': 15, 'min_score': 45,
    'batch_entry_enabled': True, 'batch_entry_days': 2,
    'batch_entry_ratios': [0.5, 0.5], 'batch_improvement_check': True,
    'score_weights': {
        'oversold': 0.14, 'volume_shrink': 0.13, 'small_candle': 0.08,
        'prior_surge': 0.12, 'reversal': 0.11, 'bbi_trend': 0.10,
        'ma60_proximity': 0.07, 'hot_industry': 0.12, 'amihud': 0.03,
        'rebound_elasticity': 0.10,
    },
}
engine = BacktestEngine(config)
daily_stats = engine.run(valid, index_df, stock_sectors=stock_sectors)
engine.report()
engine.export()
