"""Backtest: all_filtered >50亿 market cap, 2020-2025"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import *
from factor_engine import *
from backtest_engine import BacktestEngine

init_tushare()

# Get market cap for all stocks
print('Getting market cap data...', flush=True)
pool = pd.read_csv(CACHE_DIR / 'stock_pool_all_20260624.csv', dtype=str)
all_codes = pool['ts_code'].tolist()

# Try to get latest market cap from daily_basic
pro = get_pro()
mv_data = []
for batch_start in range(0, len(all_codes), 500):
    batch = all_codes[batch_start:batch_start+500]
    try:
        df = pro.daily_basic(ts_code=','.join(batch),
                             fields='ts_code,total_mv')
        if df is not None and len(df) > 0:
            mv_data.append(df)
        print('  batch {}/{}: {} stocks'.format(batch_start//500+1, (len(all_codes)+499)//500, len(df) if df is not None else 0), flush=True)
    except Exception as e:
        print('  batch {}/{} error: {}'.format(batch_start//500+1, (len(all_codes)+499)//500, str(e)[:80]), flush=True)

if mv_data:
    mv_df = pd.concat(mv_data, ignore_index=True)
    mv_df = mv_df.dropna(subset=['total_mv'])
    # Filter > 50亿 = 5,000,000,000
    # tushare total_mv unit: 万元
    big_caps = mv_df[mv_df['total_mv'] > 500000]  # 50亿 = 500000万
    stock_list = big_caps['ts_code'].tolist()
    print('Filtered: {} stocks (total_mv > 50亿)'.format(len(stock_list)), flush=True)
else:
    print('API failed, falling back to top1500', flush=True)
    pool = pd.read_csv(CACHE_DIR / 'stock_pool_top1500_20260624.csv', dtype=str)
    stock_list = pool['ts_code'].tolist()

# Backtest
start, end = '20200101', '20251201'
print('Loading {} ~ {}...'.format(start, end), flush=True)
stock_data = load_multi_stock_data(stock_list, start, end)
stock_data = filter_by_liquidity(stock_data, min_daily_amount=20_000)
print('After liquidity: {} stocks'.format(len(stock_data)), flush=True)

index_df = download_index_daily('000001.SH', start, end)
print('Index: {} days'.format(len(index_df)), flush=True)

print('Factors...', flush=True)
valid = {}
for i, (code, df) in enumerate(stock_data.items()):
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j','bbi','atr14','rebound_elasticity'])) >= 60:
            valid[code] = df_f
    except: pass
    if i % 100 == 99: print('  {}/{}'.format(i+1, len(stock_data)), flush=True)
print('Valid: {}'.format(len(valid)), flush=True)

stock_sectors = get_stock_sectors()
print('Industry coverage: {}/{}'.format(sum(1 for c in valid if c in stock_sectors), len(valid)), flush=True)

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
