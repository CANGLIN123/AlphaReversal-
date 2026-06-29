"""Full backtest: top1500 pool, 2020-2025, all optimizations"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import *
from factor_engine import *
from backtest_engine import BacktestEngine

init_tushare()

# Use top1500 pool (≈ >50亿)
pool = pd.read_csv(CACHE_DIR / 'stock_pool_top1500_20260624.csv', dtype=str)
stock_list = pool['ts_code'].tolist()
print('Stock pool: {} stocks'.format(len(stock_list)), flush=True)

start, end = '20200101', '20251201'
print('Loading {} ~ {}...'.format(start, end), flush=True)
stock_data = load_multi_stock_data(stock_list, start, end)
stock_data = filter_by_liquidity(stock_data, min_daily_amount=20_000)
print('After liquidity: {} stocks'.format(len(stock_data)), flush=True)

index_df = download_index_daily('000001.SH', start, end)
print('Index: {} days'.format(len(index_df)), flush=True)

print('Computing factors...', flush=True)
valid = {}
for i, (code, df) in enumerate(stock_data.items()):
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j','bbi','atr14','rebound_elasticity'])) >= 60:
            valid[code] = df_f
    except: pass
    if i % 100 == 99:
        print('  Factor: {}/{}'.format(i+1, len(stock_data)), flush=True)
print('Valid: {} stocks'.format(len(valid)), flush=True)

stock_sectors = get_stock_sectors()
covered = sum(1 for c in valid if c in stock_sectors)
print('Industry coverage: {}/{}'.format(covered, len(valid)), flush=True)

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

# Benchmark: Shanghai Composite buy-and-hold
print("\n[Benchmark] Computing Shanghai Composite benchmark...", flush=True)
b_ret = index_df['close'].pct_change().dropna()
b_ny = len(b_ret) / 252
b_total = index_df['close'].iloc[-1] / index_df['close'].iloc[0] - 1
b_cagr = (1 + b_total) ** (1 / b_ny) - 1 if b_ny > 0 else 0
b_vol = b_ret.std() * np.sqrt(252)
b_cum = index_df['close'] / index_df['close'].iloc[0]
b_mdd = ((b_cum - b_cum.expanding().max()) / b_cum.expanding().max()).min()
b_sharpe = (b_cagr - 0.03) / b_vol if b_vol > 0 else 0

# Strategy metrics
s_ret = daily_stats['returns'].dropna()
s_ny = len(s_ret) / 252
s_total = daily_stats['equity'].iloc[-1] / config['initial_capital'] - 1
s_cagr = (1 + s_total) ** (1 / s_ny) - 1 if s_ny > 0 else 0
s_cum = daily_stats['equity'] / config['initial_capital']
s_mdd = ((s_cum - s_cum.expanding().max()) / s_cum.expanding().max()).min()
s_vol = s_ret.std() * np.sqrt(252)
s_sharpe = (s_cagr - 0.03) / s_vol if s_vol > 0 else 0

excess = s_total - b_total

benchmark_metrics = {
    '总收益率': f'{b_total:.2%}',
    '年化收益(CAGR)': f'{b_cagr:.2%}',
    '年化波动率': f'{b_vol:.2%}',
    '夏普比率(Sharpe)': f'{b_sharpe:.3f}',
    '最大回撤': f'{b_mdd:.2%}',
}

print(f"\n{'='*65}")
print(f"  策略 vs 基准 对比")
print(f"{'='*65}")
print(f"  {'指标':<22} {'策略':>12} {'上证指数':>12} {'超额':>12}")
print(f"  {'─'*58}")
print(f"  {'总收益率':<22} {s_total:>11.2%} {b_total:>11.2%} {excess:>11.2%} {'WIN' if excess > 0 else 'LOSS'}")
print(f"  {'年化收益(CAGR)':<22} {s_cagr:>11.2%} {b_cagr:>11.2%} {s_cagr-b_cagr:>+11.2%}")
print(f"  {'年化波动率':<22} {s_vol:>11.2%} {b_vol:>11.2%}")
print(f"  {'夏普比率':<22} {s_sharpe:>11.3f} {b_sharpe:>11.3f}")
print(f"  {'最大回撤':<22} {s_mdd:>11.2%} {b_mdd:>11.2%}")
print(f"  {'='*65}")
if excess > 0:
    print(f"  Result: Strategy outperformed by {excess:.2%}")
else:
    print(f"  Result: Strategy underperformed by {excess:.2%}")
print(f"{'='*65}")

engine.report(benchmark_metrics)
engine.export()
