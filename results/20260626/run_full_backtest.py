"""全量回测 - 使用缓存数据，不调API"""
import sys, time, pandas as pd, numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))

from factor_engine import compute_all_factors, calc_market_regime, calc_adaptive_params
from backtest_engine import BacktestEngine

cache_dir = Path(__file__).parent / "cache"

# 直接从缓存读取（跳过API）
START, END = '20200101', '20251201'
DATE_TAG = f'{START}_{END}_qfq'

print("=" * 65)
print("  OPTIMIZED BACKTEST (cache-only, no API)")
print("=" * 65)

# 1. 加载缓存数据
print("[1/4] Loading from cache...")
stock_data = {}
for f in sorted(cache_dir.glob(f"*_{DATE_TAG}.csv")):
    code = f.name[:9]
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if not df.empty and 'close' in df.columns:
            stock_data[code] = df
    except Exception:
        pass
print(f"      Loaded: {len(stock_data)} stocks")

# 2. 因子计算
print("[2/4] Computing factors...")
valid = {}
t0 = time.time()
for i, (code, df) in enumerate(stock_data.items()):
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j', 'bbi', 'atr14'])) >= 60:
            valid[code] = df_f
    except Exception:
        pass
    if (i+1) % 150 == 0:
        print(f"      {i+1}/{len(stock_data)} ({time.time()-t0:.0f}s)")
print(f"      Valid: {len(valid)} ({time.time()-t0:.0f}s)")

# 3. 指数（读缓存）
print("[3/4] Index from cache...")
idx_file = cache_dir / f"IDX_000001.SH_{START}_{END}.csv"
if idx_file.exists():
    index_df = pd.read_csv(idx_file, index_col=0, parse_dates=True)
    print(f"      Loaded: {len(index_df)} days")
    regime = calc_market_regime(index_df)
    print(f"      Regime: {regime['regime']} | Trend: {regime['trend']} | Vol: {regime['vol_percentile']:.0f}%")
else:
    index_df = None
    print("      No cached index!")

# 4. 加载行业数据
print("[4/5] Industry data...")
ind_file = cache_dir / "stock_industry.csv"
stock_sectors = {}
if ind_file.exists():
    ind_df = pd.read_csv(ind_file, dtype=str)
    stock_sectors = dict(zip(ind_df['ts_code'], ind_df['industry']))
    print(f"      Loaded: {len(stock_sectors)} stocks → {ind_df['industry'].nunique()} industries")
else:
    print("      No industry file, using fallback")

# 5. 回测
print("[5/5] Running backtest...")
config = {
    'initial_capital': 1_000_000,
    'max_positions': 5,
    'max_positions_low': 3,
    'execution_model': 'next_open',
    'batch_entry_enabled': True,
    'batch_entry_days': 2,
    'batch_entry_ratios': [0.5, 0.5],
    'batch_improvement_check': True,
    'adaptive_params_enabled': True,
    'trailing_stop_multiplier': 2.0,
    'trailing_activation': 0.03,
    'partial_profit_taking': True,
    'max_hold_days': 15,
    'profit_deadline_days': 7,
    'atr_stop_multiplier': 2.0,
    'portfolio_stop_loss': 0.20,
    'j_threshold': 20,
    'min_score': 50,
    'score_weights': {
        'j_value': 0.12, 'volume_pattern': 0.20,
        'bbi_trend': 0.18, 'reversal': 0.14,
        'hot_industry': 0.13, 'ma60_proximity': 0.13,
        'quality': 0.10,
    },
}

engine = BacktestEngine(config)
stats = engine.run(valid, index_df, stock_sectors=stock_sectors)

if len(stats) > 0:
    engine.report()
    engine.export()

    if engine.trades:
        # 归类
        cats = {'Rotation': 0, 'N-day loss': 0, 'Partial profit': 0,
                'Trailing stop': 0, 'Hard stop': 0, 'BBI protect': 0,
                'Max hold': 0, 'Volume dump': 0, 'Circuit breaker': 0}
        for t in engine.trades:
            r = t['reason']
            if '调仓' in r: cats['Rotation'] += 1
            elif '天未盈利' in r: cats['N-day loss'] += 1
            elif '分批止盈' in r: cats['Partial profit'] += 1
            elif '移动止盈' in r: cats['Trailing stop'] += 1
            elif '硬止损' in r: cats['Hard stop'] += 1
            elif 'BBI' in r: cats['BBI protect'] += 1
            elif '持仓超' in r: cats['Max hold'] += 1
            elif '放量阴线' in r: cats['Volume dump'] += 1
            elif '熔断' in r: cats['Circuit breaker'] += 1

        total = len(engine.trades)
        print(f"\n{'=' * 65}")
        print(f"  SELL REASON BREAKDOWN ({total} trades)")
        print("=" * 65)
        for cat, c in sorted(cats.items(), key=lambda x: -x[1]):
            if c > 0:
                bar = '#' * int(c/total*50)
                print(f"  {cat:<18} {c:>5} ({c/total*100:>5.1f}%) {bar}")

        new_exits = total - cats['Rotation']
        print(f"\n  New signal exits: {new_exits}/{total} ({new_exits/total*100:.1f}%)")
        print(f"  Rotation exits: {cats['Rotation']}/{total} ({cats['Rotation']/total*100:.1f}%)")

    print(f"\n  Final Equity: {stats['equity'].iloc[-1]:,.0f}")
else:
    print("\n[ERROR] No results!")
