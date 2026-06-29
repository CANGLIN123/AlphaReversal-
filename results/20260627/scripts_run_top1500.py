"""全市场优化版回测 — 取市值前1500只（缓存数据，不调API）"""
import sys, time, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from factor_engine import compute_all_factors, calc_market_regime, calc_adaptive_params
from backtest_engine import BacktestEngine
from data_engine import filter_by_liquidity

cache_dir = Path(__file__).parent.parent / "cache"
START, END = '20200101', '20251201'
DATE_TAG = f'{START}_{END}_qfq'

print("=" * 65)
print("  全市场优化版回测 — Top1500 (cache-only)")
print("=" * 65)

# 1. 加载缓存数据（全部股票，后续按市值筛选）
print("[1/5] Loading from cache...")
all_stock_data = {}
t0 = time.time()
cache_files = sorted(cache_dir.glob(f"*_{DATE_TAG}.csv"))
total = len([f for f in cache_files if not f.name.startswith('IDX_')])
loaded = 0
for f in cache_files:
    if f.name.startswith('IDX_'):
        continue
    code = f.name[:9]
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if not df.empty and 'close' in df.columns and len(df) >= 120:
            # 取最近一段数据的日均成交额做初步过滤
            recent = df.tail(60)
            if 'amount' in df.columns:
                avg_amt = recent['amount'].mean()
                if avg_amt < 2_000:  # 日均成交<200万（千元→2000万），跳过
                    continue
            all_stock_data[code] = df
            loaded += 1
    except Exception:
        pass
    if loaded % 500 == 0 and loaded > 0:
        print(f"      {loaded} stocks loaded ({time.time()-t0:.0f}s)")

print(f"      Loaded: {len(all_stock_data)} stocks ({time.time()-t0:.0f}s)")

# 2. 按市值排序取前1500（用最近一日的amount做代理）
print("[2/5] Selecting top 1500 by avg turnover...")
stock_scores = []
for code, df in all_stock_data.items():
    if 'amount' in df.columns:
        avg_turnover = df.tail(120)['amount'].mean()
    else:
        avg_turnover = 0
    stock_scores.append((code, avg_turnover))
stock_scores.sort(key=lambda x: -x[1])
top_codes = [c for c, _ in stock_scores[:1500]]
stock_data = {c: all_stock_data[c] for c in top_codes if c in all_stock_data}
print(f"      Selected: {len(stock_data)} stocks")

# 3. 因子计算
print("[3/5] Computing factors...")
valid = {}
t0 = time.time()
for i, (code, df) in enumerate(stock_data.items()):
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j', 'bbi', 'atr14'])) >= 60:
            valid[code] = df_f
    except Exception:
        pass
    if (i + 1) % 300 == 0:
        print(f"      {i+1}/{len(stock_data)} ({time.time()-t0:.0f}s)  valid:{len(valid)}")
print(f"      Valid: {len(valid)} ({time.time()-t0:.0f}s)")

# 4. 指数 + 行业
print("[4/5] Index & Industry...")
idx_file = cache_dir / f"IDX_000001.SH_{START}_{END}.csv"
if idx_file.exists():
    index_df = pd.read_csv(idx_file, index_col=0, parse_dates=True)
    print(f"      Index: {len(index_df)} days")
    regime = calc_market_regime(index_df)
    print(f"      Regime: {regime['regime']} | Trend: {regime['trend']} | Vol: {regime['vol_percentile']:.0f}%")
else:
    index_df = None
    print("      No index!")

ind_file = cache_dir / "stock_industry.csv"
stock_sectors = {}
if ind_file.exists():
    ind_df = pd.read_csv(ind_file, dtype=str)
    stock_sectors = dict(zip(ind_df['ts_code'], ind_df['industry']))
    print(f"      Industry: {len(stock_sectors)} stocks, {ind_df['industry'].nunique()} industries")
else:
    print("      No industry file")

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
    'trailing_activation': 0.02,
    'partial_profit_taking': True,
    'max_hold_days': 15,
    'profit_deadline_days': 7,
    'atr_stop_multiplier': 2.5,
    'portfolio_stop_loss': 0.20,
    'j_threshold': 20,
    'j_60d_min_threshold': 15,       # 60日内J曾低于15（曾极度超卖）
    'min_score': 45,                 # 放宽最低得分
    'score_weights': {
        'oversold': 0.18,            # 超卖深度 (J_60d_min)
        'volume_shrink': 0.16,       # 缩量得分
        'small_candle': 0.14,        # 小K线企稳
        'prior_surge': 0.12,         # 前期放量异动
        'reversal': 0.12,            # 反转确认 (拐头+缩量+小K线)
        'bbi_trend': 0.10,           # BBI趋势
        'ma60_proximity': 0.07,      # MA60贴近
        'hot_industry': 0.05,        # 行业热度
        'amihud': 0.03,              # 弹性因子
        'quality': 0.03,             # 质量因子
    },
}

engine = BacktestEngine(config)
stats = engine.run(valid, index_df, stock_sectors=stock_sectors)

if len(stats) > 0:
    engine.report()
    engine.export()

    if engine.trades:
        cats = {
            'Hard stop': 0, 'Trailing stop': 0, 'Partial profit': 0,
            'BBI protect': 0, 'N-day loss': 0, 'Max hold': 0,
            'Volume dump': 0, 'Circuit breaker': 0,
        }
        for t in engine.trades:
            r = t['reason']
            if '硬止损' in r: cats['Hard stop'] += 1
            elif '移动止盈' in r: cats['Trailing stop'] += 1
            elif '分批止盈' in r: cats['Partial profit'] += 1
            elif 'BBI' in r: cats['BBI protect'] += 1
            elif '天未盈利' in r or '天亏损' in r: cats['N-day loss'] += 1
            elif '持仓超' in r: cats['Max hold'] += 1
            elif '放量阴线' in r: cats['Volume dump'] += 1
            elif '熔断' in r: cats['Circuit breaker'] += 1

        total = len(engine.trades)
        print(f"\n{'=' * 65}")
        print(f"  SELL REASON BREAKDOWN ({total} trades)")
        print("=" * 65)
        for cat, c in sorted(cats.items(), key=lambda x: -x[1]):
            if c > 0:
                bar = '#' * int(c / max(1, total) * 50)
                print(f"  {cat:<18} {c:>5} ({c/max(1,total)*100:>5.1f}%) {bar}")

    print(f"\n  Final Equity: {stats['equity'].iloc[-1]:,.0f}")
else:
    print("\n[ERROR] No results!")
