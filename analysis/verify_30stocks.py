"""30只股票优化回测验证"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_engine import init_tushare, load_multi_stock_data, download_index_daily
from factor_engine import compute_all_factors, calc_market_regime, calc_adaptive_params
from backtest_engine import BacktestEngine
from collections import Counter

# 使用缓存中的30只股票
STOCKS = [
    '000001.SZ','000002.SZ','000004.SZ','000006.SZ','000007.SZ',
    '000008.SZ','000009.SZ','000010.SZ','000011.SZ','000012.SZ',
    '000014.SZ','000016.SZ','000017.SZ','000019.SZ','000020.SZ',
    '000021.SZ','000025.SZ','000026.SZ','000027.SZ','000028.SZ',
    '000029.SZ','000030.SZ','000031.SZ','000032.SZ','000034.SZ',
    '000035.SZ','000036.SZ','000037.SZ','000039.SZ','000042.SZ',
]
START, END = '20200101', '20240601'

print("=" * 60)
print("  30 Stock Optimization Verification")
print("=" * 60)

print(f"\n1. Loading data ({START} ~ {END})...")
stock_data = load_multi_stock_data(STOCKS, START, END)

print("2. Computing factors...")
valid = {}
for code, df in stock_data.items():
    try:
        df_f = compute_all_factors(df)
        if len(df_f.dropna(subset=['kdj_j', 'bbi', 'atr14'])) >= 60:
            valid[code] = df_f
    except Exception as e:
        print(f"  Skip {code}: {e}")
print(f"   Valid: {len(valid)}/{len(stock_data)}")

print("3. Index data...")
index_df = download_index_daily('000001.SH', START, END)

print("4. Market regime...")
if not index_df.empty:
    regime = calc_market_regime(index_df)
    print(f"   Regime: {regime['regime']} | Trend: {regime['trend']} | Vol: {regime['vol_percentile']:.1f}%")

print("\n5. Running backtest...")
config = {
    'initial_capital': 1000000,
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
    'max_hold_days': 10,
    'profit_deadline_days': 5,
    'atr_stop_multiplier': 2.0,
    'portfolio_stop_loss': 0.20,
    'j_threshold': 20,
    'min_score': 40,
}
engine = BacktestEngine(config)
stats = engine.run(valid, index_df)

if len(stats) > 0:
    engine.report()

    if engine.trades:
        reasons = Counter(t['reason'] for t in engine.trades)
        print(f"\n{'=' * 60}")
        print("  Sell Reason Distribution")
        print("=" * 60)
        for r, c in reasons.most_common(15):
            pct = c / len(engine.trades) * 100
            bar = '#' * int(pct / 2)
            print(f"  {r:<35} {c:>4} ({pct:>5.1f}%) {bar}")

        # Check new sell signals
        print(f"\n  New Features Check:")
        checks = [
            ('移动止盈', 'Trailing Stop'),
            ('硬止损', 'Hard Stop'),
            ('分批止盈', 'Partial Profit'),
            ('持仓超', 'Max Hold Days'),
        ]
        for cn, en in checks:
            count = sum(1 for t in engine.trades if cn in t['reason'])
            status = 'ACTIVE' if count > 0 else 'N/A (need more data)'
            print(f"    {en}: {count} [{status}]")

    print(f"\n  Done! Final Equity: {stats['equity'].iloc[-1]:,.0f}")
    print(f"  Total Trades: {len(engine.trades)}")
else:
    print("\n[ERROR] No backtest results!")
