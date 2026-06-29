"""快速验证 — 优化后回测系统端到端测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_engine import init_tushare, load_multi_stock_data, download_index_daily
from factor_engine import compute_all_factors, calc_market_regime, calc_adaptive_params
from backtest_engine import BacktestEngine
from collections import Counter

print("=" * 60)
print("  优化后回测系统 — 端到端验证")
print("=" * 60)

# 5只股票快速测试
test_stocks = ['000001.SZ', '000002.SZ', '600036.SH', '600519.SH', '601318.SH']

START, END = '20200101', '20240601'

print(f"\n1. 加载数据 ({START} ~ {END})...")
stock_data = load_multi_stock_data(test_stocks, START, END)

print("2. 计算因子...")
valid = {}
for code, df in stock_data.items():
    df_f = compute_all_factors(df)
    if len(df_f.dropna(subset=['kdj_j', 'bbi', 'atr14'])) >= 60:
        valid[code] = df_f
print(f"   有效股票: {len(valid)}/{len(stock_data)}")

print("3. 下载指数...")
index_df = download_index_daily('000001.SH', START, END)

print("4. 市场状态识别...")
if not index_df.empty:
    regime = calc_market_regime(index_df)
    print(f"   状态: {regime['regime']}")
    print(f"   趋势: {regime['trend']}, 波动分位: {regime['vol_percentile']:.1f}%")
    adaptive = calc_adaptive_params(regime)
    print(f"   自适应参数: J阈值={adaptive['j_threshold']}, 仓位={adaptive['max_positions']}")

print("\n5. 运行回测 (成交模型=next_open, 分批建仓=启用, 移动止盈=启用)...")
config = {
    'initial_capital': 500000,
    'max_positions': 3,
    'max_positions_low': 2,
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
}
engine = BacktestEngine(config)
stats = engine.run(valid, index_df)

if len(stats) > 0:
    metrics = engine.report()

    print(f"\n{'=' * 60}")
    print("  卖出原因分布")
    print("=" * 60)
    if engine.trades:
        reasons = Counter(t['reason'] for t in engine.trades)
        for r, c in reasons.most_common():
            pct = c / len(engine.trades) * 100
            bar = '#' * int(pct / 2)
            print(f"  {r:<30} {c:>4} ({pct:>5.1f}%) {bar}")

        # 统计是否有新的卖出原因
        new_reasons = ['移动止盈', '硬止损', '分批止盈', '持仓超']
        print(f"\n  新卖出信号出现情况:")
        for nr in new_reasons:
            count = sum(1 for t in engine.trades if nr in t['reason'])
            status = 'YES' if count > 0 else 'not triggered'
            print(f"    {nr}: {count} trades [{status}]")
    else:
        print("  无交易记录")

    print(f"\n  回测完成! 净值: {stats['equity'].iloc[-1]:,.0f}")
else:
    print("\n[错误] 回测无结果!")
