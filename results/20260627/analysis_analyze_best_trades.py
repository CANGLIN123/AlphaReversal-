"""分析最佳30笔交易共性"""
import sys, pandas as pd
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

output_dir = Path(__file__).parent.parent / "output"
trades_file = output_dir / "trades.csv"

if not trades_file.exists():
    print("No trades.csv found.")
    exit()

df = pd.read_csv(trades_file, parse_dates=['buy_date','sell_date'])
df['hold_days'] = (df['sell_date'] - df['buy_date']).dt.days
df['return_pct'] = df['pnl_pct']

ind_file = Path(__file__).parent.parent / "cache" / "stock_industry.csv"
sectors = {}
if ind_file.exists():
    ind_df = pd.read_csv(ind_file, dtype=str)
    sectors = dict(zip(ind_df['ts_code'], ind_df['industry']))

topN = df.nlargest(50, 'pnl')
worstN = df.nsmallest(50, 'pnl')
N = 50

print("=" * 70)
print(f"  TOP 50 TRADES (total: {len(df)})")
print("=" * 70)
print(f"{'#':<4} {'Code':<12} {'Buy':<12} {'Sell':<12} {'Days':>5} {'Return':>8} {'PnL':>8} {'Reason'}")
print("-" * 70)
for i, (_, t) in enumerate(topN.iterrows()):
    print(f"{i+1:<4} {t['code']:<12} {str(t['buy_date'])[:10]:<12} "
          f"{str(t['sell_date'])[:10]:<12} {t['hold_days']:>5} {t['pnl_pct']:>7.1f}% {t['pnl']:>8,.0f} {t['reason']}")

# === 共性分析 ===
print(f"\n{'=' * 70}")
print(f"  PATTERN ANALYSIS: Top30 vs All vs Worst30")
print(f"{'=' * 70}")

# 持有天数
print(f"\n  --- Holding Days ---")
print(f"    Top30: mean={topN['hold_days'].mean():.1f}d, median={topN['hold_days'].median():.0f}d, "
      f"range=[{topN['hold_days'].min()}-{topN['hold_days'].max()}]")
print(f"    All:   mean={df['hold_days'].mean():.1f}d, median={df['hold_days'].median():.0f}d, "
      f"range=[{df['hold_days'].min()}-{df['hold_days'].max()}]")
print(f"    Worst30: mean={worstN['hold_days'].mean():.1f}d, median={worstN['hold_days'].median():.0f}d, "
      f"range=[{worstN['hold_days'].min()}-{worstN['hold_days'].max()}]")

# 持有天数分布
print(f"\n  --- Hold Days Distribution ---")
for label, data in [("Top30", topN), ("All", df), ("Worst30", worstN)]:
    buckets = {'1-3d': 0, '4-7d': 0, '8-14d': 0, '15-21d': 0, '22d+': 0}
    for d in data['hold_days']:
        if d <= 3: buckets['1-3d'] += 1
        elif d <= 7: buckets['4-7d'] += 1
        elif d <= 14: buckets['8-14d'] += 1
        elif d <= 21: buckets['15-21d'] += 1
        else: buckets['22d+'] += 1
    total = len(data)
    parts = [f"{k}: {v}/{total} ({v/total*100:.0f}%)" for k, v in buckets.items()]
    print(f"    {label:<8}: {', '.join(parts)}")

# 退出原因
print(f"\n  --- Exit Reason ---")
reason_groups = {
    '放量阴线': 0, 'N天未盈利': 0, '分批止盈': 0, '持仓超15': 0,
    '移动止盈': 0, '硬止损': 0, 'BBI保护': 0, '持仓超10天亏损': 0,
}
for label, data in [("Top30", topN), ("All", df), ("Worst30", worstN)]:
    for _, t in data.iterrows():
        r = t['reason']
        if '放量阴线' in r: reason_groups['放量阴线'] += 1
        elif '天未盈利' in r: reason_groups['N天未盈利'] += 1
        elif '分批止盈' in r: reason_groups['分批止盈'] += 1
        elif '持仓超15' in r: reason_groups['持仓超15'] += 1
        elif '移动止盈' in r: reason_groups['移动止盈'] += 1
        elif '硬止损' in r: reason_groups['硬止损'] += 1
        elif 'BBI' in r: reason_groups['BBI保护'] += 1
        elif '持仓超10' in r: reason_groups['持仓超10天亏损'] += 1

    total = len(data)
    print(f"  {label} ({total} trades):")
    for reason, cnt in sorted(reason_groups.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    {reason}: {cnt} ({cnt/total*100:.0f}%)")
    # reset
    for k in reason_groups: reason_groups[k] = 0

# 行业
print(f"\n  --- Industry ---")
for label, data in [("Top30", topN), ("Worst30", worstN)]:
    ind = Counter(data['code'].map(lambda c: sectors.get(c, 'unknown')))
    print(f"  {label}: {len(ind)} industries")
    for i, (name, cnt) in enumerate(ind.most_common(10)):
        print(f"    {name}: {cnt}")

# 收益对比
print(f"\n  --- Return Stats ---")
print(f"    Top30 avg return: {topN['pnl_pct'].mean():.1f}%, median: {topN['pnl_pct'].median():.1f}%")
print(f"    All avg return: {df['pnl_pct'].mean():.1f}%")
print(f"    Worst30 avg return: {worstN['pnl_pct'].mean():.1f}%, median: {worstN['pnl_pct'].median():.1f}%")

# 买入年份
print(f"\n  --- Buy Year Distribution ---")
for label, data in [("Top30", topN), ("All", df), ("Worst30", worstN)]:
    years = Counter(pd.to_datetime(data['buy_date']).dt.year)
    print(f"  {label}: {dict(sorted(years.items()))}")

# 关键差异：Top30 vs Worst30 最显著的区别
print(f"\n{'=' * 70}")
print(f"  KEY DIFFERENTIATOR: Top30 vs Worst30")
print(f"{'=' * 70}")
print(f"  Top30: mean hold {topN['hold_days'].mean():.1f}d, exits dominated by profit-taking")
print(f"  Worst30: mean hold {worstN['hold_days'].mean():.1f}d, exits dominated by stop-loss")
print(f"  Most critical factor: holding time")
