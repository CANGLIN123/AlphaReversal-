"""深度分析：找出盈利/亏损交易的本质差异"""
import sys, pandas as pd, numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

output_dir = Path(__file__).parent.parent / "output"
trades = pd.read_csv(output_dir / "trades.csv", parse_dates=['buy_date', 'sell_date'])
trades['hold_days'] = (trades['sell_date'] - trades['buy_date']).dt.days
trades['return_pct'] = trades['pnl_pct']

N = len(trades)
win = trades[trades['pnl'] > 0]
lose = trades[trades['pnl'] < 0]

print("=" * 70)
print(f"  深度交易分析  —  {N} 笔交易 (胜{len(win)} / 负{len(lose)})")
print("=" * 70)

# ============================================================
# 1. 持仓天数 vs 盈亏关系
# ============================================================
print(f"\n{'─'*60}")
print(f"  1. 持仓天数分析")
print(f"{'─'*60}")
for label, data in [("全部", trades), ("盈利", win), ("亏损", lose)]:
    print(f"  {label}: 均值={data['hold_days'].mean():.1f}d  "
          f"中位={data['hold_days'].median():.0f}d  "
          f"范围=[{data['hold_days'].min()}-{data['hold_days'].max()}]")

# 持仓天数桶 vs 胜率
print(f"\n  持仓天数 → 胜率:")
buckets = [(1,3), (4,5), (6,8), (9,12), (13,18), (19,25)]
for lo, hi in buckets:
    sub = trades[(trades['hold_days'] >= lo) & (trades['hold_days'] <= hi)]
    if len(sub) > 0:
        wr = (sub['pnl'] > 0).mean()
        avg = sub['return_pct'].mean()
        bar = '#' * int(wr * 30)
        print(f"  [{lo:2d}-{hi:2d}]天: {len(sub):4d}笔  胜率={wr:.1%}  均收益={avg:.1%}  {bar}")

# ============================================================
# 2. 退出原因深度分析
# ============================================================
print(f"\n{'─'*60}")
print(f"  2. 退出原因 × 盈亏交叉分析")
print(f"{'─'*60}")

reason_map = {}
for _, t in trades.iterrows():
    r = t['reason']
    if '硬止损' in r: key = '硬止损(ATR)'
    elif '移动止盈' in r: key = '移动止盈'
    elif '分批止盈' in r: key = '分批止盈'
    elif 'BBI' in r: key = 'BBI保护'
    elif '天亏损' in r or '天未盈利' in r: key = '时间止损'
    elif '持仓超' in r: key = '持仓超时'
    elif '放量阴线' in r: key = '放量阴线'
    elif '熔断' in r: key = '熔断'
    else: key = r[:8]
    reason_map.setdefault(key, []).append(t)

print(f"  {'原因':<16} {'笔数':>5} {'胜率':>7} {'均收益':>8} {'均持仓':>7} {'贡献':>8}")
print(f"  {'─'*50}")
for reason, items in sorted(reason_map.items(), key=lambda x: -len(x[1])):
    sub = pd.DataFrame(items)
    wr = (sub['pnl'] > 0).mean()
    avg_ret = sub['return_pct'].mean()
    avg_hold = sub['hold_days'].mean()
    total_pnl = sub['pnl'].sum()
    print(f"  {reason:<16} {len(sub):>5} {wr:>6.1%} {avg_ret:>7.1%}% {avg_hold:>6.1f}d {total_pnl:>+8,.0f}")

# ============================================================
# 3. 盈亏金额分布（关键！）
# ============================================================
print(f"\n{'─'*60}")
print(f"  3. 盈亏金额分布")
print(f"{'─'*60}")

print(f"\n  盈利交易 (N={len(win)}):")
print(f"    均值: {win['pnl'].mean():,.0f}  中位数: {win['pnl'].median():,.0f}")
print(f"    P25: {win['pnl'].quantile(0.25):,.0f}  P75: {win['pnl'].quantile(0.75):,.0f}")
print(f"    P90: {win['pnl'].quantile(0.90):,.0f}  P95: {win['pnl'].quantile(0.95):,.0f}")

print(f"\n  亏损交易 (N={len(lose)}):")
print(f"    均值: {lose['pnl'].mean():,.0f}  中位数: {lose['pnl'].median():,.0f}")
print(f"    P25: {lose['pnl'].quantile(0.25):,.0f}  P75: {lose['pnl'].quantile(0.75):,.0f}")
print(f"    P10: {lose['pnl'].quantile(0.10):,.0f}  P5: {lose['pnl'].quantile(0.05):,.0f}")

# 盈亏比详细
print(f"\n  盈亏分析:")
win_mean = win['pnl'].mean()
lose_mean = abs(lose['pnl'].mean())
print(f"    平均盈利/平均亏损 = {win_mean:,.0f} / {lose_mean:,.0f} = {win_mean/lose_mean:.2f}")
print(f"    盈利中位/亏损中位 = {win['pnl'].median():,.0f} / {abs(lose['pnl'].median()):,.0f} = {win['pnl'].median()/abs(lose['pnl'].median()):.2f}")

# 亏大钱的占比
big_lose = lose[lose['pnl'] < -3000]
print(f"\n    大亏(<-3000): {len(big_lose)}笔, 占总交易{len(big_lose)/N*100:.1f}%")
print(f"    大亏合计: {big_lose['pnl'].sum():,.0f} (占亏损总额{big_lose['pnl'].sum()/lose['pnl'].sum()*100:.1f}%)")

# ============================================================
# 4. 买入年份表现
# ============================================================
print(f"\n{'─'*60}")
print(f"  4. 按买入年份表现")
print(f"{'─'*60}")
print(f"  {'年份':<8} {'笔数':>5} {'胜率':>7} {'均收益':>8} {'总盈亏':>10} {'均持仓':>7}")
for yr in sorted(trades['buy_date'].dt.year.unique()):
    sub = trades[trades['buy_date'].dt.year == yr]
    wr = (sub['pnl'] > 0).mean()
    avg = sub['return_pct'].mean()
    total = sub['pnl'].sum()
    hold = sub['hold_days'].mean()
    bar = '+' if total > 0 else '-'
    print(f"  {yr:<8} {len(sub):>5} {wr:>6.1%} {avg:>7.1%}% {total:>+10,.0f} {hold:>6.1f}d {bar}")

# ============================================================
# 5. 连续亏损/盈利分析
# ============================================================
print(f"\n{'─'*60}")
print(f"  5. 连续盈亏序列")
print(f"{'─'*60}")

trades_sorted = trades.sort_values('sell_date')
streaks = []
current_win = None
current_len = 0
for _, t in trades_sorted.iterrows():
    is_win = t['pnl'] > 0
    if is_win == current_win:
        current_len += 1
    else:
        if current_win is not None:
            streaks.append(('赢' if current_win else '亏', current_len))
        current_win = is_win
        current_len = 1
if current_win is not None:
    streaks.append(('赢' if current_win else '亏', current_len))

win_streaks = [l for t, l in streaks if t == '赢']
lose_streaks = [l for t, l in streaks if t == '亏']
print(f"  连赢: 均值={np.mean(win_streaks):.1f}笔  最长={max(win_streaks)}笔")
print(f"  连亏: 均值={np.mean(lose_streaks):.1f}笔  最长={max(lose_streaks)}笔")

# ============================================================
# 6. 关键发现：赚钱的交易有什么共性？
# ============================================================
print(f"\n{'─'*60}")
print(f"  6. 赚钱 vs 亏钱 — 关键差异")
print(f"{'─'*60}")

# 持仓天数
print(f"\n  持仓天数: 赚钱={win['hold_days'].mean():.1f}d vs 亏钱={lose['hold_days'].mean():.1f}d")

# 退出原因分布
print(f"\n  赚钱退出原因 TOP5:")
for reason, items in sorted(reason_map.items(), key=lambda x: -(pd.DataFrame(x[1])['pnl'] > 0).mean() if len(x[1]) > 5 else 0):
    sub = pd.DataFrame(items)
    wr = (sub['pnl'] > 0).mean()
    if len(sub) >= 3:
        print(f"    {reason}: 胜率={wr:.1%} ({len(sub)}笔)")

# 亏损退出原因
print(f"\n  亏钱退出原因 TOP5:")
for reason, items in sorted(reason_map.items(), key=lambda x: (pd.DataFrame(x[1])['pnl'] > 0).mean()):
    sub = pd.DataFrame(items)
    wr = (sub['pnl'] > 0).mean()
    if len(sub) >= 3:
        print(f"    {reason}: 胜率={wr:.1%} ({len(sub)}笔)")

# ============================================================
# 7. 尾部风险
# ============================================================
print(f"\n{'─'*60}")
print(f"  7. 尾部风险 (最差20笔)")
print(f"{'─'*60}")
worst20 = trades.nsmallest(20, 'pnl')
for i, (_, t) in enumerate(worst20.iterrows()):
    print(f"  {i+1:2d}. {t['code']}  {str(t['buy_date'])[:10]}→{str(t['sell_date'])[:10]}  "
          f"持{t['hold_days']:2d}d  {t['return_pct']:>+6.1f}%  {t['reason'][:30]}")

print(f"\n  尾部20笔合计: {worst20['pnl'].sum():,.0f}")
print(f"  占全部亏损: {worst20['pnl'].sum()/lose['pnl'].sum()*100:.1f}%")

# ============================================================
# 8. 买入月份季节性
# ============================================================
print(f"\n{'─'*60}")
print(f"  8. 月度表现")
print(f"{'─'*60}")
for m in range(1, 13):
    sub = trades[trades['buy_date'].dt.month == m]
    if len(sub) > 0:
        wr = (sub['pnl'] > 0).mean()
        avg = sub['return_pct'].mean()
        total = sub['pnl'].sum()
        bar = '#' * int(wr * 30)
        print(f"  {m:2d}月: {len(sub):3d}笔  胜率={wr:.1%}  均收益={avg:+.1f}%  总={total:+,.0f}  {bar}")

print(f"\n{'='*70}")
print(f"  分析完成")
print(f"{'='*70}")
