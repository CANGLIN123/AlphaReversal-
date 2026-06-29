"""分析参考买点的共性特征（纯缓存，不调API）"""
import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from factor_engine import compute_all_factors

cache_dir = Path(__file__).parent.parent / "cache"

# 参考买点
buy_points = [
    ('688799.SH', '华纳药厂', '20250513'),
    ('600366.SH', '宁波韵升', '20250804'),
    ('600601.SH', '方正科技', '20250808'),
    ('688321.SH', '微芯生物', '20250620'),
    ('002961.SZ', '瑞达期货', '20251106'),
    ('000547.SZ', '航天发展', '20251112'),
    ('001316.SZ', '润贝航科', '20260116'),
    ('000700.SZ', '模塑科技', '20260601'),
    ('003036.SZ', '泰坦股份', '20260408'),
    ('002940.SZ', '昂利康',   '20250711'),
    ('603083.SH', '剑桥科技', '20250908'),
]

print("=" * 80)
print("  参考买点共性分析 (纯缓存)")
print("=" * 80)

results = []
skipped = []

for code, name, buy_date_str in buy_points:
    buy_dt = pd.Timestamp(buy_date_str)

    # 找缓存文件
    cache_files = list(cache_dir.glob(f"{code}_20200101_*_qfq.csv"))
    if not cache_files:
        skipped.append((name, "无缓存"))
        print(f"  X {code} {name}: 无缓存文件")
        continue

    cache_file = cache_files[0]
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

    if df.empty or len(df) < 120:
        skipped.append((name, "数据不足"))
        continue

    # 检查买点是否在数据范围内
    if buy_dt not in df.index:
        last_date = df.index[-1]
        if last_date < buy_dt:
            print(f"  ~ {code} {name}: 缓存截止{last_date.strftime('%Y-%m-%d')}, 买点{buy_dt.strftime('%Y-%m-%d')}超出范围")
            # 使用最后一天作为近似
            buy_dt_actual = last_date
        else:
            nearby = df.index[df.index <= buy_dt]
            buy_dt_actual = nearby[-1] if len(nearby) > 0 else df.index[-1]
    else:
        buy_dt_actual = buy_dt

    df_f = compute_all_factors(df)
    buy_idx = df_f.index.get_loc(buy_dt_actual)
    row = df_f.iloc[buy_idx]

    # === 提取所有关键指标 ===
    kdj_j = row.get('kdj_j', np.nan)
    kdj_j_prev = row.get('kdj_j_prev', np.nan)
    kdj_k = row.get('kdj_k', np.nan)
    kdj_d = row.get('kdj_d', np.nan)
    bbi = row.get('bbi', np.nan)
    ma5 = row.get('ma5', np.nan)
    ma20 = row.get('ma20', np.nan)
    ma60 = row.get('ma60', np.nan)
    close = row['close']
    atr14 = row.get('atr14', np.nan)
    vol_pattern = row.get('vol_pattern_valid', 0)
    vol_pattern_score = row.get('vol_pattern_score', 0)
    volume_ratio = row.get('volume_ratio', np.nan)
    ret_1d = row.get('returns', np.nan)
    ret_5d = row.get('ret_5d', np.nan)
    ret_20d = row.get('ret_20d', np.nan)
    adx = row.get('adx', np.nan)
    di_diff = row.get('di_diff', np.nan)
    amihud = row.get('amihud', np.nan)

    # 前60日统计
    j_60d = df_f['kdj_j'].iloc[max(0,buy_idx-60):buy_idx+1].dropna()
    j_min_60d = j_60d.min()
    j_low_days = (j_60d < 20).sum()
    j_rank = (j_60d < kdj_j).mean() if not np.isnan(kdj_j) else np.nan  # J值分位数

    # 前20日统计
    vol_20d = df_f['volume'].iloc[max(0,buy_idx-20):buy_idx+1]
    vol_ma20d_avg = vol_20d.mean()
    vol_today = row['volume']
    vol_vs_avg = vol_today / vol_ma20d_avg if vol_ma20d_avg > 0 else np.nan

    # 价格相对位置
    price_vs_ma5 = (close / ma5 - 1) * 100 if not np.isnan(ma5) else np.nan
    price_vs_ma20 = (close / ma20 - 1) * 100 if not np.isnan(ma20) else np.nan
    price_vs_ma60 = (close / ma60 - 1) * 100 if not np.isnan(ma60) else np.nan
    price_vs_bbi = (close / bbi - 1) * 100 if not np.isnan(bbi) else np.nan
    bbi_vs_ma60 = (bbi / ma60 - 1) * 100 if not (np.isnan(bbi) or np.isnan(ma60)) else np.nan
    ma5_vs_ma20 = (ma5 / ma20 - 1) * 100 if not (np.isnan(ma5) or np.isnan(ma20)) else np.nan

    # J值拐头
    j_turning = (not np.isnan(kdj_j) and not np.isnan(kdj_j_prev) and kdj_j > kdj_j_prev)
    # J值拐头天数
    j_rising_days = 0
    for ii in range(buy_idx, max(0,buy_idx-10), -1):
        if ii > 0 and not np.isnan(df_f['kdj_j'].iloc[ii]) and not np.isnan(df_f['kdj_j'].iloc[ii-1]):
            if df_f['kdj_j'].iloc[ii] > df_f['kdj_j'].iloc[ii-1]:
                j_rising_days += 1
            else:
                break

    # 买点后表现
    future_ret_5 = np.nan
    future_ret_10 = np.nan
    future_ret_20 = np.nan
    if buy_idx + 5 < len(df_f):
        future_ret_5 = (df_f['close'].iloc[buy_idx+5] / close - 1) * 100
    if buy_idx + 10 < len(df_f):
        future_ret_10 = (df_f['close'].iloc[buy_idx+10] / close - 1) * 100
    if buy_idx + 20 < len(df_f):
        future_ret_20 = (df_f['close'].iloc[buy_idx+20] / close - 1) * 100

    result = {
        'name': name, 'code': code,
        'buy_date': buy_dt_actual.strftime('%Y-%m-%d'),
        'close': round(close, 2),
        'J': round(kdj_j, 1), 'J_prev': round(kdj_j_prev, 1),
        'J_turning': j_turning, 'J_rising_days': j_rising_days,
        'J_60d_min': round(j_min_60d, 1), 'J<20_days': int(j_low_days),
        'J_rank': f"{j_rank*100:.0f}%" if not np.isnan(j_rank) else 'NaN',
        'VolRatio': round(volume_ratio, 2) if not np.isnan(volume_ratio) else 'NaN',
        'VolVSAvg': round(vol_vs_avg, 2) if not np.isnan(vol_vs_avg) else 'NaN',
        'VolPattern': vol_pattern, 'VolScore': round(vol_pattern_score, 1),
        'Ret_1d': f"{ret_1d*100:+.1f}%" if not np.isnan(ret_1d) else 'NaN',
        'Ret_5d': f"{ret_5d*100:+.1f}%" if not np.isnan(ret_5d) else 'NaN',
        'Ret_20d': f"{ret_20d*100:+.1f}%" if not np.isnan(ret_20d) else 'NaN',
        'P_vs_MA5': f"{price_vs_ma5:+.1f}%" if not np.isnan(price_vs_ma5) else 'NaN',
        'P_vs_MA20': f"{price_vs_ma20:+.1f}%" if not np.isnan(price_vs_ma20) else 'NaN',
        'P_vs_MA60': f"{price_vs_ma60:+.1f}%" if not np.isnan(price_vs_ma60) else 'NaN',
        'BBI_vs_MA60': f"{bbi_vs_ma60:+.1f}%" if not np.isnan(bbi_vs_ma60) else 'NaN',
        'MA5_vs_MA20': f"{ma5_vs_ma20:+.1f}%" if not np.isnan(ma5_vs_ma20) else 'NaN',
        'ATR14': round(atr14, 3) if not np.isnan(atr14) else 'NaN',
        'ATR%': f"{atr14/close*100:.2f}%" if not (np.isnan(atr14) or close == 0) else 'NaN',
        'ADX': round(adx, 1) if not np.isnan(adx) else 'NaN',
        'DI_diff': round(di_diff, 1) if not np.isnan(di_diff) else 'NaN',
        'Fut_5d': f"{future_ret_5:+.1f}%" if not np.isnan(future_ret_5) else 'N/A',
        'Fut_10d': f"{future_ret_10:+.1f}%" if not np.isnan(future_ret_10) else 'N/A',
        'Fut_20d': f"{future_ret_20:+.1f}%" if not np.isnan(future_ret_20) else 'N/A',
    }
    results.append(result)
    print(f"  V {name:<8} {buy_dt_actual.strftime('%Y-%m-%d')}  J={kdj_j:.1f}  "
          f"J拐头={'V' if j_turning else 'X'} 量比={volume_ratio:.2f}  "
          f"vsMA5={price_vs_ma5:+.1f}%  后10d={future_ret_10:+.1f}%")

# ============================================================
# 详细表格
# ============================================================
print(f"\n{'=' * 130}")
print("  详细指标对比（已分析 {}/{} 只）".format(len(results), len(buy_points)))
print(f"{'=' * 130}")

# 简洁版
header = (f"{'股票':<10}{'日期':<12}{'J':>6}{'J拐头':>5}{'J低':>6}{'J<20':>5}"
          f"{'J分位':>6}{'量比':>6}{'V形':>4}"
          f"{'1d':>7}{'5d':>7}{'20d':>7}{'vMA5':>7}{'vMA20':>7}{'vMA60':>7}"
          f"{'BvM60':>7}{'ATR%':>7}{'ADX':>6}{'后5d':>7}{'后10d':>7}{'后20d':>7}")
print(header)
print("-" * 130)

for r in results:
    line = (f"{r['name']:<10}{r['buy_date']:<12}"
            f"{r['J']:>6}{'V' if r['J_turning'] else 'X':>5}"
            f"{r['J_60d_min']:>6}{r['J<20_days']:>5}{str(r['J_rank']):>6}"
            f"{str(r['VolRatio']):>6}{'V' if r['VolPattern'] else 'X':>4}"
            f"{str(r['Ret_1d']):>7}{str(r['Ret_5d']):>7}{str(r['Ret_20d']):>7}"
            f"{str(r['P_vs_MA5']):>7}{str(r['P_vs_MA20']):>7}{str(r['P_vs_MA60']):>7}"
            f"{str(r['BBI_vs_MA60']):>7}{str(r['ATR%']):>7}{str(r['ADX']):>6}"
            f"{str(r['Fut_5d']):>7}{str(r['Fut_10d']):>7}{str(r['Fut_20d']):>7}")
    print(line)

# ============================================================
# 共性统计
# ============================================================
print(f"\n{'=' * 80}")
print(f"  共性统计 (n={len(results)})")
print(f"{'=' * 80}")

n = len(results)
# Numeric arrays
j_vals = np.array([r['J'] for r in results if isinstance(r['J'], (int,float))])
vol_ratios = np.array([r['VolRatio'] for r in results if isinstance(r['VolRatio'], (int,float))])
j_min = np.array([r['J_60d_min'] for r in results if isinstance(r['J_60d_min'], (int,float))])

# Boolean signals
j_turn = sum(1 for r in results if r['J_turning'])
vol_pat = sum(1 for r in results if r['VolPattern'])
j_low = sum(1 for r in results if isinstance(r['J'], (int,float)) and r['J'] < 20)
j_below_30 = sum(1 for r in results if isinstance(r['J'], (int,float)) and r['J'] < 30)

# Price vs MA
def parse_pct(s):
    try: return float(s.replace('%', ''))
    except: return np.nan

p_ma5 = np.array([parse_pct(r['P_vs_MA5']) for r in results])
p_ma20 = np.array([parse_pct(r['P_vs_MA20']) for r in results])
p_ma60 = np.array([parse_pct(r['P_vs_MA60']) for r in results])
b_ma60 = np.array([parse_pct(r['BBI_vs_MA60']) for r in results])

above_ma5 = (p_ma5 > 0).sum()
above_ma20 = (p_ma20 > 0).sum()
above_ma60 = (p_ma60 > 0).sum()
bbi_abv_ma60 = (b_ma60 > 0).sum()

# 买点后收益
fut5 = np.array([parse_pct(r['Fut_5d']) for r in results if r['Fut_5d'] != 'N/A'])
fut10 = np.array([parse_pct(r['Fut_10d']) for r in results if r['Fut_10d'] != 'N/A'])
fut20 = np.array([parse_pct(r['Fut_20d']) for r in results if r['Fut_20d'] != 'N/A'])

print(f"\n  1. KDJ指标:")
print(f"     J值: 均值={np.mean(j_vals):.1f}  中位={np.median(j_vals):.1f}  "
      f"范围=[{np.min(j_vals):.1f}, {np.max(j_vals):.1f}]")
print(f"     J<20(超卖): {j_low}/{n} ({j_low/n*100:.0f}%)")
print(f"     J<30: {j_below_30}/{n} ({j_below_30/n*100:.0f}%)")
print(f"     J拐头向上: {j_turn}/{n} ({j_turn/n*100:.0f}%)")
print(f"     J 60日最低: 均值={np.mean(j_min):.1f}")

print(f"\n  2. 量价形态:")
print(f"     量比(volume/MA20): 均值={np.mean(vol_ratios):.2f}  中位={np.median(vol_ratios):.2f}")
print(f"     量比<0.8(缩量): {sum(vol_ratios < 0.8)}/{n} ({sum(vol_ratios < 0.8)/n*100:.0f}%)")
print(f"     放量+缩量回踩形态: {vol_pat}/{n} ({vol_pat/n*100:.0f}%)")

print(f"\n  3. 趋势与均线:")
print(f"     价格>MA5:  {above_ma5}/{n} ({above_ma5/n*100:.0f}%)  均值偏离={np.nanmean(p_ma5):+.1f}%")
print(f"     价格>MA20: {above_ma20}/{n} ({above_ma20/n*100:.0f}%)  均值偏离={np.nanmean(p_ma20):+.1f}%")
print(f"     价格>MA60: {above_ma60}/{n} ({above_ma60/n*100:.0f}%)  均值偏离={np.nanmean(p_ma60):+.1f}%")
print(f"     BBI>MA60: {bbi_abv_ma60}/{n} ({bbi_abv_ma60/n*100:.0f}%)  均值偏离={np.nanmean(b_ma60):+.1f}%")

print(f"\n  4. 买点后表现:")
if len(fut5) > 0:
    print(f"     5日后: 均值={np.mean(fut5):+.1f}% 中位={np.median(fut5):+.1f}%  "
          f"胜率={sum(fut5>0)}/{len(fut5)}")
if len(fut10) > 0:
    print(f"     10日后: 均值={np.mean(fut10):+.1f}% 中位={np.median(fut10):+.1f}%  "
          f"胜率={sum(fut10>0)}/{len(fut10)}")
if len(fut20) > 0:
    print(f"     20日后: 均值={np.mean(fut20):+.1f}% 中位={np.median(fut20):+.1f}%  "
          f"胜率={sum(fut20>0)}/{len(fut20)}")

# ============================================================
# 最关键的发现：信号交集
# ============================================================
print(f"\n{'=' * 80}")
print(f"  信号组合分析 — 找出最强共性")
print(f"{'=' * 80}")

signals_check = {
    'J<30 (相对超卖)': lambda r: isinstance(r['J'], (int,float)) and r['J'] < 30,
    'J拐头向上': lambda r: r['J_turning'],
    '量比<0.8 (缩量)': lambda r: isinstance(r['VolRatio'], (int,float)) and r['VolRatio'] < 0.8,
    '量价形态完整': lambda r: r['VolPattern'],
    '价格>MA5': lambda r: parse_pct(r['P_vs_MA5']) > 0,
    '价格>MA20': lambda r: parse_pct(r['P_vs_MA20']) > 0,
    '价格>MA60 (牛市背景)': lambda r: parse_pct(r['P_vs_MA60']) > 0,
    'BBI>MA60 (趋势确认)': lambda r: parse_pct(r['BBI_vs_MA60']) > 0,
    'J从极低回升 (min<15)': lambda r: isinstance(r['J_60d_min'], (int,float)) and r['J_60d_min'] < 15,
}

for desc, fn in signals_check.items():
    cnt = sum(1 for r in results if fn(r))
    bar = '#' * int(cnt / n * 40)
    print(f"  [{cnt}/{n}] {desc:<35} {bar}")

# 多条件交集
print(f"\n  三级信号组合:")
tier1 = ['J<30 (相对超卖)', 'J拐头向上', '量比<0.8 (缩量)']
tier2 = tier1 + ['价格>MA5']
tier3 = tier2 + ['BBI>MA60 (趋势确认)']

c1 = sum(1 for r in results if all(fn(r) for fn in [signals_check[s] for s in tier1]))
c2 = sum(1 for r in results if all(fn(r) for fn in [signals_check[s] for s in tier2]))
c3 = sum(1 for r in results if all(fn(r) for fn in [signals_check[s] for s in tier3]))
print(f"  L1 [J<30+拐头+缩量]:              {c1}/{n}")
print(f"  L2 [L1+价格>MA5]:                 {c2}/{n}")
print(f"  L3 [L2+BBI>MA60]:                 {c3}/{n}")

# 哪个组合后收益最高？
print(f"\n  最强组合 vs 买点后收益:")
combo = [('J<30+拐头+缩量', ['J<30 (相对超卖)', 'J拐头向上', '量比<0.8 (缩量)']),
         ('J<30+拐头+缩量+>MA5', ['J<30 (相对超卖)', 'J拐头向上', '量比<0.8 (缩量)', '价格>MA5']),
         ('J<30+拐头+缩量+>MA5+B>M60', ['J<30 (相对超卖)', 'J拐头向上', '量比<0.8 (缩量)', '价格>MA5', 'BBI>MA60 (趋势确认)'])]

for name, checks in combo:
    matched = [r for r in results if all(fn(r) for fn in [signals_check[s] for s in checks])]
    if matched:
        m_fut10 = np.mean([parse_pct(r['Fut_10d']) for r in matched if r['Fut_10d'] != 'N/A'])
        m_fut20 = np.mean([parse_pct(r['Fut_20d']) for r in matched if r['Fut_20d'] != 'N/A'])
        print(f"  {name}: {len(matched)}只, 后10d均={m_fut10:+.1f}%, 后20d均={m_fut20:+.1f}%")

print(f"\n{'=' * 80}")
print(f"  完成")
print(f"{'=' * 80}")
