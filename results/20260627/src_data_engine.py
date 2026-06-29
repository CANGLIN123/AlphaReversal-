"""
===========================================================
数据引擎 — tushare 数据获取 + 清洗 + 缓存
===========================================================
职责：
  1. 通过 tushare pro 获取 A 股日线数据（OHLCV）
  2. 获取基本面数据（PE/PB/ROE/市值）
  3. 获取指数数据（上证指数用于择时）
  4. 数据清洗：停牌处理 / 复权 / 除权 / ST 过滤
  5. 本地 CSV 缓存，避免重复请求

设计原则：
  - 所有数据以 pandas DataFrame 形式返回，index 为 DatetimeIndex
  - 缓存优先：先查本地 → 没有再调 API → 自动存缓存
  - 容错设计：API 失败时 fallback 到缓存或提示

面试考点：
  Q: 为什么要做复权处理？
  A: 分红送股会导致价格跳空，不复权=虚假涨跌信号。
     后复权保留历史真实成交价，前复权让最新价=实际价。
     回测中一般用前复权（保证最新价格直观）。
     这里用 tushare 的 qfq（前复权）接口。

  Q: 为什么停牌股票要单独处理？
  A: 停牌期间 price 不变但 volume=0，如果不过滤：
     ① 计算收益率时出现假的 0% 收益
     ② 均线/波动率被假数据污染
     ③ 可能在停牌日"买入"——现实中不可能
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

# ============================================================
# 0. 配置
# ============================================================
# tushare token（从 config.py 读取或直接设置）
TUSHARE_TOKEN = "ec50312699c5183319c286098fe6ead9a3c7a86f185a5ffb5c62c058"

# 缓存目录
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# tushare pro 全局连接
_pro_api = None


def init_tushare(token: str = None):
    """
    初始化 tushare pro 连接

    参数:
        token: tushare token，不传则使用默认值

    为什么延迟初始化而不是在模块顶部直接连？
      ① 允许运行时切换 token（比如自己的 vs 学校的）
      ② 导入模块时不会因为网络问题报错
      ③ 面试时可以说出"延迟初始化"这个设计模式
    """
    global _pro_api
    import tushare as ts
    _token = token or TUSHARE_TOKEN
    ts.set_token(_token)
    _pro_api = ts.pro_api()
    print(f"[tushare] 初始化完成，token: {_token[:8]}...")
    return _pro_api


def get_pro():
    """获取 tushare pro 连接，未初始化则自动初始化"""
    global _pro_api
    if _pro_api is None:
        init_tushare()
    return _pro_api


# ============================================================
# 1. 股票池获取
# ============================================================

def get_stock_pool(method: str = "hs300", date: str = None) -> list:
    """
    获取候选股票池

    参数:
        method: "hs300" | "zz500" | "top1500" | "all" | "all_filtered"
        date: 指定日期的成分股（如 '20240101'）

    返回:
        list[str]: 股票代码列表，如 ['000001.SZ', '000002.SZ', ...]

    面试考点：
      Q: 为什么选沪深300而不是全市场？
      A: ① 流动性好，滑点小（大市值的 bid-ask spread 窄）
         ② 财务数据可靠（小市值公司财报质量参差不齐）
         ③ 避免幸存者偏差（沪深300成分股动态调整）
         ④ 机构实际可交易（基金经理有市值约束）
    """
    pro = get_pro()

    # 统一的缓存逻辑：所有池都缓存到 CSV，避免重复调用受限于 1次/小时的接口
    date_tag = date or datetime.now().strftime('%Y%m%d')
    cache_file = CACHE_DIR / f"stock_pool_{method}_{date_tag}.csv"

    if cache_file.exists():
        stocks = pd.read_csv(cache_file, dtype=str)['ts_code'].tolist()
        print(f"[股票池] {method}: {len(stocks)} 只股票 (来自缓存)")
        return stocks

    if method == "hs300":
        # 沪深300成分股
        df = pro.index_weight(
            index_code='000300.SH',
            trade_date=date_tag
        )
        stocks = [f"{c[:6]}.{'SH' if c[:6].startswith('6') else 'SZ'}"
                  for c in df['con_code'].tolist()]

    elif method == "zz500":
        df = pro.index_weight(
            index_code='000905.SH',
            trade_date=date_tag
        )
        stocks = [f"{c[:6]}.{'SH' if c[:6].startswith('6') else 'SZ'}"
                  for c in df['con_code'].tolist()]

    elif method == "top1500":
        # 按市值排序取前1500（需要基本面数据）
        df = pro.daily_basic(
            trade_date=date_tag,
            fields='ts_code,total_mv'
        )
        df = df.dropna(subset=['total_mv'])
        df = df.sort_values('total_mv', ascending=False).head(1500)
        stocks = df['ts_code'].tolist()

    elif method in ("all", "all_filtered"):
        # 全A股（上市状态=L）
        df = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,name,list_date'
        )
        if method == "all_filtered":
            # 过滤规则1: 去掉ST/*ST（名称含ST的）
            is_st = df['name'].str.contains('ST', na=False)
            df = df[~is_st]
            print(f"  [过滤] 剔除ST: {is_st.sum()} 只")

            # 过滤规则2: 去掉上市不足60天的次新股
            if 'list_date' in df.columns:
                list_dates = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
                cutoff = pd.Timestamp(date_tag) - pd.Timedelta(days=60)
                is_new = list_dates > cutoff
                df = df[~is_new]
                print(f"  [过滤] 剔除次新股(<60天): {is_new.sum()} 只")

            # 过滤规则3: 去掉北交所（代码以8开头）
            is_bj = df['ts_code'].str[:1] == '8'
            df = df[~is_bj]
            print(f"  [过滤] 剔除北交所: {is_bj.sum()} 只")

            # 过滤规则4: 去掉科创板68开头的（波动太大，涨跌停20%）
            # is_kcb = df['ts_code'].str.startswith('688')
            # df = df[~is_kcb]

            print(f"  [过滤后] 剩余: {len(df)} 只")

        stocks = df['ts_code'].tolist()

    else:
        raise ValueError(f"不支持的股票池类型: {method}")

    # 保存缓存
    pd.DataFrame({'ts_code': stocks}).to_csv(cache_file, index=False)
    print(f"[股票池] {method}: {len(stocks)} 只股票")
    return stocks


# ============================================================
# 1.5 动态流动性过滤（每只股票检查日均成交额）
# ============================================================

def filter_by_liquidity(stock_data: dict, min_daily_amount: float = 20_000,
                        lookback_days: int = 60) -> dict:
    """
    过滤流动性不足的股票

    参数:
        stock_data: {code: DataFrame} 字典
        min_daily_amount: 最低日均成交额（默认2000万，tushare amount=千元所以=20000）
        lookback_days: 回溯天数

    返回:
        dict: 过滤后的股票数据

    逻辑：
      小市值股票日均成交额可能<500万，大单根本进不去。
      2000万≈散户策略的安全线（单日成交5%=100万仓位上限）

    注意：tushare amount字段单位为千元，所以2000万=20000千元
    """
    filtered = {}
    removed = 0
    for code, df in stock_data.items():
        if len(df) < lookback_days:
            removed += 1
            continue
        recent = df.tail(lookback_days)
        if 'amount' not in recent.columns:
            filtered[code] = df  # 无成交额数据则保留
            continue
        avg_amount = recent['amount'].mean()
        if avg_amount >= min_daily_amount:
            filtered[code] = df
        else:
            removed += 1

    if removed > 0:
        print(f"  [流动性过滤] 剔除 {removed} 只 (日均成交额<{min_daily_amount/10:.0f}万)")
    print(f"  [流动性过滤后] 剩余 {len(filtered)} 只")
    return filtered


# ============================================================
# 1.6 涨跌停检测
# ============================================================

def is_price_limit_day(df: pd.DataFrame, date_idx: int = -1) -> Tuple[bool, bool]:
    """
    判断某一天是否为涨跌停日

    A股涨跌停规则:
      - 主板(60/00开头): ±10%
      - 创业板(30开头): ±20%
      - 科创板(68开头): ±20%
      - ST股票: ±5%
      - 上市首日: 无限制(±44%)

    返回:
        (is_limit_up, is_limit_down)
    """
    if len(df) < 2 or abs(date_idx) > len(df):
        return False, False

    row = df.iloc[date_idx]
    prev = df.iloc[date_idx - 1]

    if prev['close'] <= 0 or row['close'] <= 0:
        return False, False

    ret = (row['close'] - prev['close']) / prev['close']
    code = df.index.name if hasattr(df.index, 'name') else ''

    # 简化版：统一用 ±9.5% 作为涨跌停阈值（留缓冲）
    # ST 用 ±4.5%
    if 'ST' in str(code).upper():
        limit_up = ret > 0.045 and row['high'] == row['low']  # 一字板
        limit_down = ret < -0.045 and row['high'] == row['low']
    else:
        limit_up = ret > 0.095 and abs(row['high'] - row['close']) / row['close'] < 0.001
        limit_down = ret < -0.095 and abs(row['low'] - row['close']) / row['close'] < 0.001

    return limit_up, limit_down


# ============================================================
# 1.7 市场宽度计算
# ============================================================

def calc_market_breadth(stock_data: dict, date) -> dict:
    """
    计算全市场宽度指标

    参数:
        stock_data: {code: DataFrame} 字典
        date: 目标日期

    返回:
        dict: {
            'breadth_ma20': float,   # 站上MA20的股票比例
            'breadth_ma60': float,   # 站上MA60的股票比例
            'up_down_ratio': float,  # 涨跌比（涨家数/跌家数）
            'volume_percentile': float, # 全市场成交额分位数
            'new_high_ratio': float, # 创20日新高比例
        }
    """
    total = 0
    above_ma20 = 0
    above_ma60 = 0
    up_count = 0
    new_high_count = 0
    total_volume = 0.0
    volume_hist = []

    for code, df in stock_data.items():
        if date not in df.index:
            continue
        total += 1
        row = df.loc[date]

        # MA20 / MA60
        if 'ma20' in df.columns:
            ma20 = row.get('ma20', np.nan)
            if not pd.isna(ma20) and row['close'] > ma20:
                above_ma20 += 1
        if 'ma60' in df.columns:
            ma60 = row.get('ma60', np.nan)
            if not pd.isna(ma60) and row['close'] > ma60:
                above_ma60 += 1

        # 涨跌
        ret = row.get('returns', np.nan)
        if not pd.isna(ret):
            if ret > 0:
                up_count += 1
            total_volume += row.get('volume', 0)

        # 创20日新高
        if len(df.loc[:date]) >= 20:
            recent_high = df.loc[:date].iloc[-21:-1]['high'].max()
            if row['high'] >= recent_high * 0.995:
                new_high_count += 1

    if total == 0:
        return {'breadth_ma20': 0.5, 'breadth_ma60': 0.5,
                'up_down_ratio': 1.0, 'volume_percentile': 50, 'new_high_ratio': 0.05}

    down_count = total - up_count
    up_down_ratio = up_count / max(1, down_count)

    return {
        'breadth_ma20': above_ma20 / total,
        'breadth_ma60': above_ma60 / total,
        'up_down_ratio': up_down_ratio,
        'volume_percentile': 50,  # 需要历史对比，先用中性值
        'new_high_ratio': new_high_count / total,
        'total_stocks': total,
    }


# ============================================================
# 1.8 根据市场宽度输出仓位建议
# ============================================================

def breadth_to_position(breadth: dict) -> dict:
    """
    将市场宽度转化为仓位建议

    逻辑:
      - 宽度>60% + 涨跌比>1.5 = 强势牛市 → 满仓
      - 宽度30-60% = 震荡 → 中性仓位
      - 宽度<30% + 涨跌比<0.7 = 弱势熊市 → 低仓/空仓
    """
    b20 = breadth.get('breadth_ma20', 0.5)
    b60 = breadth.get('breadth_ma60', 0.5)
    udr = breadth.get('up_down_ratio', 1.0)

    if b20 > 0.60 and b60 > 0.55 and udr > 1.5:
        regime = 'bull'
        max_pos = 6
        risk = 0.025
    elif b20 > 0.40 and udr > 1.0:
        regime = 'neutral'
        max_pos = 5
        risk = 0.020
    elif b20 > 0.25:
        regime = 'cautious'
        max_pos = 4
        risk = 0.015
    else:
        regime = 'bear'
        max_pos = 3
        risk = 0.010

    return {
        'regime': regime,
        'max_positions': max_pos,
        'risk_per_trade': risk,
        'breadth_ma20': b20,
        'up_down_ratio': udr,
    }


# ============================================================
# 2. 日线数据获取（核心）
# ============================================================

def download_daily(stock_code: str, start_date: str, end_date: str,
                   adj: str = 'qfq', use_cache: bool = True) -> pd.DataFrame:
    """
    下载单只股票的日线数据（前复权）

    参数:
        stock_code: 股票代码，如 '000001.SZ'
        start_date: 起始日期 '20200101'
        end_date:   结束日期 '20241231'
        adj:        复权类型 'qfq'=前复权, 'hfq'=后复权, None=不复权
        use_cache:  是否使用本地缓存

    返回:
        DataFrame with columns: open, high, low, close, volume, amount
        index = DatetimeIndex

    面试考点：
      Q: 前复权和后复权的区别？回测用哪个？
      A: 前复权=以最新股本为基准往前调整，最新价=实际价，历史价被压缩
         后复权=以上市时股本为基准往后调整，历史价=实际价，最新价被放大
         回测用前复权：因为你的买卖决策基于当前实际价格
    """
    # —— 缓存检查 ——
    cache_file = CACHE_DIR / f"{stock_code}_{start_date}_{end_date}_{adj}.csv"
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if not df.empty:
            return df

    # —— 调 tushare API ——
    pro = get_pro()

    # tushare 接口每次最多返回 5000 条，需要分批获取
    all_dfs = []
    current_start = start_date

    while current_start < end_date:
        try:
            df_chunk = pro.daily(
                ts_code=stock_code,
                start_date=current_start,
                end_date=end_date,
                adj=adj,
                fields='trade_date,open,high,low,close,vol,amount'
            )
        except Exception as e:
            print(f"  [警告] {stock_code} 数据获取失败 ({current_start}-{end_date}): {e}")
            break

        if df_chunk is None or df_chunk.empty:
            break

        all_dfs.append(df_chunk)

        # 更新下次请求的起始日期
        last_date = df_chunk['trade_date'].min()
        if last_date <= current_start:
            break
        end_date = str(int(last_date) - 1)

    if not all_dfs:
        print(f"  [错误] {stock_code} 无数据")
        return pd.DataFrame()

    # —— 数据整理 ——
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.rename(columns={
        'trade_date': 'date',
        'vol': 'volume',
    })
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]

    # 清洗：去掉 volume=0 的行（停牌日）
    df = df[df['volume'] > 0]

    # —— 保存缓存 ——
    if use_cache:
        df.to_csv(cache_file)

    return df


def download_index_daily(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    下载指数日线数据（用于择时判断）

    参数:
        index_code: 指数代码，如 '000001.SH'（上证指数）
    """
    pro = get_pro()

    cache_file = CACHE_DIR / f"IDX_{index_code}_{start_date}_{end_date}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    all_dfs = []
    current_start = start_date

    while current_start < end_date:
        df_chunk = pro.index_daily(
            ts_code=index_code,
            start_date=current_start,
            end_date=end_date,
            fields='trade_date,open,high,low,close,vol,amount'
        )

        if df_chunk is None or df_chunk.empty:
            break

        all_dfs.append(df_chunk)
        last_date = df_chunk['trade_date'].min()
        if last_date <= current_start:
            break
        end_date = str(int(last_date) - 1)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
    df = df[df['volume'] > 0]

    df.to_csv(cache_file)
    return df


# ============================================================
# 3. 基本面数据
# ============================================================

def download_fundamentals(trade_date: str, stock_list: list = None) -> pd.DataFrame:
    """
    获取指定日期的基本面数据

    参数:
        trade_date: 交易日 '20240101'
        stock_list: 限定股票列表

    返回:
        DataFrame with columns: ts_code, pe, pb, roe, total_mv, turnover_rate

    面试考点：
      Q: 为什么要把基本面因子和技术面因子结合？
      A: 技术面=市场情绪+短期供需，基本面=内在价值
         两者结合=既不被情绪带跑（只看技术面），也不死守估值（只看基本面）
         业界称为"Quantamental"——量化+基本面融合
    """
    pro = get_pro()

    cache_file = CACHE_DIR / f"FUNDA_{trade_date}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0)
        if stock_list:
            df = df[df['ts_code'].isin(stock_list)]
        return df

    try:
        df = pro.daily_basic(
            trade_date=trade_date,
            fields='ts_code,total_mv,pe,pb,roe,turnover_rate,volume_ratio'
        )
    except Exception as e:
        print(f"  [警告] 基本面数据获取失败 ({trade_date}): {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # 清洗
    df = df.dropna(subset=['total_mv'])  # 没有市值的去掉
    df = df[df['pe'] > 0]                # PE为负=亏损，去掉
    df = df[df['pb'] > 0]                # PB为负=资不抵债，去掉

    df.to_csv(cache_file)
    return df


# ============================================================
# 4. 批量数据加载（一键拉取）
# ============================================================

def load_multi_stock_data(stock_codes: list, start_date: str, end_date: str,
                          progress: bool = True) -> dict:
    """
    批量下载多只股票的数据

    参数:
        stock_codes: 股票代码列表
        start_date:  起始日期
        end_date:    结束日期
        progress:    是否显示进度

    返回:
        dict: {stock_code: DataFrame}

    面试考点：
      Q: 如果500只股票并行下载，tushare有频率限制怎么办？
      A: ① tushare pro 免费版限制 200次/分钟
         ② 用 time.sleep 控制频率
         ③ 或者先拉缓存，只对新股票调API
         ④ 面试时可以提"生产者-消费者模型"做异步下载
    """
    import time
    result = {}
    n = len(stock_codes)
    api_calls = 0  # 统计实际 API 调用次数

    for i, code in enumerate(stock_codes):
        if progress and i % 100 == 0:
            print(f"  [进度] {i}/{n} 只 (已获取 {len(result)}, API调用 {api_calls})")

        # —— 先查缓存，命中则直接读取，不消耗 API 配额 ——
        cache_file = CACHE_DIR / f"{code}_{start_date}_{end_date}_qfq.csv"
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    result[code] = df
                    continue  # 缓存命中 → 跳过 API
            except Exception:
                pass  # 缓存损坏，走 API 下载

        # —— API 下载 ——
        try:
            df = download_daily(code, start_date, end_date)
            if not df.empty:
                result[code] = df
                api_calls += 1
        except Exception as e:
            print(f"  [跳过] {code}: {e}")

        # 控制 API 频率（tushare 免费版: 50次/分钟 → 每次间隔 ≥1.3秒）
        time.sleep(1.3)

    print(f"  [完成] 成功获取 {len(result)}/{n} 只股票数据")
    return result


# ============================================================
# 5. 便捷入口
# ============================================================

if __name__ == "__main__":
    # 测试：初始化 + 拉一只股票数据
    init_tushare()
    print("测试数据获取...\n")

    # 测试1：日线数据
    df = download_daily('000001.SZ', '20230101', '20240601')
    print(f"平安银行 日线数据: {len(df)} 条")
    print(f"  日期范围: {df.index[0]} ~ {df.index[-1]}")
    print(f"  列: {df.columns.tolist()}")
    print()

    # 测试2：指数数据
    df_idx = download_index_daily('000001.SH', '20230101', '20240601')
    print(f"上证指数 日线数据: {len(df_idx)} 条")

    # 测试3：股票池
    stocks = get_stock_pool('hs300')
    print(f"沪深300成分股: {len(stocks)} 只")
    print(f"  前5只: {stocks[:5]}")
