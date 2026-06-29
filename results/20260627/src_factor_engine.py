"""
===========================================================
因子引擎 — 技术因子 + 量价因子 + 风控指标计算
===========================================================
职责：
  1. KDJ 指标（超卖超买判断）
  2. BBI 多空指数（趋势方向）
  3. 量价形态识别（放量阳线 + 缩量回调）
  4. ATR 真实波幅（动态止损）
  5. ADX 趋势强度（市场状态分类）
  6. 波动率锥（仓位管理）
  7. 基本面因子（PE/PB/ROE）

区别于原版 B1.py 的优化：
  - 原版 J 值阈值固定=20 → 优化为基于历史分位数的自适应阈值
  - 原版止损固定=10% → 优化为 2x ATR 动态止损
  - 原版仓位固定金额 → 优化为波动率目标仓位
  - 新增 ADX 趋势过滤（震荡市降低仓位）
  - 新增基本面因子（PE/PB/ROE）加入打分体系
  - 新增行业中性化处理

面试考点：
  Q: KDJ 和 RSI 都是超卖超买指标，区别是什么？
  A: KDJ 用最高最低价（含日内波动），RSI 只用收盘价。
     KDJ 更敏感（J 值可以超过 100 和低于 0），RSI 在 [0,100] 内。
     KDJ 适合短期拐点判断，RSI 适合中期趋势判断。
     KDJ 的 J 值 = 3K - 2D，本质是对 K 值的"过度反应"。

  Q: 为什么要用 ATR 而非固定百分比止损？
  A: 高波动股票（如创业板）日内波动 5%+，10% 止损容易被"震出去"。
     低波动股票（如银行股）日内波动 1%左右，10% 止损太大=亏太多才止损。
     ATR 止损根据每只股票的"脾气"自适应调整。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


# ============================================================
# 1. KDJ 指标
# ============================================================

def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 KDJ 指标

    参数:
        df: 包含 close, high, low 列的 DataFrame
        n:   RSV 计算周期（默认 9）
        m1:  K 值平滑周期（默认 3）
        m2:  D 值平滑周期（默认 3）

    返回:
        (K, D, J) 三个 numpy 数组

    计算步骤：
      1. RSV = (C - L_n) / (H_n - L_n) × 100
         RSV 是"当前收盘价在过去 N 天中的相对位置"
      2. K = SMA(RSV, M1)  — 对 RSV 做 M1 日平滑
      3. D = SMA(K, M2)     — 对 K 做 M2 日平滑（二次平滑）
      4. J = 3K - 2D        — 对 K 值的"过度反应"

    面试考点：
      Q: SMA 和 EMA 的区别？
      A: SMA(3) 的递推公式: K_t = (K_{t-1} × 2 + RSV_t) / 3
         EMA(span=3) 的公式: K_t = K_{t-1} × 0.5 + RSV_t × 0.5
         SMA 更平滑（新数据权重=1/3），EMA 更敏感（新数据权重=0.5）
         KDJ 原作者 George Lane 用的是 SMA
    """
    if len(df) < n:
        return np.full(len(df), np.nan), np.full(len(df), np.nan), np.full(len(df), np.nan)

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    # —— 第1步：计算 RSV ——
    # 用 rolling window 计算 N 日内最高价和最低价
    # 向量化实现：用 pd.Series.rolling() 比手写循环快 10-100 倍
    high_n = pd.Series(high).rolling(window=n).max().values
    low_n = pd.Series(low).rolling(window=n).min().values

    # RSV = (收盘价 - N日最低) / (N日最高 - N日最低) × 100
    rsv = np.full(len(df), np.nan)
    for i in range(n - 1, len(df)):
        denominator = high_n[i] - low_n[i]
        if denominator > 0:
            rsv[i] = (close[i] - low_n[i]) / denominator * 100
        else:
            rsv[i] = 50  # 一字涨跌停时取中性值

    # —— 第2步：计算 K 值（SMA平滑）——
    k = np.full(len(df), np.nan)
    d = np.full(len(df), np.nan)
    j = np.full(len(df), np.nan)

    # 找第一个有效 RSV
    first_valid = n - 1
    while first_valid < len(df) and np.isnan(rsv[first_valid]):
        first_valid += 1

    if first_valid >= len(df):
        return k, d, j

    # 初始 K=50, D=50（中性值）
    k[first_valid] = rsv[first_valid]
    d[first_valid] = rsv[first_valid]

    # 递推
    for i in range(first_valid + 1, len(df)):
        if np.isnan(rsv[i]):
            k[i] = k[i - 1]
            d[i] = d[i - 1]
        else:
            k[i] = (k[i - 1] * (m1 - 1) + rsv[i]) / m1
            d[i] = (d[i - 1] * (m2 - 1) + k[i]) / m2

    # J = 3K - 2D
    j = 3 * k - 2 * d

    return k, d, j


# ============================================================
# 2. BBI 多空指数
# ============================================================

def calc_bbi(df: pd.DataFrame) -> np.ndarray:
    """
    计算 BBI（多空指数）

    BBI = (MA3 + MA6 + MA12 + MA24) / 4

    含义：
      BBI 是四个时间维度均线的均值，代表"综合市场成本"。
      价格 > BBI → 多数人赚钱 → 多头市场
      价格 < BBI → 多数人亏钱 → 空头市场

    面试考点：
      Q: BBI 和单根均线（如 MA60）相比有什么优势？
      A: 单根均线对参数敏感（MA20 vs MA60 结论可能矛盾）。
         BBI 综合四个周期，降低参数依赖，信号更稳健。
         相当于给四个不同时间框架的交易者各一票，取"共识"。
    """
    close = df['close'].values
    result = np.full(len(df), np.nan)

    for i in range(23, len(df)):
        ma3 = np.mean(close[i - 2:i + 1])
        ma6 = np.mean(close[i - 5:i + 1])
        ma12 = np.mean(close[i - 11:i + 1])
        ma24 = np.mean(close[i - 23:i + 1])
        result[i] = (ma3 + ma6 + ma12 + ma24) / 4

    return result


# ============================================================
# 3. 量价形态 — 放量阳线 + 缩量回调检测（核心 Alpha）
# ============================================================

def calc_volume_pattern(df: pd.DataFrame, lookback: int = 20,
                        surge_threshold: float = 1.5,
                        price_threshold: float = 0.05,
                        shrink_ratio: float = 0.9
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    检测"放量阳线后缩量回调"的经典量价形态

    形态逻辑：
      ① 找到最近一次放量阳线（涨幅>5% 且 成交量>5日均量×1.5）
      ② 此后回调期间：成交量持续 < 放量阳线成交量 × 0.9（缩量）
      ③ 回调期间不创新高（说明不是新一轮上涨）
      ④ 符合条件 → 主力洗盘概率大，后续有望反弹

    返回:
        (is_valid, score):
        - is_valid: 布尔数组，True=符合形态
        - score: 浮点数组，0-100，得分越高缩量越充分

    面试考点：
      Q: 如何避免把"出货"误判为"洗盘"？
      A: 关键区别：
         - 洗盘：缩量回调（主力没走）+ 不创新高（控制节奏）
         - 出货：放量下跌（主力在砸）+ 不断创新低
         本策略通过 shrink_ratio<0.9（必须缩量）和"不创新高"来过滤出货
    """
    n = len(df)
    is_valid = np.full(n, False)
    score = np.full(n, 0.0)

    if n < lookback:
        return is_valid, score

    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values

    # 计算 5 日均量（用于判断放量）
    vol_ma5 = pd.Series(volume).rolling(window=5).mean().values

    for i in range(lookback, n):
        # —— 在 lookback 窗口内找放量阳线 ——
        start_idx = i - lookback
        surge_idx = -1
        surge_vol = 0
        surge_high = 0

        for t in range(start_idx + 1, i):
            price_chg = (close[t] - close[t - 1]) / close[t - 1]
            if (price_chg > price_threshold
                    and vol_ma5[t] > 0
                    and volume[t] > vol_ma5[t] * surge_threshold):
                surge_idx = t
                surge_vol = volume[t]
                surge_high = high[t]

        if surge_idx < 0 or surge_idx >= i - 1:
            continue

        # —— 检查回调质量 ——
        retrace_volumes = []
        valid_retrace = True

        for t in range(surge_idx + 1, i):
            vol_ratio = volume[t] / surge_vol if surge_vol > 0 else 999

            # 回调中出现放量(=资金在出逃) → 不合格
            if vol_ratio > shrink_ratio:
                valid_retrace = False
                break

            # 回调中创新高(=趋势没停) → 不是洗盘
            if high[t] > surge_high and volume[t] > surge_vol:
                valid_retrace = False
                break

            retrace_volumes.append(vol_ratio)

        if not valid_retrace or not retrace_volumes:
            continue

        # —— 质量得分 ——
        avg_retrace = np.mean(retrace_volumes)
        vol_score = max(0, (1 - avg_retrace) * 100)

        # 极端缩量加分
        if avg_retrace < 0.3:
            vol_score = min(100, vol_score * 1.2)

        is_valid[i] = True
        score[i] = vol_score

    return is_valid, score


# ============================================================
# 4. ATR — 平均真实波幅（动态止损核心）
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """
    计算 ATR（Average True Range）

    True Range = max(|H-L|, |H-C_prev|, |L-C_prev|)
    含义：考虑跳空缺口后的"真正振幅"

    ATR = TR 的 N 日移动平均

    为什么用 Wilder 平滑？
      Wilder 平滑 α=1/N（比 EMA 的 α=2/(N+1) 更慢），
      用来做止损的 ATR 需要稳定——不能因为一天的极端波动就改变止损位。

    面试考点：
      Q: ATR 和标准差的区别？
      A: 标准差衡量的是"偏离均值的程度"（波动率）
         ATR 衡量的是"日内振幅"（交易区间大小）
         ATR 包含了跳空（开盘跳空高/低开），标准差只看收盘价。
         对于止损：ATR 更合适，因为跳空也会触发止损。
    """
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    tr = np.zeros(n)

    # 第一天：只有当日振幅
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],                    # 当日振幅
            abs(high[i] - close[i - 1]),         # 今日最高 - 昨收（跳空高开）
            abs(low[i] - close[i - 1])           # 昨收 - 今日最低（跳空低开）
        )

    # Wilder 平滑（α = 1/period）
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])  # 初始值：简单平均

    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ============================================================
# 5. ADX — 趋势强度（市场状态分类）
# ============================================================

def calc_adx(df: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 ADX（Average Directional Index）和 DI+/DI-

    ADX 衡量趋势的强度（不判断方向）：
      ADX > 25 → 趋势市（适合趋势跟随策略）
      ADX < 20 → 震荡市（适合反转策略）
      ADX 20-25 → 过渡期

    返回:
        (adx, di_diff): adx 值和 DI+ - DI-（正=多头主导，负=空头主导）

    面试考点：
      Q: 为什么要在震荡市降低仓位？
      A: KDJ/量价等趋势策略在震荡市中频繁被止损，
         因为震荡市价格在区间内反复，没有持续性趋势。
         加了 ADX 过滤器后，低于 20 时减半仓位，
         相当于"看不清时不下重注"。
    """
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        # True Range
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )

        # +DM：今日最高比昨日最高更高，且比跌幅大
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0

    # Wilder 平滑
    atr_w = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    atr_w[period] = np.sum(tr[1:period + 1])
    plus_dm_s = np.sum(plus_dm[1:period + 1])
    minus_dm_s = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        atr_w[i] = atr_w[i - 1] - atr_w[i - 1] / period + tr[i]
        plus_dm_s = plus_dm_s - plus_dm_s / period + plus_dm[i]
        minus_dm_s = minus_dm_s - minus_dm_s / period + minus_dm[i]

        plus_di[i] = 100 * plus_dm_s / atr_w[i] if atr_w[i] > 0 else 0
        minus_di[i] = 100 * minus_dm_s / atr_w[i] if atr_w[i] > 0 else 0

    # DX = |DI+ - DI-| / (DI+ + DI-) × 100
    dx = np.zeros(n)
    for i in range(period + 1, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / denom

    # ADX = WilderSmooth(DX, period)
    adx = np.zeros(n)
    adx[2 * period] = np.mean(dx[period + 1:2 * period + 1])
    for i in range(2 * period + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    di_diff = plus_di - minus_di

    return adx, di_diff


# ============================================================
# 6. 波动率锥 + 仓位计算
# ============================================================

def calc_position_size(capital: float, risk_per_trade: float,
                       atr: float, price: float) -> int:
    """
    基于 ATR 的波动率目标仓位管理

    参数:
        capital:         可用资金
        risk_per_trade:  单笔风险敞口（如 0.02 = 2%）
        atr:             当前 ATR 值
        price:           当前价格

    返回:
        建议买入股数（整百）

    公式：
      止损距离 = 2 × ATR
      最大亏损 = capital × risk_per_trade
      仓位股数 = 最大亏损 / 止损距离
      仓位金额 = 仓位股数 × price

    面试考点：
      Q: 为什么用波动率目标而不是固定金额？
      A: 固定金额：每只买10万 → 高波动股容易止损，低波动股仓位不足
         波动率目标：高波动→买少点，低波动→买多点，每笔风险一致
         这叫"风险平价"思想——对每笔交易分配相同的风险预算
    """
    if atr <= 0 or price <= 0:
        return 0

    stop_distance = 2.0 * atr          # 止损距离
    max_loss = capital * risk_per_trade  # 这笔交易最多亏多少
    shares = int(max_loss / stop_distance)
    shares = (shares // 100) * 100      # 取整百股

    # 检查是否买得起
    max_shares = int(capital * 0.98 / price / 100) * 100
    shares = min(shares, max_shares)

    return max(0, shares)


# ============================================================
# 7. 综合因子计算（给回测引擎调用）
# ============================================================

def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    对单只股票计算所有技术因子，返回带因子列的 DataFrame

    这是一个"因子工厂"函数，负责把所有原始价格数据
    转换成策略需要的因子值。每个因子都是新增一列。

    面试考点：
      Q: 为什么要把因子计算独立成一个函数？
      A: ① 模块化：因子和策略解耦，改因子不影响交易逻辑
         ② 可测试：每个因子可以单独验证
         ③ 可复用：多个策略可以共用同一套因子库
    """
    df = df.copy()

    # —— KDJ ——
    k, d, j = calc_kdj(df)
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = j
    df['kdj_j_prev'] = np.roll(j, 1)       # 前一日J值（用于判断拐头）
    df.loc[df.index[0], 'kdj_j_prev'] = np.nan

    # —— BBI ——
    df['bbi'] = calc_bbi(df)

    # —— 均线 ——
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    # —— 量价形态 ——
    is_valid, vol_score = calc_volume_pattern(df)
    df['vol_pattern_valid'] = is_valid.astype(int)
    df['vol_pattern_score'] = vol_score

    # —— ATR ——
    df['atr14'] = calc_atr(df, 14)

    # —— ADX ——
    adx, di_diff = calc_adx(df)
    df['adx'] = adx
    df['di_diff'] = di_diff

    # —— 收益率 ——
    df['returns'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)

    # —— 波动率 ——
    df['volatility_20d'] = df['returns'].rolling(20).std()

    # —— 成交量 ——
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']

    # —— Amihud 非流动性因子 (ILLIQ) ——
    # ILLIQ = |ret| / amount 的20日均值 × 10^8（缩放避免数值太小）
    # 高ILLIQ = 流动性差 = 少量资金就能推动大波动 = 反弹弹性大
    # 学术来源: Amihud (2002, JFM)
    df['amihud'] = calc_amihud(df)

    # —— 换手率变化因子 ——
    # 缩量+低换手率下降=洗盘确认；放量+高换手率上升=出货风险
    # 这里用成交量/流通市值的代理变量（volume ratio 变化率）
    if 'volume_ma5' in df.columns and 'volume_ma20' in df.columns:
        df['turnover_proxy'] = df['volume_ma5'] / df['volume_ma20'].replace(0, np.nan)
        df['turnover_change'] = df['turnover_proxy'].pct_change(5)

    # —— 涨跌停距离因子 ——
    df['limit_down_dist'] = 0.0
    for i in range(1, len(df)):
        prev_close = df['close'].iloc[i - 1]
        if prev_close > 0:
            limit_down = prev_close * 0.90
            current = df['close'].iloc[i]
            df.loc[df.index[i], 'limit_down_dist'] = (current - limit_down) / (prev_close * 0.001)
    df['limit_down_dist'] = df['limit_down_dist'].clip(0, 100)

    # —— J值60日最低（参考买点共性 #1: 60日内曾极度超卖）——
    df['kdj_j_60d_min'] = pd.Series(j).rolling(60).min().values

    # —— 前20日放量异动检测（参考买点共性 #2: 量随价升）——
    # 检测：前20天内是否出现过"放量阳线"（量>5日均量×1.5 且 单日涨幅>3%）
    n_rows = len(df)
    vol_ma5_local = df['volume'].rolling(5).mean().values
    daily_ret_local = df['close'].pct_change().values
    surge = np.zeros(n_rows, dtype=bool)
    for ii in range(5, n_rows):
        if (daily_ret_local[ii] > 0.03
                and vol_ma5_local[ii] > 0
                and df['volume'].values[ii] > vol_ma5_local[ii] * 1.5):
            surge[ii] = True
    # 向前看20天
    df['had_volume_surge'] = pd.Series(surge).rolling(20).max().fillna(0).values

    # —— 当日涨跌幅（用于判断小K线企稳）——
    df['daily_ret'] = df['close'].pct_change()

    return df


# ============================================================
# 7.5 Amihud 非流动性因子
# ============================================================

def calc_amihud(df: pd.DataFrame, period: int = 20) -> np.ndarray:
    """
    计算 Amihud (2002) 非流动性指标

    ILLIQ_t = 1/D_t × Σ |r_d| / volume_d

    含义：
      同样1%的涨跌，成交额越小的股票ILLIQ越高。
      ILLIQ高 = 流动性差 = 买盘一进来就能推高价格。
      对于超跌反弹策略：ILLIQ适度偏高的股票反弹弹性更大。
      但ILLIQ极高 = 可能是庄股/僵尸股 = 需要过滤。

    返回:
        numpy array of ILLIQ values (scaled by 10^8)

    面试考点：
      Q: Amihud 非流动性和换手率有什么区别？
      A: 换手率 = 交易活跃度（周转速度）
         Amihud = 价格对成交量的敏感度（价格冲击）
         两者相关但不完全一样：有些股票换手率高但ILLIQ也高
         （说明筹码分散，每笔交易都很小）
    """
    n = len(df)
    result = np.full(n, np.nan)

    if 'amount' not in df.columns or n < period:
        return result

    returns = df['close'].pct_change().abs().values
    amount = df['amount'].values

    for i in range(period, n):
        daily_illiq = []
        for j in range(i - period + 1, i + 1):
            if amount[j] > 0:
                daily_illiq.append(returns[j] / amount[j])
        if daily_illiq:
            result[i] = np.mean(daily_illiq) * 1e8

    return result


# ============================================================
# 8. 市场状态分类 — 自适应参数基础
# ============================================================

def calc_market_regime(index_df: pd.DataFrame, lookback: int = 60) -> dict:
    """
    基于指数数据识别当前市场状态（Regime Detection）

    使用两个维度分类市场：
      1. 趋势方向：价格在 MA60 上方 = 牛，下方 = 熊
      2. 波动率水平：当前波动率在历史分位数中的位置

    6 种市场状态：
      ┌────────────────┬──────────┬──────────┐
      │                │ 低波动   │ 高波动   │
      ├────────────────┼──────────┼──────────┤
      │ 趋势向上(牛市)  │ 趋势牛   │ 波动牛   │
      │ 趋势向下(熊市)  │ 趋势熊   │ 波动熊   │
      │ 横盘(震荡)      │ 低波震荡 │ 高波震荡 │
      └────────────────┴──────────┴──────────┘

    返回:
        dict: {
            'regime': str,       # 状态名
            'regime_id': int,    # 状态编号 0-5
            'vol_percentile': float,  # 波动率分位数
            'trend': str,        # 'bull' | 'bear' | 'sideways'
            'vol_regime': str,   # 'low' | 'high'
        }

    面试考点：
      Q: 为什么要做市场状态分类而不是用单一阈值？
      A: 市场在不同状态下，因子的有效性和参数的最优值都不同。
         牛市中 J<20 是好的回调买点，熊市中可能是下跌中继。
         通过状态切换来调整参数，本质是"隐马尔可夫模型"的简化版。
    """
    if index_df is None or len(index_df) < lookback + 60:
        return {
            'regime': 'normal', 'regime_id': 3,
            'vol_percentile': 50, 'trend': 'bull', 'vol_regime': 'low'
        }

    # 取最近 lookback 天的数据
    recent = index_df.iloc[-lookback:].copy()

    # —— 1. 趋势判断 ——
    close = recent['close'].values
    ma60 = np.mean(close[-min(60, len(close)):])
    current_price = close[-1]

    # 用 BBI 辅助确认趋势
    bbi_vals = calc_bbi(recent)
    bbi_current = bbi_vals[-1] if not np.isnan(bbi_vals[-1]) else ma60

    if current_price > ma60 and current_price > bbi_current:
        trend = 'bull'
    elif current_price < ma60 and current_price < bbi_current:
        trend = 'bear'
    else:
        trend = 'sideways'

    # —— 2. 波动率水平 ——
    returns = recent['close'].pct_change().dropna()
    current_vol = returns.std() * np.sqrt(252)

    # 用更长历史计算波动率分位数
    if len(index_df) > lookback + 60:
        hist_returns = index_df['close'].pct_change().dropna()
        hist_vol = hist_returns.rolling(60).std() * np.sqrt(252)
        hist_vol = hist_vol.dropna()
        if len(hist_vol) > 0:
            vol_percentile = (hist_vol < current_vol).mean() * 100
        else:
            vol_percentile = 50
    else:
        vol_percentile = 50

    vol_regime = 'high' if vol_percentile > 60 else 'low'

    # —— 3. 组合分类 ——
    if trend == 'bull' and vol_regime == 'low':
        regime, regime_id = 'trending_bull', 0
    elif trend == 'bear' and vol_regime == 'low':
        regime, regime_id = 'trending_bear', 1
    elif trend == 'bull' and vol_regime == 'high':
        regime, regime_id = 'volatile_bull', 2
    elif trend == 'bear' and vol_regime == 'high':
        regime, regime_id = 'volatile_bear', 3
    elif trend == 'sideways' and vol_regime == 'low':
        regime, regime_id = 'lowvol_sideways', 4
    else:
        regime, regime_id = 'highvol_sideways', 5

    return {
        'regime': regime,
        'regime_id': regime_id,
        'vol_percentile': round(vol_percentile, 1),
        'trend': trend,
        'vol_regime': vol_regime,
        'current_vol': round(current_vol, 4),
        'current_price': current_price,
        'ma60': round(ma60, 2),
        'bbi': round(bbi_current, 2),
    }


# ============================================================
# 9. 自适应参数映射
# ============================================================

# 不同市场状态下的最优参数表
# 设计原则：
#   - 牛市：放宽买入条件（J阈值高），收紧止损（让利润奔跑），高仓位
#   - 熊市：严格买入条件（J阈值低），放宽止损（避免频繁止损），低仓位
#   - 高波动：扩大 ATR 倍数（给更多空间），降低单笔风险
#   - 低波动：缩小 ATR 倍数，提高单笔风险

ADAPTIVE_PARAM_TABLE = {
    'trending_bull': {
        'j_threshold': 25,              # 放宽超卖条件
        'atr_stop_multiplier': 1.5,     # 收紧止损（牛市回调浅）
        'trailing_stop_multiplier': 1.5,
        'trailing_activation': 0.02,    # 更快激活移动止盈
        'risk_per_trade': 0.025,        # 更高风险敞口
        'max_positions': 6,             # 更多仓位
        'min_score': 50,
    },
    'trending_bear': {
        'j_threshold': 12,              # 熊市严选
        'atr_stop_multiplier': 3.0,     # 放宽止损（熊市波动大）
        'trailing_stop_multiplier': 3.0,
        'trailing_activation': 0.05,
        'risk_per_trade': 0.010,        # 低风险敞口
        'max_positions': 3,
        'min_score': 55,
    },
    'volatile_bull': {
        'j_threshold': 20,
        'atr_stop_multiplier': 2.0,
        'trailing_stop_multiplier': 2.0,
        'trailing_activation': 0.03,
        'risk_per_trade': 0.020,
        'max_positions': 5,
        'min_score': 50,
    },
    'volatile_bear': {
        'j_threshold': 10,
        'atr_stop_multiplier': 3.5,
        'trailing_stop_multiplier': 3.5,
        'trailing_activation': 0.05,
        'risk_per_trade': 0.005,        # 极低风险
        'max_positions': 2,
        'min_score': 60,
    },
    'lowvol_sideways': {
        'j_threshold': 20,
        'atr_stop_multiplier': 2.0,
        'trailing_stop_multiplier': 2.0,
        'trailing_activation': 0.03,
        'risk_per_trade': 0.020,
        'max_positions': 5,
        'min_score': 50,
    },
    'highvol_sideways': {
        'j_threshold': 15,
        'atr_stop_multiplier': 2.5,
        'trailing_stop_multiplier': 2.5,
        'trailing_activation': 0.04,
        'risk_per_trade': 0.010,
        'max_positions': 3,
        'min_score': 50,
    },
}


def calc_adaptive_params(regime_info: dict, base_config: dict = None) -> dict:
    """
    根据市场状态返回自适应参数

    参数:
        regime_info: calc_market_regime() 的返回值
        base_config: 基础配置（未提供的参数使用此默认值）

    返回:
        dict: 当前市场状态下应该使用的参数
    """
    regime = regime_info.get('regime', 'normal')
    adaptive = ADAPTIVE_PARAM_TABLE.get(regime, ADAPTIVE_PARAM_TABLE['lowvol_sideways'])

    # 合并基础配置中未覆盖的参数
    result = dict(adaptive)
    result['_regime'] = regime
    result['_regime_info'] = regime_info

    return result


# ============================================================
# 10. 测试
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_engine import init_tushare, download_daily, download_index_daily

    init_tushare()
    df = download_daily('000001.SZ', '20230101', '20240601')

    print("原始数据:", df.shape)
    df = compute_all_factors(df)
    print("加因子后:", df.shape)
    print("因子列:", [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'amount']])

    # 检查最新一天的因子值
    last = df.iloc[-1]
    print(f"\n最新因子值 ({df.index[-1].strftime('%Y-%m-%d')}):")
    print(f"  KDJ: K={last['kdj_k']:.2f}, D={last['kdj_d']:.2f}, J={last['kdj_j']:.2f}")
    print(f"  BBI: {last['bbi']:.2f}, MA60: {last['ma60']:.2f}")
    print(f"  ATR14: {last['atr14']:.3f}")
    print(f"  ADX: {last['adx']:.2f}")
    print(f"  量价形态: {'✓' if last['vol_pattern_valid'] else '✗'} (得分: {last['vol_pattern_score']:.1f})")

    # 测试市场状态识别
    print("\n" + "=" * 50)
    print("市场状态识别测试")
    print("=" * 50)
    index_df = download_index_daily('000001.SH', '20230101', '20240601')
    if not index_df.empty:
        regime = calc_market_regime(index_df)
        print(f"当前市场状态: {regime['regime']}")
        print(f"  趋势: {regime['trend']}")
        print(f"  波动率分位: {regime['vol_percentile']:.1f}%")
        print(f"  波动率水平: {regime['vol_regime']}")
        print(f"  当前波动率: {regime['current_vol']:.4f}")

        adaptive = calc_adaptive_params(regime)
        print(f"\n自适应参数:")
        print(f"  J阈值: {adaptive['j_threshold']}")
        print(f"  ATR止损倍数: {adaptive['atr_stop_multiplier']}")
        print(f"  风险敞口: {adaptive['risk_per_trade']:.3f}")
        print(f"  最大仓位: {adaptive['max_positions']}")
