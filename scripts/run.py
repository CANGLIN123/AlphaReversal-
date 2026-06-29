"""
===========================================================
主入口 — 一键运行完整回测流程
===========================================================
用法:
  python run.py                     # 默认：沪深300，2020-2024
  python run.py --pool hs300        # 沪深300
  python run.py --pool zz500        # 中证500
  python run.py --pool top20        # 前20只（快速测试）
  python run.py --start 20220101    # 自定义起始日期
  python run.py --no-plot           # 不生成图表（更快）
"""

import sys
import argparse
from pathlib import Path

# 确保能 import src 目录的模块
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_engine import (
    init_tushare, get_stock_pool, download_index_daily,
    load_multi_stock_data, download_daily, filter_by_liquidity,
    get_stock_sectors
)
from factor_engine import compute_all_factors
from backtest_engine import BacktestEngine
from visualizer import plot_full_report
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='增强版KDJ+量价多因子策略回测')
    parser.add_argument('--pool', default='hs300',
                        choices=['hs300', 'zz500', 'top1500', 'all', 'all_filtered', 'top20'],
                        help='股票池 (默认: hs300)')
    parser.add_argument('--start', default='20200101', help='起始日期 YYYYMMDD')
    parser.add_argument('--end', default='20240601', help='结束日期 YYYYMMDD')
    parser.add_argument('--capital', type=float, default=1_000_000, help='初始资金')
    parser.add_argument('--no-plot', action='store_true', help='不生成图表')
    parser.add_argument('--execution', default='next_open',
                        choices=['close', 'next_open', 'next_vwap', 'next_close'],
                        help='成交模型 (默认: next_open)')
    parser.add_argument('--no-adaptive', action='store_true', help='禁用自适应参数')
    parser.add_argument('--no-batch', action='store_true', help='禁用分批建仓')
    parser.add_argument('--config', default=None, help='自定义配置JSON文件')
    args = parser.parse_args()

    # ============================================================
    # 0. 初始化
    # ============================================================
    print("=" * 65)
    print("   增强版 KDJ+量价 多因子选股策略 — 回测系统")
    print("=" * 65)
    print(f"  启动时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    init_tushare()

    # ============================================================
    # 1. 获取股票池
    # ============================================================
    if args.pool == 'top20':
        # 快速测试：市值前20
        stock_list = get_stock_pool('top1500')[:20]
        print(f"\n[快速模式] 使用前20只股票进行测试")
    else:
        stock_list = get_stock_pool(args.pool)
    print(f"  股票池: {len(stock_list)} 只")

    # ============================================================
    # 2. 下载数据
    # ============================================================
    print(f"\n[数据下载] {args.start} ~ {args.end}")
    stock_data = load_multi_stock_data(stock_list, args.start, args.end)

    if len(stock_data) == 0:
        print("[错误] 未获取到任何股票数据，请检查 tushare token 或网络")
        return

    # 流动性过滤（全市场池时自动启用，日均成交额<2000万过滤）
    if args.pool in ('all', 'all_filtered', 'top1500'):
        stock_data = filter_by_liquidity(stock_data, min_daily_amount=20_000)

    # 下载指数
    print("\n[指数数据]")
    index_df = download_index_daily('000001.SH', args.start, args.end)
    if index_df.empty:
        print("  [警告] 指数数据获取失败，将不使用指数择时")
        index_df = None
    else:
        print(f"  上证指数: {len(index_df)} 条")

    # ============================================================
    # 3. 计算因子
    # ============================================================
    print(f"\n[因子计算] 处理 {len(stock_data)} 只股票...")
    valid_stocks = {}
    for i, (code, df) in enumerate(stock_data.items()):
        try:
            df_with_factors = compute_all_factors(df)
            # 只保留有足够因子数据的
            if len(df_with_factors.dropna(subset=['kdj_j', 'bbi', 'atr14'])) >= 60:
                valid_stocks[code] = df_with_factors
        except Exception as e:
            print(f"  [跳过] {code}: {e}")

        if i % 50 == 49:
            print(f"  进度: {i+1}/{len(stock_data)}")

    print(f"  有效股票: {len(valid_stocks)} 只 (过滤掉了 {len(stock_data) - len(valid_stocks)} 只)")

    # ============================================================
    # 3.5 行业分类
    # ============================================================
    print(f"\n[行业分类] 加载申万行业数据...")
    stock_sectors = get_stock_sectors()
    if stock_sectors:
        # 检查覆盖率
        covered = sum(1 for c in valid_stocks if c in stock_sectors)
        print(f"  行业覆盖: {covered}/{len(valid_stocks)} ({covered/len(valid_stocks)*100:.0f}%)")
    else:
        print("  [警告] 行业数据为空，将跳过行业中性化")

    # ============================================================
    # 4. 配置
    # ============================================================
    config = {
        # —— 资金与仓位 ——
        'initial_capital': args.capital,
        'max_positions': 4,                # 降低持仓上限（5→4，集中火力）
        'max_positions_low': 2,            # 震荡市更低（3→2）
        'risk_per_trade': 0.015,           # 单笔风险降低（2%→1.5%，控制回撤）

        # —— 成交模型 ——
        'execution_model': 'next_open',     # 'close'|'next_open'|'next_vwap'|'next_close'
        'volume_participation_rate': 0.05,

        # —— 止损与止盈 ——
        'atr_stop_multiplier': 2.0,         # 硬止损 ATR 倍数（2.5→2.0，更快截断亏损）
        'trailing_stop_multiplier': 2.0,    # 移动止盈 ATR 距离
        'trailing_activation': 0.04,        # 浮盈 4% 激活移动止盈（3→4，让利润跑更远）
        'partial_profit_taking': True,      # 分批止盈
        'partial_profit_levels': [0.10, 0.20],
        'partial_exit_ratios': [0.25, 0.25], # 每级止盈只卖25%（30→25，留更多仓位）

        # —— 持仓时间 ——
        'max_hold_days': 25,                # 最大持仓天数
        'profit_deadline_days': 10,         # 利润检查日
        'profit_deadline_threshold': -0.02, # 允许微亏
        'bbi_profit_protect_pct': 0.05,     # BBI 保护激活浮盈
        'bbi_profit_exit_ratio': 0.3,       # BBI 保护卖出比例（50→30%，让利润继续跑）

        # —— 分批建仓 ——
        'batch_entry_enabled': True,
        'batch_entry_days': 2,
        'batch_entry_ratios': [0.5, 0.5],
        'batch_improvement_check': True,
        'batch_exit_enabled': True,
        'batch_exit_days': 2,
        'batch_exit_ratios': [0.5, 0.5],

        # —— 熔断与风控 ——
        'portfolio_stop_loss': 0.20,
        'max_sector_exposure': 0.40,
        'adx_trend_threshold': 25,         # ADX 趋势阈值 20→25（更严格判断"趋势市"）
        'adaptive_params_enabled': True,

        # —— 选股 ——
        'j_threshold': 20,
        'j_60d_min_threshold': 12,     # 60日内J曾低于12（更严格要求超卖深度）
        'min_score': 50,               # 最低综合得分 45→50（提高选股门槛）
        'score_weights': {
            'oversold': 0.14,              # 18→14（弹性因子覆盖部分逻辑）
            'volume_shrink': 0.13,         # 16→13（弹性看全程缩量更全面）
            'small_candle': 0.08,          # 10→8
            'prior_surge': 0.12,
            'reversal': 0.11,              # 12→11（弹性与反转互补）
            'bbi_trend': 0.10,
            'ma60_proximity': 0.07,
            'hot_industry': 0.12,
            'amihud': 0.03,
            'rebound_elasticity': 0.10,    # 🆕 反弹弹性
        },
    }

    # 应用 CLI 参数覆盖
    config['execution_model'] = args.execution
    if args.no_adaptive:
        config['adaptive_params_enabled'] = False
    if args.no_batch:
        config['batch_entry_enabled'] = False
        config['batch_exit_enabled'] = False

    # 可选：从 JSON 加载自定义配置
    if args.config:
        import json
        with open(args.config, 'r') as f:
            custom = json.load(f)
        config.update(custom)
        print(f"\n[配置] 加载自定义配置: {args.config}")

    # ============================================================
    # 5. 运行回测
    # ============================================================
    print(f"\n[回测] 开始...")
    engine = BacktestEngine(config)
    daily_stats = engine.run(valid_stocks, index_df, stock_sectors=stock_sectors)

    if daily_stats is None or len(daily_stats) == 0:
        print("[错误] 回测无结果")
        return

    # ============================================================
    # 6. 计算基准收益（沪深300买入持有）
    # ============================================================
    benchmark_metrics = None
    benchmark_stats = None
    try:
        hs300_df = download_index_daily('000300.SH', args.start, args.end)
        if not hs300_df.empty:
            hs300_df['returns'] = hs300_df['close'].pct_change()
            hs300_df['equity'] = config['initial_capital'] * (
                1 + hs300_df['returns']
            ).cumprod()

            # 计算基准指标
            r = hs300_df['returns'].dropna()
            n_years = len(r) / 252
            total_ret = hs300_df['equity'].iloc[-1] / config['initial_capital'] - 1
            cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
            ann_vol = r.std() * np.sqrt(252)
            cum = hs300_df['equity'] / config['initial_capital']
            max_dd = ((cum - cum.expanding().max()) / cum.expanding().max()).min()
            sharpe = (cagr - 0.03) / ann_vol if ann_vol > 0 else 0

            benchmark_metrics = {
                '总收益率': f"{total_ret:.2%}",
                '年化收益(CAGR)': f"{cagr:.2%}",
                '年化波动率': f"{ann_vol:.2%}",
                '夏普比率(Sharpe)': f"{sharpe:.3f}",
                '最大回撤': f"{max_dd:.2%}",
            }
            benchmark_stats = hs300_df
    except Exception as e:
        print(f"  [警告] 基准计算失败: {e}")

    # ============================================================
    # 7. 打印报告
    # ============================================================
    engine.report(benchmark_metrics)

    # ============================================================
    # 8. 导出
    # ============================================================
    engine.export()

    # ============================================================
    # 9. 可视化
    # ============================================================
    if not args.no_plot:
        print("\n[可视化] 生成图表...")
        trades_df = pd.DataFrame(engine.trades) if engine.trades else None
        plot_full_report(
            daily_stats=engine.daily_stats,
            trades_df=trades_df,
            benchmark_stats=benchmark_stats,
            score_weights=config['score_weights'],
            save_path=Path(__file__).parent.parent / "output" / "backtest_report.png"
        )

    print("\n" + "=" * 65)
    print("   回测完成！查看 output/ 目录下的结果文件")
    print("=" * 65)


if __name__ == "__main__":
    main()
