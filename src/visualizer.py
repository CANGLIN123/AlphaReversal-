"""
===========================================================
可视化模块 — 回测结果图表 + 分析
===========================================================
生成 6 张标准量化报告图表：
  1. 净值曲线（策略 vs 基准）
  2. 回撤曲线
  3. 月度收益热力图
  4. 年度收益分布
  5. 交易盈亏散点图
  6. 因子权重饼图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Optional

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_full_report(daily_stats: pd.DataFrame,
                     trades_df: Optional[pd.DataFrame] = None,
                     benchmark_stats: Optional[pd.DataFrame] = None,
                     score_weights: Optional[dict] = None,
                     save_path: Optional[str] = None):
    """
    生成完整的回测报告图表（6面板）

    参数:
        daily_stats:     每日净值 DataFrame（含 equity, returns, drawdown）
        trades_df:       交易记录（含 buy_date, sell_date, pnl, reason）
        benchmark_stats: 基准净值 DataFrame
        score_weights:   因子权重 dict
        save_path:       保存路径
    """
    if daily_stats is None or len(daily_stats) == 0:
        print("[警告] 无数据，无法绘图")
        return

    fig = plt.figure(figsize=(18, 14))

    # ===== 面板1：净值曲线 =====
    ax1 = plt.subplot(3, 2, 1)
    plot_equity_curve(ax1, daily_stats, benchmark_stats)

    # ===== 面板2：回撤曲线 =====
    ax2 = plt.subplot(3, 2, 2)
    plot_drawdown(ax2, daily_stats, benchmark_stats)

    # ===== 面板3：月度收益热力图 =====
    ax3 = plt.subplot(3, 2, 3)
    plot_monthly_heatmap(ax3, daily_stats)

    # ===== 面板4：年度收益 =====
    ax4 = plt.subplot(3, 2, 4)
    plot_annual_returns(ax4, daily_stats)

    # ===== 面板5：交易分布 =====
    ax5 = plt.subplot(3, 2, 5)
    if trades_df is not None and len(trades_df) > 0:
        plot_trade_scatter(ax5, trades_df)
    else:
        ax5.text(0.5, 0.5, '无交易数据', ha='center', va='center',
                 transform=ax5.transAxes, fontsize=14)

    # ===== 面板6：因子权重 =====
    ax6 = plt.subplot(3, 2, 6)
    if score_weights:
        plot_factor_weights(ax6, score_weights)
    else:
        # 默认权重
        plot_factor_weights(ax6, {
            'J值因子': 0.25,
            '量价因子': 0.25,
            'BBI趋势': 0.20,
            '动量因子': 0.15,
            '质量因子': 0.15,
        })

    plt.tight_layout(pad=3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[图表] 已保存: {save_path}")
    else:
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "backtest_report.png", dpi=150, bbox_inches='tight')
        print(f"[图表] 已保存: {output_dir / 'backtest_report.png'}")


def plot_equity_curve(ax, daily_stats, benchmark_stats=None):
    """净值曲线"""
    norm_equity = daily_stats['equity'] / daily_stats['equity'].iloc[0]

    ax.plot(daily_stats.index, norm_equity, color='#1f77b4', linewidth=1.8, label='策略净值')

    if benchmark_stats is not None:
        norm_bench = benchmark_stats['equity'] / benchmark_stats['equity'].iloc[0]
        ax.plot(benchmark_stats.index, norm_bench, color='gray', linewidth=1,
                alpha=0.6, label='沪深300')

    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.fill_between(daily_stats.index, 1, norm_equity,
                     where=norm_equity >= 1, color='#1f77b4', alpha=0.1)
    ax.fill_between(daily_stats.index, 1, norm_equity,
                     where=norm_equity < 1, color='red', alpha=0.1)

    ax.set_title('累计净值曲线', fontsize=13, fontweight='bold')
    ax.set_ylabel('净值', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def plot_drawdown(ax, daily_stats, benchmark_stats=None):
    """回撤曲线"""
    cum = daily_stats['equity'] / daily_stats['equity'].iloc[0]
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max * 100

    ax.fill_between(daily_stats.index, 0, dd, color='#d62728', alpha=0.35,
                     label=f'策略回撤 (最大: {dd.min():.1f}%)')

    if benchmark_stats is not None:
        bench_cum = benchmark_stats['equity'] / benchmark_stats['equity'].iloc[0]
        bench_max = bench_cum.expanding().max()
        bench_dd = (bench_cum - bench_max) / bench_max * 100
        ax.plot(benchmark_stats.index, bench_dd, color='gray', linewidth=1,
                alpha=0.6, label=f'沪深300回撤')

    ax.set_title('回撤曲线', fontsize=13, fontweight='bold')
    ax.set_ylabel('回撤 (%)', fontsize=10)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


def plot_monthly_heatmap(ax, daily_stats):
    """月度收益热力图"""
    if 'returns' not in daily_stats.columns:
        ax.text(0.5, 0.5, '无收益率数据', ha='center', va='center',
                 transform=ax.transAxes, fontsize=12)
        return

    # 计算月度收益
    monthly = daily_stats['returns'].resample('ME').apply(
        lambda x: (1 + x).prod() - 1
    )

    # 整理成 pivot table
    monthly_df = monthly.to_frame('return')
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    pivot = monthly_df.pivot_table(values='return', index='month', columns='year', aggfunc='sum')

    # 画热力图
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)

    # 标注数字
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1%}', ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='black' if abs(val) < 0.07 else 'white')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{m}月' for m in pivot.index])
    ax.set_title('月度收益热力图', fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.8, format=mticker.PercentFormatter(xmax=1))


def plot_annual_returns(ax, daily_stats):
    """年度收益柱状图"""
    if 'returns' not in daily_stats.columns:
        ax.text(0.5, 0.5, '无收益率数据', ha='center', va='center',
                 transform=ax.transAxes, fontsize=12)
        return

    annual = daily_stats['returns'].resample('YE').apply(
        lambda x: (1 + x).prod() - 1
    )

    years = [d.year for d in annual.index]
    values = annual.values * 100

    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in values]
    bars = ax.bar(range(len(years)), values, color=colors, alpha=0.8, width=0.6)

    # 标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (1 if val >= 0 else -3),
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_title('年度收益率', fontsize=13, fontweight='bold')
    ax.set_ylabel('收益率 (%)', fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')


def plot_trade_scatter(ax, trades_df):
    """交易盈亏散点图（按时间排列）"""
    trades_df = trades_df.copy()
    trades_df['sell_date_num'] = (trades_df['sell_date'] - trades_df['sell_date'].min()).dt.days

    colors = ['#2ca02c' if pnl > 0 else '#d62728' for pnl in trades_df['pnl']]
    sizes = [min(100, abs(pnl)/500) for pnl in trades_df['pnl']]

    ax.scatter(trades_df['sell_date_num'], trades_df['pnl_pct'],
               c=colors, s=sizes, alpha=0.6, edgecolors='none')

    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

    # 统计标注
    win_count = (trades_df['pnl'] > 0).sum()
    total = len(trades_df)
    ax.text(0.02, 0.95, f'胜率: {win_count/total:.1%} ({win_count}/{total})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    ax.set_title('交易盈亏分布（按时间）', fontsize=13, fontweight='bold')
    ax.set_xlabel('交易序号', fontsize=10)
    ax.set_ylabel('收益率 (%)', fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(True, alpha=0.3)


def plot_factor_weights(ax, weights: dict):
    """因子权重饼图"""
    labels = list(weights.keys())
    values = list(weights.values())

    colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    explode = [0.05] * len(labels)

    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.0f%%',
        colors=colors_pie[:len(labels)],
        explode=explode,
        startangle=90,
        pctdistance=0.6,
    )

    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')

    ax.set_title('因子权重分配', fontsize=13, fontweight='bold')


# ============================================================
# 单图快捷函数
# ============================================================

def plot_equity_only(daily_stats, benchmark_stats=None,
                     title='策略净值曲线', save_path=None):
    """只画净值曲线（快捷版）"""
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_equity_curve(ax, daily_stats, benchmark_stats)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 测试：生成随机数据
    print("可视化模块加载成功")
    print("使用 plot_full_report() 生成完整图表")
