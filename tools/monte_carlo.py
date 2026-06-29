"""
===========================================================
蒙特卡洛模拟 — 验证策略收益是否显著优于随机
===========================================================
原理：
  对策略的日收益率序列随机打乱 N 次（破坏时序结构但保留分布特征），
  每次重构净值曲线。如果策略有真实 Alpha，实际表现应显著优于
  随机打乱后的表现分布。

方法：
  1. 取策略日收益率序列
  2. 随机打乱 500 次
  3. 每次重构净值 → 计算 Sharpe / CAGR / MaxDD
  4. 看实际表现在随机分布中的分位数位置 → 得出 p-value

用法：
  python tools/monte_carlo.py                     # 读取 output/daily_equity.csv
  python tools/monte_carlo.py --file my_stats.csv # 自定义输入

解释：
  - 如果实际 Sharpe 在随机分布的 95 分位以上 → p < 0.05 → 策略有统计显著的超额
  - 如果实际 Sharpe 在 50 分位附近 → 策略可能只是运气
  - 这个检验的本质：H0 = 日收益无序列相关性（纯噪声），
                      Ha = 日收益有时序结构（有 Alpha）

面试考点：
  Q: 蒙特卡洛模拟和传统的 t 检验有什么区别？
  A: t 检验假设收益率服从正态分布，但金融收益率通常偏态+厚尾。
     蒙特卡洛不做分布假设，直接用重抽样（bootstrap）做统计推断，
     更适合金融数据的非正态特征。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def monte_carlo_simulation(returns: np.ndarray,
                           n_simulations: int = 500,
                           initial_capital: float = 1_000_000,
                           risk_free_rate: float = 0.03,
                           trading_days: int = 252,
                           random_seed: int = 42) -> dict:
    """
    对收益率序列做蒙特卡洛模拟

    参数:
        returns:            日收益率数组
        n_simulations:      模拟次数
        initial_capital:    初始资金
        risk_free_rate:     无风险利率
        trading_days:       年交易日数
        random_seed:        随机种子（可复现）

    返回:
        dict: {
            'actual_metrics': {...},       # 实际指标
            'simulated_metrics': [...],    # 每次模拟的指标
            'p_value_sharpe': float,       # Sharpe p-value
            'p_value_cagr': float,         # CAGR p-value
            'is_significant': bool,        # 是否在95%置信水平显著
        }
    """
    np.random.seed(random_seed)
    n = len(returns)

    # —— 实际指标 ——
    actual_cagr, actual_vol, actual_sharpe, actual_maxdd = \
        _calc_metrics(returns, risk_free_rate, trading_days)

    # —— 随机打乱模拟 ——
    sim_sharpes = []
    sim_cagrs = []
    sim_maxdds = []

    for _ in range(n_simulations):
        shuffled = np.random.permutation(returns)
        cagr, vol, sharpe, maxdd = _calc_metrics(
            shuffled, risk_free_rate, trading_days
        )
        sim_cagrs.append(cagr)
        sim_sharpes.append(sharpe)
        sim_maxdds.append(maxdd)

    sim_sharpes = np.array(sim_sharpes)
    sim_cagrs = np.array(sim_cagrs)
    sim_maxdds = np.array(sim_maxdds)

    # —— 统计推断 ——
    # p-value = 模拟值优于实际值的比例（单侧检验：实际好=显著）
    # 对于 Sharpe/CAGR：越高越好 → p = 模拟中比实际高的比例
    # 对于 MaxDD：越低越好（绝对值越小越好）
    p_sharpe = (sim_sharpes >= actual_sharpe).mean()
    p_cagr = (sim_cagrs >= actual_cagr).mean()
    p_maxdd = (sim_maxdds <= actual_maxdd).mean()  # MaxDD越小越好

    is_significant = p_sharpe < 0.05

    return {
        'actual_metrics': {
            'cagr': actual_cagr,
            'volatility': actual_vol,
            'sharpe': actual_sharpe,
            'max_dd': actual_maxdd,
        },
        'simulated_metrics': {
            'sharpes': sim_sharpes,
            'cagrs': sim_cagrs,
            'maxdds': sim_maxdds,
        },
        'p_value_sharpe': p_sharpe,
        'p_value_cagr': p_cagr,
        'p_value_maxdd': p_maxdd,
        'is_significant': is_significant,
        'n_simulations': n_simulations,
        'n_days': n,
    }


def _calc_metrics(returns: np.ndarray, rf: float, td: int):
    """从收益率序列计算核心指标"""
    equity = (1 + pd.Series(returns)).cumprod().values
    total_return = equity[-1] - 1
    n_years = len(returns) / td
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = np.std(returns) * np.sqrt(td)
    sharpe = (cagr - rf) / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    return cagr, annual_vol, sharpe, max_dd


def plot_monte_carlo(results: dict, save_path: str = None):
    """
    画蒙特卡洛模拟结果图（3面板）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ('Sharpes', 'sharpe', '夏普比率', results['simulated_metrics']['sharpes'],
         results['actual_metrics']['sharpe'], results['p_value_sharpe']),
        ('CAGRs', 'cagr', '年化收益率', results['simulated_metrics']['cagrs'],
         results['actual_metrics']['cagr'], results['p_value_cagr']),
        ('MaxDDs', 'maxdd', '最大回撤', results['simulated_metrics']['maxdds'],
         results['actual_metrics']['max_dd'], results['p_value_maxdd']),
    ]

    for ax, (title, key, label, sim_vals, actual_val, p_val) in zip(axes, metrics):
        ax.hist(sim_vals, bins=40, color='#7f7f7f', alpha=0.6, edgecolor='white')
        ax.axvline(actual_val, color='#d62728', linewidth=2.5,
                   label=f'实际: {actual_val:.3f}')
        ax.axvline(np.median(sim_vals), color='#1f77b4', linewidth=1.5,
                   linestyle='--', label=f'随机中位数: {np.median(sim_vals):.3f}')

        ax.set_title(f'{label} 分布 (p={p_val:.3f}{" ✓" if p_val < 0.05 else ""})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('频次', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'蒙特卡洛模拟 (N={results["n_simulations"]}, '
                 f'{"统计显著 ✅" if results["is_significant"] else "不显著 ⚠️"})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[MC] 图表已保存: {save_path}")
    else:
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        path = output_dir / "monte_carlo_report.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[MC] 图表已保存: {path}")


def print_report(results: dict):
    """打印蒙特卡洛模拟报告"""
    print(f"\n{'='*60}")
    print(f"  蒙特卡洛模拟报告")
    print(f"{'='*60}")
    print(f"  模拟次数: {results['n_simulations']}")
    print(f"  样本天数: {results['n_days']}")
    print()

    actual = results['actual_metrics']
    sim = results['simulated_metrics']

    print(f"  {'指标':<20} {'实际值':>10} {'随机均值':>10} {'随机中位':>10} {'p-value':>10} {'显著':>6}")
    print(f"  {'─'*60}")

    for name, key, actual_val, sim_vals, p_val in [
        ('Sharpe', 'sharpe', actual['sharpe'], sim['sharpes'], results['p_value_sharpe']),
        ('CAGR', 'cagr', actual['cagr'], sim['cagrs'], results['p_value_cagr']),
        ('MaxDD', 'maxdd', actual['max_dd'], sim['maxdds'], results['p_value_maxdd']),
    ]:
        sig = '✅' if p_val < 0.05 else ('⚠️' if p_val < 0.10 else '❌')
        print(f"  {name:<20} {actual_val:>10.3f} {np.mean(sim_vals):>10.3f} "
              f"{np.median(sim_vals):>10.3f} {p_val:>10.3f} {sig:>6}")

    print(f"\n  结论: {'策略超额收益统计显著 (p<0.05)' if results['is_significant'] else '策略超额收益不显著，需进一步验证'}")

    # 解释
    print(f"\n  解读:")
    print(f"    p < 0.05 → 策略 Sharpe 在随机打乱分布的前 5%，有真实 Alpha")
    print(f"    p = 0.5 → 策略表现与随机无异，可能过拟合")
    print(f"    这个检验的本质：H0 = 日收益率无时序结构（纯噪声）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='蒙特卡洛模拟 — 策略显著性检验')
    parser.add_argument('--file', default=None,
                        help='daily_equity.csv 路径 (默认: output/daily_equity.csv)')
    parser.add_argument('--sims', type=int, default=500,
                        help='模拟次数 (默认: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    args = parser.parse_args()

    # —— 加载数据 ——
    if args.file:
        equity_file = Path(args.file)
    else:
        equity_file = Path(__file__).parent.parent / "output" / "daily_equity.csv"

    if not equity_file.exists():
        print(f"[错误] 找不到文件: {equity_file}")
        print(f"  请先运行 python run.py 生成回测结果")
        exit(1)

    df = pd.read_csv(equity_file, index_col=0, parse_dates=True)
    if 'returns' not in df.columns:
        df['returns'] = df['equity'].pct_change()
    returns = df['returns'].dropna().values

    if len(returns) < 60:
        print(f"[错误] 收益率数据不足 (只有{len(returns)}天)")
        exit(1)

    print(f"加载数据: {len(returns)} 个交易日")
    print(f"  日期范围: {df.index[0]} ~ {df.index[-1]}")

    # —— 运行模拟 ——
    print(f"\n运行蒙特卡洛模拟 ({args.sims} 次)...")
    results = monte_carlo_simulation(
        returns,
        n_simulations=args.sims,
        random_seed=args.seed
    )

    # —— 报告 ——
    print_report(results)

    # —— 图表 ——
    plot_monte_carlo(results)
    plt.show()
