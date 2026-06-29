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
import subprocess
import sys
from pathlib import Path

script = Path(__file__).parent / "scripts" / "run.py"
result = subprocess.run([sys.executable, str(script)] + sys.argv[1:])
sys.exit(result.returncode)
