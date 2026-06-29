"""用 trade_cal + 其他方式获取行业数据"""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import init_tushare, get_pro

cache_dir = Path(__file__).parent.parent / "cache"
init_tushare()
pro = get_pro()

# 方法1: 尝试用 index_classify 获取行业分类
print("[1] Trying index_classify...")
try:
    df = pro.index_classify(level='L1', src='SW2021')
    print(f"    Got {len(df)} industry categories")
    print(df.head(20).to_string())
    df.to_csv(cache_dir / "industry_categories.csv", index=False)
except Exception as e:
    print(f"    Failed: {e}")

# 方法2: 获取申万行业指数成分股
print("\n[2] Trying SW index members...")
sw_indices = [
    '801010.SI', '801020.SI', '801030.SI', '801040.SI', '801050.SI',
    '801080.SI', '801110.SI', '801120.SI', '801130.SI', '801140.SI',
    '801150.SI', '801160.SI', '801170.SI', '801180.SI', '801200.SI',
    '801210.SI', '801230.SI', '801250.SI', '801260.SI', '801270.SI',
    '801280.SI', '801710.SI', '801720.SI', '801730.SI', '801740.SI',
    '801750.SI', '801760.SI', '801770.SI', '801780.SI', '801790.SI',
    '801880.SI', '801890.SI',
]

all_members = []
for idx in sw_indices[:5]:  # 先试5个，避免频率限制
    try:
        df_m = pro.index_member(index_code=idx,
                                fields='index_code,con_code,is_new')
        if df_m is not None and len(df_m) > 0:
            df_m['industry_code'] = idx
            all_members.append(df_m)
            print(f"    {idx}: {len(df_m)} members")
    except Exception as e:
        print(f"    {idx}: {e}")
        break

if all_members:
    result = pd.concat(all_members)
    result.to_csv(cache_dir / "industry_members.csv", index=False)
    print(f"\n    Total members: {len(result)}")
