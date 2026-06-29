"""
补下载缺失的 382 只新股（2020年后上市的）
按实际上市日期作为起始日下载
"""
import sys, pandas as pd
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import init_tushare, get_pro, CACHE_DIR
from data_engine import download_daily

# 1. 找出缺失股票
pool = pd.read_csv(CACHE_DIR / "stock_pool_all_20260624.csv", dtype=str)
pool_codes = set(pool['ts_code'].tolist())

cached = set()
for f in CACHE_DIR.glob("*_qfq.csv"):
    code = f.name.split('_')[0]
    cached.add(code)

missing = sorted(pool_codes - cached)
print(f"缺失: {len(missing)} 只")

# 2. 获取上市日期
pro = get_pro()
listing_info = pro.stock_basic(
    exchange='',
    list_status='L',
    fields='ts_code,list_date'
)
listing_map = dict(zip(listing_info['ts_code'], listing_info['list_date']))

# 3. 逐个下载（从上市日起，到 2025-12-01）
end_date = '20251201'
success = 0
fail = 0

for i, code in enumerate(missing):
    list_date = listing_map.get(code, '20200101')
    if list_date < '20200101':
        list_date = '20200101'

    try:
        df = download_daily(code, list_date, end_date)
        if not df.empty:
            success += 1
        else:
            fail += 1
            print(f"  [{i+1}/{len(missing)}] {code} (上市{list_date}) 无数据")
    except Exception as e:
        fail += 1
        print(f"  [{i+1}/{len(missing)}] {code} 失败: {e}")

    if (i + 1) % 50 == 0:
        print(f"  进度: {i+1}/{len(missing)}  成功: {success}  失败: {fail}")

    time.sleep(0.3)  # 这些是新股，数据量小，可以快一点

print(f"\n完成: 成功 {success}, 失败 {fail}, 总计 {len(missing)}")
