"""用 stock_company + namechange 组合获取行业数据"""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import init_tushare, get_pro

init_tushare()
pro = get_pro()

# stock_company -- 看有哪些字段
print("[1] stock_company fields...")
for exchange in ['SSE', 'SZSE', 'BSE']:
    try:
        df = pro.stock_company(exchange=exchange, fields='ts_code,industry,employees,province')
        print(f"    {exchange}: {len(df)} stocks, cols={df.columns.tolist()}")
        print(f"    Sample:\n{df.head(3)}")
        # 保存
        df.to_csv(f'cache/company_{exchange}.csv', index=False)
    except Exception as e:
        print(f"    {exchange} FAIL: {str(e)[:100]}")

# namechange 批量 -- 改下用法
print("\n[2] namechange batch (all)...")
try:
    df = pro.namechange(fields='ts_code,name,start_date')
    print(f"    {len(df)} rows, cols={df.columns.tolist()}")
    # 取最新名称：按ts_code分组取start_date最大的
    latest = df.sort_values('start_date').groupby('ts_code').tail(1)
    print(f"    {len(latest)} unique stocks")
    print(latest.head(10))
    latest.to_csv('cache/stock_names.csv', index=False)
except Exception as e:
    print(f"    FAIL: {str(e)[:150]}")
