import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_engine import init_tushare, get_pro
init_tushare()
df = get_pro().stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
df.to_csv('cache/stock_industry_full.csv', index=False)
print(f"OK: {len(df)} stocks, {df['industry'].nunique()} industries")
for i, c in df['industry'].value_counts().head(28).items():
    print(f"  {i}: {c}")
