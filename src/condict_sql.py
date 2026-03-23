import sqlite3
import pandas as pd

# データベースに接続
conn = sqlite3.connect("For_Conference_Presentation.db")

# SQLでデータを一括取得
query = """
SELECT t.trial_id, v.value as rmse, p.param_name, p.param_value_actual
FROM trials t
JOIN trial_values v ON t.trial_id = v.trial_id
JOIN trial_params p ON t.trial_id = p.trial_id
WHERE t.state = 'COMPLETE'
"""
df = pd.read_sql_query(query, conn)
conn.close()

print(df.head())