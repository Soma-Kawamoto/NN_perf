import pandas as pd
import optuna

# --- 1. データの準備 ---
# storage_url と study_name はご自身の環境に合わせてください
storage_url = "sqlite:///docs/search_result.db"
study_name = "nn_hysteresis_study"

study = optuna.load_study(study_name=study_name, storage=storage_url)
df = study.trials_dataframe()

# 完了した試行から200件抽出し、累積最小値を計算
df_plot = df[df['state'] == 'COMPLETE'].head(200).copy()
df_plot['best_value'] = df_plot['value'].cummin()

# --- 2. Excelとグラフの作成 (xlsxwriterを使用) ---
file_name = "optuna_final_report.xlsx"
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
df_plot.to_excel(writer, sheet_name='Optimization_History', index=False)

workbook  = writer.book
worksheet = writer.sheets['Optimization_History']

# グラフオブジェクトの作成（散布図タイプ）
chart = workbook.add_chart({'type': 'scatter'})

# 最大行数の取得
max_row = len(df_plot)

# A列: number, B列: state, C列: value, ... (列番号は0始まり)
# trial_dataframeの列順によりますが、通常 'number'は0列目, 'value'は2列目です。
# 列がズレている場合は index_col を調整してください。

# 1. Trial Value (青い点) の設定
chart.add_series({
    'name':       'Trial Value',
    'categories': ['Optimization_History', 1, 0, max_row, 0], # number (A列)
    'values':     ['Optimization_History', 1, 2, max_row, 2], # value (C列)
    'marker':     {'type': 'circle', 'size': 5, 'border': {'color': '#1f77b4'}, 'fill': {'color': '#1f77b4'}},
    'line':       {'none': True},
})

# 2. Best Value (赤い線) の設定
# 'best_value'列のインデックスを探す
best_val_col = df_plot.columns.get_loc('best_value')
chart.add_series({
    'name':       'Best Value',
    'categories': ['Optimization_History', 1, 0, max_row, 0],
    'values':     ['Optimization_History', 1, best_val_col, max_row, best_val_col],
    'line':       {'color': '#d62728', 'width': 2.25},
    'marker':     {'type': 'none'},
})

# グラフのレイアウト設定
chart.set_title({'name': 'Optimization History'})
chart.set_x_axis({'name': 'Trial Number', 'major_gridlines': {'visible': True, 'line': {'dash_type': 'dash'}}})
chart.set_y_axis({'name': 'RMSE', 'major_gridlines': {'visible': True, 'line': {'dash_type': 'dash'}}})
chart.set_legend({'position': 'top'})
chart.set_size({'width': 720, 'height': 480})

# グラフをシートに挿入
worksheet.insert_chart('K2', chart)

writer.close()
print(f"✅ グラフ付きExcelを保存しました: {file_name}")