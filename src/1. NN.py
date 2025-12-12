#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全結合型ニューラルネットワーク (NN) による B-H ヒステリシス回帰スクリプト
【v8: Akimaデータ使用有無(USE_AKIMA_DATA)の切り替え機能を追加】
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn # type: ignore
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
import japanize_matplotlib
from openpyxl.drawing.image import Image as OpenpyxlImage
from datetime import datetime
import configparser
from sklearn.model_selection import train_test_split
import optuna
import time

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

# ==============================================================================
# --- 設定ファイルの読み込み ---
# ==============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

config_path = os.path.join(script_dir, "..", "config", "1. NN.ini")
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

# [settings]
PERFORM_TRAINING = config.getboolean('settings', 'PERFORM_TRAINING', fallback=True)
mat_name = config.get('settings', 'mat_name')
target_freq = config.getint('settings', 'target_freq')
PERFORM_OPTUNA = config.getboolean('settings', 'PERFORM_OPTUNA', fallback=False)
N_TRIALS = config.getint('settings', 'N_TRIALS', fallback=50)

# [architecture]
hidden_layers_str = config.get('architecture', 'HIDDEN_LAYERS')
HIDDEN_LAYERS = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]
activation_func_str = config.get('architecture', 'ACTIVATION_FUNC')

# [training]
LEARNING_RATE = config.getfloat('training', 'LEARNING_RATE')
EPOCHS = config.getint('training', 'EPOCHS')
BATCH_SIZE = config.getint('training', 'BATCH_SIZE')
GRAD_CLIP = config.getfloat('training', 'GRAD_CLIP')
LossFunc = config.get('training', 'LOSS_FUNC')

# [optuna_search_space]
lr_min = config.getfloat('optuna_search_space', 'lr_min', fallback=1e-5)
lr_max = config.getfloat('optuna_search_space', 'lr_max', fallback=1e-2)
LR_RANGE = [lr_min, lr_max]

# ★★★ 追加設定: 過去の結果Excelからハイパーパラメータをロードする場合 ★★★
LOAD_PARAMS_FROM_EXCEL = False  # Trueにすると、下のパスのExcelからパラメータを読み込みます
PARAMS_EXCEL_PATH = r"C:\Users\RM-2503-1\Desktop\M1\3_研究\NN_perf\3.Answer\NN_regression_results\50A470\20\20251103_142945_RMSE_summary_50A470_20hz_NN.xlsx"

# [data]
Bmtrain_min = config.getfloat('data', 'Bmtrain_min')
Bmtrain_max = config.getfloat('data', 'Bmtrain_max')
train_step = config.getfloat('data', 'train_step')
train_amp = list(np.round(np.arange(Bmtrain_min, Bmtrain_max + 1e-8, train_step), 1))

# ★★★ Akimaデータを学習データとして使用するかどうか (True/False) ★★★
# iniファイルに設定がない場合はデフォルトでTrueとします
USE_AKIMA_DATA = config.getboolean('data', 'USE_AKIMA_DATA', fallback=True)

# [regression]
Bmreg_min = config.getfloat('regression', 'Bmreg_min')
Bmreg_max = config.getfloat('regression', 'Bmreg_max')
step = config.getfloat('regression', 'step')

# ==============================================================================
# パス設定 & 関数定義
# ==============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
except NameError:
    script_dir = os.getcwd()
    base_dir = os.path.dirname(script_dir)

akima_excel_path = os.path.join(
    base_dir,
    "2.Normal Magnetization Curve Extraction Folder", "assets", "2.Akima spline interpolation",
    f"Bm-Hb Curve_akima_{mat_name}_50hz.xlsx"
)
input_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "6.Downsampling")
output_base = os.path.join(base_dir, "3.Answer", "NN_regression_results")
model_dir = os.path.join(base_dir, "3.Answer", "NN_models")
truth_data_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "7.reference data")

plot_output_dir = os.path.join(output_base, mat_name, str(target_freq), "plots")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_output_dir, exist_ok=True)

model_info_path = os.path.join(model_dir, f"model_info_{mat_name}_{target_freq}hz.xlsx")
model_weights_path = os.path.join(model_dir, f"model_weights_{mat_name}_{target_freq}hz.pth")
scaler_X_path = os.path.join(model_dir, f"scaler_X_{mat_name}_{target_freq}hz.pkl")
scaler_Y_path = os.path.join(model_dir, f"scaler_Y_{mat_name}_{target_freq}hz.pkl")
truth_data_path = os.path.join(truth_data_base, f"summary_{mat_name}_{step}.xlsx")

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)

def get_activation_function(name):
    if name.lower() == 'relu': return nn.ReLU()
    elif name.lower() == 'tanh': return nn.Tanh()
    elif name.lower() == 'sigmoid': return nn.Sigmoid()
    else: raise ValueError(f"未対応の活性化関数です: {name}")

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_func=nn.ReLU()):
        super(FullyConnectedNN, self).__init__()
        layers = []
        in_size = input_size
        for h_size in hidden_layers:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(activation_func)
            in_size = h_size
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

def create_info_df(amp_value=None):
    info_data = {
        "項目": [
            "実行日時", "材料名", "対象周波数 (Hz)",
            "NN隠れ層", "NN活性化関数", "NN学習率", "NNエポック数", "NNバッチサイズ", "NN勾配クリップ値", "NN損失関数",
            "学習データ(振幅 T)", "Akimaデータ使用"
        ],
        "値": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mat_name, target_freq, str(HIDDEN_LAYERS), activation_func_str, LEARNING_RATE, EPOCHS, BATCH_SIZE, GRAD_CLIP, LossFunc,
            str(train_amp), str(USE_AKIMA_DATA)
        ]
    }
    if USE_AKIMA_DATA:
        info_data["項目"].append("Akimaデータファイル")
        info_data["値"].append(os.path.basename(akima_excel_path))
        
    if amp_value is not None:
        info_data["項目"].append("回帰対象振幅 (T)")
        info_data["値"].append(f"{amp_value:.2f}")
    return pd.DataFrame(info_data)

def load_hyperparams_from_excel(excel_path):
    print(f"\n📂 Excelファイルからハイパーパラメータを読み込んでいます: {excel_path}")
    try:
        df_info = pd.read_excel(excel_path, sheet_name='Info', engine='openpyxl')
        params_dict = pd.Series(df_info.値.values, index=df_info.項目).to_dict()
        hidden_layers_str = str(params_dict.get('NN隠れ層'))
        hidden_layers = eval(hidden_layers_str)
        activation = str(params_dict.get('NN活性化関数'))
        lr = float(params_dict.get('NN学習率'))
        epochs = int(params_dict.get('NNエポック数'))
        batch_size = int(params_dict.get('NNバッチサイズ'))
        grad_clip = float(params_dict.get('NN勾配クリップ値', 1.0))
        print("✅ 読み込み成功:")
        print(f"  - Hidden Layers: {hidden_layers}")
        print(f"  - Activation: {activation}")
        print(f"  - LR: {lr}")
        print(f"  - Batch Size: {batch_size}")
        return hidden_layers, activation, lr, epochs, batch_size, grad_clip
    except Exception as e:
        print(f"🔴 エラー: パラメータの読み込みに失敗しました: {e}"); exit()

def add_comparison_chart_to_sheet(ws, df_len):
    chart = ScatterChart()
    chart.title = f"B-H Curve Comparison - {ws.title}"
    chart.x_axis.title = "H [A/m]"
    chart.y_axis.title = "B [T]"
    chart.style = 13
    max_row = df_len + 1
    x_pred = Reference(ws, min_col=1, min_row=2, max_row=max_row)
    y_pred = Reference(ws, min_col=2, min_row=2, max_row=max_row)
    series_pred = Series(y_pred, x_pred, title="NN Regression")
    series_pred.marker.symbol = "none"
    series_pred.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill="FF0000", w=12700))
    x_ref = Reference(ws, min_col=3, min_row=2, max_row=max_row)
    y_ref = Reference(ws, min_col=4, min_row=2, max_row=max_row)
    series_ref = Series(y_ref, x_ref, title="Reference")
    series_ref.marker.symbol = "none"
    series_ref.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill="0000FF", w=12700))
    chart.series.append(series_pred)
    chart.series.append(series_ref)
    if chart.legend: chart.legend.position = 'r'
    ws.add_chart(chart, "F2")

# ==============================================================================
# --- データ読み込みと前処理 ---
# ==============================================================================

# --- データ読み込み ---
print("RMSE比較のため、正解データを読み込んでいます...")
try:
    df_truth_all = pd.read_excel(truth_data_path, sheet_name=f"{target_freq}Hz", header=1)
    is_nan_row = df_truth_all.isnull().all(axis=1)
    block_id = is_nan_row.cumsum()
    truth_data_blocks = [group.reset_index(drop=True) for _, group in df_truth_all.dropna(how='all').groupby(block_id)]
    print(f"✅ 正解データを読み込みました。({len(truth_data_blocks)}個のヒステリシスループデータ)")
except FileNotFoundError:
    print(f"🔴 エラー: 正解データファイルが見つかりません: {truth_data_path}"); exit()
except Exception as e:
    print(f"🔴 エラー: 正解データファイルの読み込み中に問題が発生しました: {e}"); exit()

print("\nNNモデルの学習データを読み込んでいます...")
X_list, Y_list = [], []
data_points_per_amp = {}
X_list.append([0.0, 0.0]); Y_list.append([0.0])
for amp in train_amp:
    path = os.path.join(input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
    print(f"  - Reading Hysteresis Data: {path}")
    if not os.path.exists(path):
        print("    -> 🔴 ファイルが見つかりません。スキップします。")
        data_points_per_amp[amp] = 0
        continue
    df = pd.read_excel(path, engine='openpyxl')
    B, H = df['B'].values, df['H'].values
    for b_val, h_val in zip(B, H): X_list.append([amp, b_val]); Y_list.append([h_val])

# ★★★ Akimaデータの読み込み処理（USE_AKIMA_DATAフラグで分岐） ★★★
Hb_vals, Bm_vals = np.array([]), np.array([]) # プロット用に初期化

if USE_AKIMA_DATA:
    print(f"\nAkima補間データを読み込んでいます...")
    try:
        df_akima_full = pd.read_excel(akima_excel_path, sheet_name='Interpolated Data', engine='openpyxl')
        initial_rows = len(df_akima_full)
        df_akima_full.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_akima_full.dropna(subset=['amp_Bm', 'amp_Hb'], inplace=True)
        final_rows = len(df_akima_full)
        if initial_rows > final_rows:
            print(f"  - 警告: Akimaデータから{initial_rows - final_rows}個の無効な行（NaNまたはinf）を削除しました.")
        
        # Bmregの値に近い点を許容誤差でフィルタリング
        target_regression_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
        print(f"  - 対象の回帰振幅: {target_regression_amps}")
        mask = np.any([np.isclose(df_akima_full['amp_Bm'], amp, rtol=1e-5, atol=1e-2) for amp in target_regression_amps], axis=0)
        df_akima_filtered = df_akima_full[mask].copy()

        # 0.05Tの強制追加ロジック（対象範囲に含まれる場合のみ）
        if 0.05 in target_regression_amps:
            has_005 = np.any(np.isclose(df_akima_filtered['amp_Bm'], 0.05, rtol=1e-5, atol=1e-2))
            if not has_005:
                idx_05 = df_akima_full['amp_Bm'].sub(0.05).abs().idxmin()
                if np.isclose(df_akima_full.loc[idx_05, 'amp_Bm'], 0.05, rtol=1e-5, atol=1e-2):
                    df_akima_filtered = pd.concat([df_akima_filtered, df_akima_full.loc[[idx_05]]]).drop_duplicates().reset_index(drop=True)
                    print(f"    - 手動追加: 行 {idx_05+1} (Bm={df_akima_full.loc[idx_05, 'amp_Bm']:.6f})")

        print(f"  - Akimaデータ: {len(df_akima_filtered)}点の頂点データを学習データとして使用します。")
        print(f"  - 使用するBm値: {[round(x, 6) for x in sorted(df_akima_filtered['amp_Bm'].unique())]}")

        Hb_vals = df_akima_filtered['amp_Hb'].values
        Bm_vals = df_akima_filtered['amp_Bm'].values
        for Hb, Bm in zip(Hb_vals, Bm_vals):
            X_list.append([Bm, Bm]); Y_list.append([Hb])
            X_list.append([Bm, -Bm]); Y_list.append([-Hb])
    except FileNotFoundError:
        print(f"🔴 警告: Akimaデータファイルが見つかりません。Akimaデータなしで続行します。")
else:
    print("\nAkimaデータを使用しません（設定によりスキップ）。")


X_train, Y_train = np.array(X_list), np.array(Y_list)
print("✅ 学習データの読み込みが完了しました.")

if np.isnan(X_train).any() or np.isinf(X_train).any() or np.isnan(Y_train).any() or np.isinf(Y_train).any():
    print("🔴 エラー: 学習データにNaNまたはinfが含まれています。入力データを確認してください。"); exit()
else:
    print("✅ 学習データの値は正常です.")

# --- 学習データプロット ---
print("\n学習データをプロットしています...")
plt.figure(figsize=(8, 6))
for amp in train_amp:
    path = os.path.join(input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
    if not os.path.exists(path): continue
    df = pd.read_excel(path, engine='openpyxl')
    plt.plot(df['H'], df['B'], marker='o', markersize=3, linestyle='-', label=f'{amp:.1f} T Loop', color='royalblue', alpha=0.4)

plt.scatter(0, 0, s=80, c='black', marker='o', zorder=6, label='Origin Point')
if USE_AKIMA_DATA and len(Bm_vals) > 0:
    plt.scatter(Hb_vals, Bm_vals, s=80, c='red', marker='o', edgecolors='none', zorder=5, label='Akima Points (Used)')
    plt.scatter(-Hb_vals, -Bm_vals, s=80, c='red', marker='o', edgecolors='none', zorder=5)
plt.xlabel('H [A/m]'); plt.ylabel('B [T]')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plot_save_path = os.path.join(plot_output_dir, f"training_data_distribution.png")
plt.savefig(plot_save_path)
print(f"✅ 学習データのプロットをファイルに保存しました: {plot_save_path}")
print("▶️ 学習データのプロットを表示します。このウィンドウを閉じると、モデルの学習が始まります...")
plt.show()

# --- 学習データ点数プロット ---
print("\n学習データ点数をプロットしています...")
data_points_plot_path = os.path.join(plot_output_dir, "training_data_points_vs_amp.png")
if data_points_per_amp:
    amps = list(data_points_per_amp.keys())
    points = list(data_points_per_amp.values())
    plt.figure(figsize=(10, 6))
    plt.bar(amps, points, width=train_step*0.8, align='center', color='mediumseagreen', edgecolor='black')
    plt.title(f'学習データ点数 vs 磁束密度振幅 - {mat_name} {target_freq}Hz')
    plt.xlabel('磁束密度振幅 Bm [T]')
    plt.ylabel('学習データ点数')
    plt.xticks(amps, rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(data_points_plot_path)
    print(f"✅ 学習データ点数のプロットをファイルに保存しました: {data_points_plot_path}")
    plt.show()

# --- NNモデル構築と学習 ---
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
should_load_model = not PERFORM_TRAINING
settings_match = False
if should_load_model:
    try:
        saved_info_df = pd.read_excel(model_info_path, sheet_name='Info')
        saved_settings = pd.Series(saved_info_df.値.values, index=saved_info_df.項目).to_dict()
        
        # Akimaの使用有無も設定一致確認に含める
        saved_use_akima = str(saved_settings.get('Akimaデータ使用', 'True')) 
        current_use_akima = str(USE_AKIMA_DATA)
        
        if (str(saved_settings.get('材料名')) == str(mat_name) and
            int(saved_settings.get('対象周波数 (Hz)')) == int(target_freq) and
            str(saved_settings.get('NN隠れ層')) == str(HIDDEN_LAYERS) and
            str(saved_settings.get('NN活性化関数')) == str(activation_func_str) and
            float(saved_settings.get('NN学習率')) == float(LEARNING_RATE) and
            int(saved_settings.get('NNエポック数')) == int(EPOCHS) and
            int(saved_settings.get('NNバッチサイズ')) == int(BATCH_SIZE) and
            float(saved_settings.get('NN勾配クリップ値', 1.0)) == float(GRAD_CLIP) and
            str(saved_settings.get('学習データ(振幅 T)')) == str(train_amp) and
            saved_use_akima == current_use_akima and # ★ Akima設定の一致確認
            os.path.exists(model_weights_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_Y_path)):
            settings_match = True
    except Exception:
        settings_match = False
print("学習時間計算中")
# ==============================================================================
# --- Optunaによるハイパーパラメータ最適化 ---
# ==============================================================================
def objective(trial):
    """Optunaの目的関数"""
    lr = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_layers = [trial.suggest_int(f"n_units_l{i}", 32, 256) for i in range(n_layers)]
    activation_str = trial.suggest_categorical("activation", ["relu", "tanh"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    
    activation_func = get_activation_function(activation_str)
    model = FullyConnectedNN(input_size=2, output_size=1, hidden_layers=hidden_layers, activation_func=activation_func)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RMSELoss() if LossFunc == 'RMSE' else nn.MSELoss()

    X_t, X_v, Y_t, Y_v = train_test_split(X_train_scaled, Y_train_scaled, test_size=0.2, random_state=42)
    X_train_tensor = torch.FloatTensor(X_t)
    Y_train_tensor = torch.FloatTensor(Y_t)
    X_val_tensor = torch.FloatTensor(X_v)
    Y_val_tensor = torch.FloatTensor(Y_v)
    
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss): return float('inf')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, Y_val_tensor)

    return val_loss.item()

if PERFORM_OPTUNA:
    print("\n" + "="*70)
    print("Optunaによるハイパーパラメータ探索を開始します...")
    start_time = time.time()
    print(f"試行回数: {N_TRIALS}")
    print("="*70)

    db_url = "sqlite:///search_result.db"
    study_name = "nn_hysteresis_study" 

    study = optuna.create_study(
        direction="minimize", 
        storage=db_url, 
        study_name=study_name, 
        load_if_exists=True
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*70)
    print("Optunaによる探索が完了しました。")
    end_time = time.time()
    print("最適化に要した時間は", end_time - start_time, "秒です")
    print(f"最良スコア (検証RMSE): {study.best_value}")
    print("最適なハイパーパラメータ:")
    print(study.best_params)
    print("="*70)

    best_params = study.best_params
    LEARNING_RATE = best_params['lr']
    HIDDEN_LAYERS = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    activation_func_str = best_params['activation']
    BATCH_SIZE = best_params['batch_size']
    PERFORM_TRAINING = True 
    settings_match = False 

# ==============================================================================
# --- (追加) Excelからのパラメータ読み込みと適用 ---
# ==============================================================================
if not PERFORM_OPTUNA and LOAD_PARAMS_FROM_EXCEL and os.path.exists(PARAMS_EXCEL_PATH):
    loaded_hidden, loaded_act, loaded_lr, loaded_epochs, loaded_batch, loaded_clip = load_hyperparams_from_excel(PARAMS_EXCEL_PATH)
    HIDDEN_LAYERS = loaded_hidden
    activation_func_str = loaded_act 
    if "ReLU" in activation_func_str: activation_func_str = "relu"
    elif "Tanh" in activation_func_str: activation_func_str = "tanh"
    elif "Sigmoid" in activation_func_str: activation_func_str = "sigmoid"
    LEARNING_RATE = loaded_lr
    EPOCHS = loaded_epochs
    BATCH_SIZE = loaded_batch
    GRAD_CLIP = loaded_clip
    print("🔄 ハイパーパラメータをExcelの値で上書きしました。これを使って学習を行います。")
    PERFORM_TRAINING = True 
    settings_match = False 

# ==============================================================================
# --- モデル学習と結果出力 ---
# ==============================================================================

if not settings_match and PERFORM_TRAINING:
    print("\nモデルの学習を開始します...")
    ACTIVATION_FUNC = get_activation_function(activation_func_str)
    model = FullyConnectedNN(input_size=2, output_size=1, hidden_layers=HIDDEN_LAYERS, activation_func=ACTIVATION_FUNC)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    Y_train_tensor = torch.FloatTensor(Y_train_scaled)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = RMSELoss() if LossFunc == 'RMSE' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                print(f"🔴 エラー: Epoch {epoch+1}で損失がnanになりました。学習を停止します。"); exit()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}')
            
    print("✅ 学習が完了しました。")
    torch.save(model.state_dict(), model_weights_path)
    with open(scaler_X_path, 'wb') as f: pickle.dump(scaler_X, f)
    with open(scaler_Y_path, 'wb') as f: pickle.dump(scaler_Y, f)
    info_df = create_info_df()
    with pd.ExcelWriter(model_info_path, engine='openpyxl') as writer:
        info_df.to_excel(writer, sheet_name='Info', index=False)
    print(f"✅ 学習済みモデルと設定情報を保存しました:\n {model_dir}")
else:
    print(f"\n✅ 設定が一致したため、保存済みのモデルを読み込みます:\n {model_dir}")
    ACTIVATION_FUNC = get_activation_function(activation_func_str)
    model = FullyConnectedNN(input_size=2, output_size=1, hidden_layers=HIDDEN_LAYERS, activation_func=ACTIVATION_FUNC)
    model.load_state_dict(torch.load(model_weights_path))
    with open(scaler_X_path, 'rb') as f: scaler_X = pickle.load(f)
    with open(scaler_Y_path, 'rb') as f: scaler_Y = pickle.load(f)
    print("✅ モデルとスケーラーの読み込みが完了しました.")

# --- 結果プロット、Excel出力、およびRMSE計算 ---
print("\n回帰結果を計算し、出力しています...")
plt.figure(figsize=(10, 8))
plt.title(f'NN Regression vs Reference - {mat_name} {target_freq}Hz')
if USE_AKIMA_DATA and len(Bm_vals) > 0:
    plt.scatter(Hb_vals, Bm_vals, marker='x', c='k', s=50, zorder=3, label='Akima (Train)')
    plt.scatter(-Hb_vals, -Bm_vals, marker='x', c='k', s=50, zorder=3, label='_nolegend_')

pred_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
train_amp_set = set(np.round(train_amp, 2)) 

plt.plot([], [], color='red', linestyle='-', label='Extrapolation (Not in Train)')
plt.plot([], [], color='blue', linestyle='-', alpha=0.6, label='Interpolation (In Train)')

rmse_results = []
comparison_sheets_data = []
model.eval()
with torch.no_grad():
    for i, amp in enumerate(pred_amps):
        num_points = int(round(2 * amp / step)) + 1
        if num_points <= 1: 
            print(f"   Bm = {amp:.2f}T, 点数が1以下なのでスキップします。")
            continue
        
        Breg = np.linspace(-amp, amp, num_points)
        X_pred = np.array([[amp, b] for b in Breg])
        X_pred_scaled = scaler_X.transform(X_pred)
        X_pred_tensor = torch.FloatTensor(X_pred_scaled)
        Hpred_scaled = model(X_pred_tensor)
        Hpred_means = scaler_Y.inverse_transform(Hpred_scaled.numpy())
        Hpred = Hpred_means.flatten()
        has_ref_data = False
        
        if i < len(truth_data_blocks):
            df_truth_loop = truth_data_blocks[i]
            if 'B' in df_truth_loop.columns and 'H_descending' in df_truth_loop.columns and np.allclose(Breg, df_truth_loop['B'].values, rtol=1e-5, atol=1e-2):
                has_ref_data = True
                h_true_desc = df_truth_loop['H_descending'].values
                b_true = df_truth_loop['B'].values
                Hb_pred = Hpred[-1]
                rmse = np.sqrt(np.mean((h_true_desc - Hpred)**2))
                relative_rmse = rmse / abs(Hb_pred) if Hb_pred != 0 else np.nan
                rmse_results.append({'Amplitude (T)': amp, 'RMSE (H_descending)': rmse, 'Hb [A/m]': Hb_pred, 'RMSE/Hb': relative_rmse})
                print(f"   Bm = {amp:.2f}T, RMSE = {rmse:.4f}, Hb = {Hb_pred:.2f}, RMSE/Hb = {relative_rmse:.4%}")
                
                label_ref = f'Ref {amp:.2f}T' if amp in [pred_amps.min(), 1.0, pred_amps.max()] else None
                plt.plot(h_true_desc, b_true, marker='.', color='gray', linestyle='none', markersize=5, zorder=1, label=label_ref)
                
                df_comp = pd.DataFrame({
                    'H_pred [A/m]': Hpred, 'B_reg [T]': Breg,
                    'H_ref [A/m]': h_true_desc, 'B_ref [T]': b_true
                })
                comparison_sheets_data.append({'amp': amp, 'df': df_comp})
            else:
                print(f"   Bm = {amp:.2f}T, 警告: 正解データとB軸の点が一致しないためRMSE計算と参照プロットをスキップします。")
        else:
             print(f"   Bm = {amp:.2f}T, 警告: 対応する正解データブロックが見つかりません。")

        if amp in train_amp_set:
            plt.plot(Hpred, Breg, color='blue', linestyle='-', zorder=2, alpha=0.6) 
        else:
            plt.plot(Hpred, Breg, color='red', linestyle='-', zorder=2) 

plt.xlabel(r'$\it{H}$ [A/m]'); plt.ylabel(r'$\it{B}$ [T]'); 
plt.grid(True)
plt.legend()
plot_save_path_results = os.path.join(plot_output_dir, f"regression_results.png")
plt.savefig(plot_save_path_results)
print(f"✅ 回帰結果のプロットをファイルに保存しました: {plot_save_path_results}")
plt.show()

# --- Excelへの書き込み ---
if rmse_results:
    print("\n" + "="*70)
    print("RMSE 計算結果サマリー")
    print("="*70)
    df_rmse = pd.DataFrame(rmse_results)
    print(df_rmse.to_string(index=False))
    
    final_output_dir = os.path.join(output_base, mat_name, str(target_freq))
    os.makedirs(final_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_RMSE_summary_{mat_name}_{target_freq}hz_NN.xlsx"
    rmse_out_path = os.path.join(final_output_dir, filename)
    
    if not PERFORM_TRAINING and settings_match:
        try:
            pattern = f"_RMSE_summary_{mat_name}_{target_freq}hz_NN.xlsx"
            for f in os.listdir(final_output_dir):
                if f.endswith(pattern):
                    print(f"\n🔄 既存の古い結果ファイルを削除します: {f}")
                    os.remove(os.path.join(final_output_dir, f))
                    break
        except Exception as e:
            print(f"\n⚠️ 既存ファイルの削除に失敗しました: {e}.")
            
    print(f"\n結果をExcelファイルに保存します:\n {rmse_out_path}")
    try:
        with pd.ExcelWriter(rmse_out_path, engine='openpyxl') as writer:
            info_df = create_info_df()
            info_df.to_excel(writer, sheet_name='Info', index=False)
            
            df_rmse.to_excel(writer, sheet_name='RMSE_Summary', index=False)
            
            for item in comparison_sheets_data:
                amp, df_data = item['amp'], item['df']
                sheet_name = f"{amp:.2f}T"
                df_data.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                add_comparison_chart_to_sheet(ws, len(df_data))
                
        print(f"\n✅ 結果を保存しました.")
    except PermissionError:
        print(f"\n🔴 保存エラー: ファイルへのアクセスが拒否されました。'{os.path.basename(rmse_out_path)}'が開かれていないか確認してください。")
    except Exception as e:
        print(f"🔴 Excelファイルへの書き込み中に予期せぬエラーが発生しました: {e.__class__.__name__}: {e}")
else:
    print("\n⚠️ RMSE結果が計算されませんでした。Excelへの出力をスキップします。")

print("\n全ての処理が完了しました.")