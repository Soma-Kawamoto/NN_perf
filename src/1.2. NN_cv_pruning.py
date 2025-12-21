#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全結合型ニューラルネットワーク (NN) による B-H ヒステリシス回帰スクリプト
【v11修正版: USE_GPU_SETTINGによる明示的切り替え対応】
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
from sklearn.model_selection import train_test_split, KFold
import optuna
from typing import Any, Dict, List, Tuple
import time

# フォント設定
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

# ★★★ ここでGPU使用の有無を手動で切り替えられます ★★★
# iniファイルから読み込みますが、コード上で直接 False に書き換えることも可能です
USE_GPU_SETTING = config.getboolean('settings', 'USE_GPU', fallback=True)

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
LOAD_PARAMS_FROM_EXCEL = False  
PARAMS_EXCEL_PATH = "/mnt/c/Users/RM-2503-1/Desktop/M1/3_研究/NN_perf/3.Answer/NN_regression_results/50A470/20/20251103_142945_RMSE_summary_50A470_20hz_NN.xlsx"

# [data]
Bmtrain_min = config.getfloat('data', 'Bmtrain_min')
Bmtrain_max = config.getfloat('data', 'Bmtrain_max')
train_step = config.getfloat('data', 'train_step')
train_amp = list(np.round(np.arange(Bmtrain_min, Bmtrain_max + 1e-8, train_step), 1))

# ★★★ Akimaデータを学習データとして使用するかどうか ★★★
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

# ==============================================================================
# デバイスの決定ロジック (USE_GPU_SETTINGを尊重)
# ==============================================================================
def get_device(use_gpu_setting: bool) -> torch.device:
    if use_gpu_setting:
        if torch.cuda.is_available():
            current_device = torch.device("cuda")
            print(f"🚀 GPU (CUDA) を使用して計算を行います: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ GPU使用が設定されていますが、利用可能なGPUが見つかりません。CPUを使用します。")
            current_device = torch.device("cpu")
    else:
        current_device = torch.device("cpu")
        print("💻 設定により CPU を使用して計算を行います")
    return current_device

device = get_device(USE_GPU_SETTING)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)

def get_activation_function(name: str) -> nn.Module:
    if name.lower() == 'relu': return nn.ReLU()
    elif name.lower() == 'tanh': return nn.Tanh()
    elif name.lower() == 'sigmoid': return nn.Sigmoid()
    else: raise ValueError(f"未対応の活性化関数です: {name}")

class FullyConnectedNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int], activation_func: nn.Module = nn.ReLU()):
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
            "学習データ(振幅 T)", "Akimaデータ使用", "使用デバイス"
        ],
        "値": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mat_name, target_freq, str(HIDDEN_LAYERS), activation_func_str, LEARNING_RATE, EPOCHS, BATCH_SIZE, GRAD_CLIP, LossFunc,
            str(train_amp), str(USE_AKIMA_DATA), str(device)
        ]
    }
    if USE_AKIMA_DATA:
        info_data["項目"].append("Akimaデータファイル")
        info_data["値"].append(os.path.basename(akima_excel_path))
        
    if amp_value is not None:
        info_data["項目"].append("回帰対象振幅 (T)")
        info_data["値"].append(f"{amp_value:.2f}")
    return pd.DataFrame(info_data)

def load_hyperparams_from_excel(excel_path: str) -> Tuple[List[int], str, float, int, int, float]:
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
    if not os.path.exists(path):
        print(f"    -> 🔴 ファイルが見つかりません: {path} (スキップ)")
        data_points_per_amp[amp] = 0
        continue
    df = pd.read_excel(path, engine='openpyxl')
    B, H = df['B'].values, df['H'].values
    for b_val, h_val in zip(B, H): X_list.append([amp, b_val]); Y_list.append([h_val])

NUM_NORMAL_SAMPLES = len(X_list)
Hb_vals, Bm_vals = np.array([]), np.array([]) 

if USE_AKIMA_DATA:
    print(f"\nAkima補間データを読み込んでいます...")
    try:
        df_akima_full = pd.read_excel(akima_excel_path, sheet_name='Interpolated Data', engine='openpyxl')
        df_akima_full.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_akima_full.dropna(subset=['amp_Bm', 'amp_Hb'], inplace=True)
        
        target_regression_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
        mask = np.any([np.isclose(df_akima_full['amp_Bm'], amp, rtol=1e-5, atol=1e-2) for amp in target_regression_amps], axis=0)
        df_akima_filtered = df_akima_full[mask].copy()

        if 0.05 in target_regression_amps:
            has_005 = np.any(np.isclose(df_akima_filtered['amp_Bm'], 0.05, rtol=1e-5, atol=1e-2))
            if not has_005:
                idx_05 = df_akima_full['amp_Bm'].sub(0.05).abs().idxmin()
                if np.isclose(df_akima_full.loc[idx_05, 'amp_Bm'], 0.05, rtol=1e-5, atol=1e-2):
                    df_akima_filtered = pd.concat([df_akima_filtered, df_akima_full.loc[[idx_05]]]).drop_duplicates().reset_index(drop=True)

        print(f"  - Akimaデータ: {len(df_akima_filtered)}点の頂点データを使用します。")
        Hb_vals = df_akima_filtered['amp_Hb'].values
        Bm_vals = df_akima_filtered['amp_Bm'].values
        for Hb, Bm in zip(Hb_vals, Bm_vals):
            X_list.append([Bm, Bm]); Y_list.append([Hb])
            X_list.append([Bm, -Bm]); Y_list.append([-Hb])
    except FileNotFoundError:
        print(f"🔴 警告: Akimaデータファイルが見つかりません。")
else:
    print("\nAkimaデータを使用しません。")

X_train, Y_train = np.array(X_list), np.array(Y_list)
print(f"✅ 学習データ準備完了. 総数: {len(X_train)}")

if np.isnan(X_train).any() or np.isinf(X_train).any() or np.isnan(Y_train).any() or np.isinf(Y_train).any():
    print("🔴 エラー: データにNaN/infが含まれています"); exit()

# --- 学習データプロット ---
print("\n学習データをプロットして保存します...")
plt.figure(figsize=(8, 6))
for amp in train_amp:
    path = os.path.join(input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
    if os.path.exists(path):
        df = pd.read_excel(path, engine='openpyxl')
        plt.plot(df['H'], df['B'], marker='o', markersize=3, linestyle='-', alpha=0.4)
if USE_AKIMA_DATA and len(Bm_vals) > 0:
    plt.scatter(Hb_vals, Bm_vals, s=80, c='red', marker='o', zorder=5)
    plt.scatter(-Hb_vals, -Bm_vals, s=80, c='red', marker='o', zorder=5)
plt.xlabel('H [A/m]'); plt.ylabel('B [T]')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, f"training_data_distribution.png"))
# 画面に表示させたい場合は以下のコメントアウトを外してください
plt.show()

# --- スケーリング ---
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_Y.fit_transform(Y_train)

# --- 既存モデル読み込み判定 ---
should_load_model = not PERFORM_TRAINING
settings_match = False
if should_load_model:
    try:
        saved_info_df = pd.read_excel(model_info_path, sheet_name='Info')
        saved_settings = pd.Series(saved_info_df.値.values, index=saved_info_df.項目).to_dict()
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
            saved_use_akima == current_use_akima and
            os.path.exists(model_weights_path)):
            settings_match = True
    except Exception:
        settings_match = False

# ==============================================================================
# --- Optunaによるハイパーパラメータ最適化 (Pruning実装版) ---
# ==============================================================================
def objective(trial: optuna.trial.Trial) -> float:
    """Optunaの目的関数 (Pruning + Early Stopping + CV)"""
    lr = trial.suggest_float("lr", LR_RANGE[0], LR_RANGE[1], log=True)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_layers = [trial.suggest_int(f"n_units_l{i}", 32, 256) for i in range(n_layers)]
    activation_str = trial.suggest_categorical("activation", ["relu", "tanh"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    
    PATIENCE = 500 
    activation_func = get_activation_function(activation_str)
    criterion = RMSELoss() if LossFunc == 'RMSE' else nn.MSELoss()
    
    X_normal = X_train_scaled[:NUM_NORMAL_SAMPLES]
    Y_normal = Y_train_scaled[:NUM_NORMAL_SAMPLES]
    X_akima = X_train_scaled[NUM_NORMAL_SAMPLES:]
    Y_akima = Y_train_scaled[NUM_NORMAL_SAMPLES:]

    K_SPLITS = 5
    kf = KFold(n_splits=K_SPLITS, shuffle=True, random_state=42)
    fold_scores = [] 

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_normal)):
        X_t_norm, X_v_norm = X_normal[train_idx], X_normal[val_idx]
        Y_t_norm, Y_v_norm = Y_normal[train_idx], Y_normal[val_idx]
        X_t = np.concatenate([X_t_norm, X_akima], axis=0)
        Y_t = np.concatenate([Y_t_norm, Y_akima], axis=0)
        X_v = X_v_norm
        Y_v = Y_v_norm

        # デバイスへの転送
        X_train_tensor = torch.FloatTensor(X_t).to(device)
        Y_train_tensor = torch.FloatTensor(Y_t).to(device)
        X_val_tensor = torch.FloatTensor(X_v).to(device)
        Y_val_tensor = torch.FloatTensor(Y_v).to(device)

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        model = FullyConnectedNN(
            input_size=2, output_size=1, hidden_layers=hidden_layers, activation_func=activation_func
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss_in_fold = float('inf')
        no_improve_cnt = 0

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
            
            val_loss_val = val_loss.item()
            current_step = fold * EPOCHS + epoch
            trial.report(val_loss_val, current_step)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss_val < best_loss_in_fold:
                best_loss_in_fold = val_loss_val
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1
            
            if no_improve_cnt >= PATIENCE:
                break

        fold_scores.append(best_loss_in_fold)
    
    return np.mean(fold_scores)

# --- Optuna実行 ---
if PERFORM_OPTUNA:
    print("\n" + "="*70)
    print("Optunaによるハイパーパラメータ探索を開始します...")
    start_time = time.time()
    
    db_path = "/mnt/z/distributed_search_result.db" # 共有フォルダ等の環境に合わせる
    db_url = f"sqlite:///{db_path}"

    study_name = (
        f"nn_cv_({Bmtrain_min:.2f},{Bmtrain_max:.2f},{train_step:.2f})"
        f"_to_({Bmreg_min:.2f},{Bmreg_max:.2f},{step:.2f})"
        f"_Akima-{USE_AKIMA_DATA}"
    )
    
    # 既存のStudyがあるか確認し、無ければ新規作成
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1000)
    study = optuna.create_study(direction="minimize", storage=db_url, study_name=study_name, load_if_exists=True, pruner=pruner)
    
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*70)
    print(f"探索完了. 最良スコア (RMSE): {study.best_value}")
    best_params = study.best_params
    LEARNING_RATE = best_params['lr']
    HIDDEN_LAYERS = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]
    activation_func_str = best_params['activation']
    BATCH_SIZE = best_params['batch_size']
    PERFORM_TRAINING = True 
    settings_match = False 

# ==============================================================================
# --- モデル学習と結果出力 (Loss Plot追加) ---
# ==============================================================================
if not settings_match and PERFORM_TRAINING:
    print("\nモデルの最終学習を開始します...")
    ACTIVATION_FUNC = get_activation_function(activation_func_str)
    
    model = FullyConnectedNN(
        input_size=2, output_size=1, hidden_layers=HIDDEN_LAYERS, activation_func=ACTIVATION_FUNC
    ).to(device)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train_scaled).to(device)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    criterion = RMSELoss() if LossFunc == 'RMSE' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss): print("Loss is NaN"); exit()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}')
            
    print("📈 学習曲線(Loss)を保存しています...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_loss_history, label='Training Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss (RMSE)'); plt.grid(True, alpha=0.7); plt.legend()
    plt.savefig(os.path.join(plot_output_dir, "learning_curve.png"))

    # 保存時は汎用性のためにCPUに戻す
    torch.save(model.to('cpu').state_dict(), model_weights_path)
    with open(scaler_X_path, 'wb') as f: pickle.dump(scaler_X, f)
    with open(scaler_Y_path, 'wb') as f: pickle.dump(scaler_Y, f)
    create_info_df().to_excel(model_info_path, index=False)
    # 推論用に再度デバイスへ
    model.to(device)
else:
    print(f"\n✅ モデルを読み込みます")
    ACTIVATION_FUNC = get_activation_function(activation_func_str)
    model = FullyConnectedNN(input_size=2, output_size=1, hidden_layers=HIDDEN_LAYERS, activation_func=ACTIVATION_FUNC)
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    with open(scaler_X_path, 'rb') as f: scaler_X = pickle.load(f)
    with open(scaler_Y_path, 'rb') as f: scaler_Y = pickle.load(f)

# --- 回帰結果・Excel出力 ---
print("\n回帰結果を計算中...")
plt.figure(figsize=(10, 8))
plt.title(f'NN Regression vs Reference - {mat_name} {target_freq}Hz')

pred_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
train_amp_set = set(np.round(train_amp, 2)) 
rmse_results, comparison_sheets_data = [], []
model.eval()

with torch.no_grad():
    for i, amp in enumerate(pred_amps):
        num_points = int(round(2 * amp / step)) + 1
        if num_points <= 1: continue
        Breg = np.linspace(-amp, amp, num_points)
        X_pred = np.array([[amp, b] for b in Breg])
        X_pred_scaled = scaler_X.transform(X_pred)
        X_pred_tensor = torch.FloatTensor(X_pred_scaled).to(device)
        
        Hpred_scaled = model(X_pred_tensor)
        Hpred = scaler_Y.inverse_transform(Hpred_scaled.cpu().numpy()).flatten()
        
        if i < len(truth_data_blocks):
            df_truth_loop = truth_data_blocks[i]
            if np.allclose(Breg, df_truth_loop['B'].values, rtol=1e-5, atol=1e-2):
                h_true = df_truth_loop['H_descending'].values
                rmse = np.sqrt(np.mean((h_true - Hpred)**2))
                Hb_pred = Hpred[-1]
                rmse_results.append({'Amplitude (T)': amp, 'RMSE (H_descending)': rmse, 'Hb [A/m]': Hb_pred, 'RMSE/Hb': rmse/abs(Hb_pred) if Hb_pred!=0 else np.nan})
                plt.plot(h_true, df_truth_loop['B'].values, marker='.', color='gray', linestyle='none', alpha=0.3)
                comparison_sheets_data.append({'amp': amp, 'df': pd.DataFrame({'H_pred [A/m]': Hpred, 'B_reg [T]': Breg, 'H_ref [A/m]': h_true, 'B_ref [T]': df_truth_loop['B'].values})})

        color = 'blue' if amp in train_amp_set else 'red'
        plt.plot(Hpred, Breg, color=color, alpha=0.6)

plt.xlabel('H [A/m]'); plt.ylabel('B [T]'); plt.grid(True)
plt.savefig(os.path.join(plot_output_dir, f"regression_results.png"))
plt.show()

if rmse_results:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rmse_out_path = os.path.join(output_base, mat_name, str(target_freq), f"{timestamp}_RMSE_summary_{mat_name}_{target_freq}hz_NN.xlsx")
    with pd.ExcelWriter(rmse_out_path, engine='openpyxl') as writer:
        create_info_df().to_excel(writer, sheet_name='Info', index=False)
        pd.DataFrame(rmse_results).to_excel(writer, sheet_name='RMSE_Summary', index=False)
        for item in comparison_sheets_data:
            item['df'].to_excel(writer, sheet_name=f"{item['amp']:.2f}T", index=False)
    print(f"\n✅ 結果を保存しました: {rmse_out_path}")

print("\n全ての処理が完了しました.")