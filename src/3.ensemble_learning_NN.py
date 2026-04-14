#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全結合型ニューラルネットワーク (NN) によるアンサンブル学習（バギング）
B-H ヒステリシス回帰スクリプト
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import japanize_matplotlib
from datetime import datetime
import configparser
from typing import List, Tuple
import time

# --- プロット設定 ---
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

# ==============================================================================
# --- ユーザー設定（アンサンブル学習用） ---
# ==============================================================================
NUM_MODELS = 10  # アンサンブルするモデルの数（分類機の数）

# ==============================================================================
# --- 設定ファイルの読み込み ---
# ==============================================================================
try:
    # __file__ は現在のスクリプト(src/3.ensemble_learning_NN.py)の絶対パス
    script_dir = os.path.dirname(os.path.abspath(__file__)) # -> ~/NN_perf/src
    
    # 【修正ポイント】src/ に移動したので、1段階戻るだけでルート(NN_perf)に到達します
    base_dir = os.path.dirname(script_dir) # -> ~/NN_perf
except NameError:
    script_dir = os.getcwd()
    base_dir = os.path.dirname(script_dir)

print(f"📂 プロジェクトルートを特定しました: {base_dir}")

# 【修正ポイント】script_dir(src) から1つ戻って config フォルダを見るように変更
config_path = os.path.join(script_dir, "..", "config", "1. NN.ini")

# パスが本当に正しいか確認するためのデバッグ用表示
print(f"🔍 設定ファイルを読み込んでいます: {os.path.abspath(config_path)}")

if not os.path.exists(config_path):
    print(f"🔴 エラー: 指定されたパスに設定ファイルが見つかりません: {config_path}")
    exit()

config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

# [settings]
PERFORM_TRAINING = config.getboolean('settings', 'PERFORM_TRAINING', fallback=True)
mat_name = config.get('settings', 'mat_name')
target_freq = config.getint('settings', 'target_freq')
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

# [data/regression]
Bmtrain_min = config.getfloat('data', 'Bmtrain_min')
Bmtrain_max = config.getfloat('data', 'Bmtrain_max')
train_step = config.getfloat('data', 'train_step')
train_amp = list(np.round(np.arange(Bmtrain_min, Bmtrain_max + 1e-8, train_step), 1))
USE_AKIMA_DATA = config.getboolean('data', 'USE_AKIMA_DATA', fallback=True)

Bmreg_min = config.getfloat('regression', 'Bmreg_min')
Bmreg_max = config.getfloat('regression', 'Bmreg_max')
step = config.getfloat('regression', 'step')

# パス設定（Linux/WSL対応）
base_dir = os.path.dirname(script_dir)
akima_excel_path = os.path.join(
    base_dir,
    "2.Normal Magnetization Curve Extraction Folder", "assets", "2.Akima spline interpolation",
    f"Bm-Hb Curve_akima_{mat_name}_50hz.xlsx"
)
input_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "6.Downsampling")
output_base = os.path.join(base_dir, "3.Answer", "NN_ensemble_results")
model_dir = os.path.join(base_dir, "3.Answer", "NN_models_ensemble")
truth_data_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "7.reference data")
plot_output_dir = os.path.join(output_base, mat_name, str(target_freq), "plots")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_output_dir, exist_ok=True)

model_weights_base_path = os.path.join(model_dir, f"model_weights_{mat_name}_{target_freq}hz") # .pthの前に番号がつく
scaler_X_path = os.path.join(model_dir, f"scaler_X_{mat_name}_{target_freq}hz.pkl")
scaler_Y_path = os.path.join(model_dir, f"scaler_Y_{mat_name}_{target_freq}hz.pkl")
truth_data_path = os.path.join(truth_data_base, f"summary_{mat_name}_{step}.xlsx")

# --- デバイス決定 ---
device = torch.device("cuda" if (USE_GPU_SETTING and torch.cuda.is_available()) else "cpu")
print(f"使用デバイス: {device}")

# ==============================================================================
# --- クラス・関数定義 ---
# ==============================================================================
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)

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
    def forward(self, x): return self.network(x)

def get_activation_function(name):
    if name.lower() == 'relu': return nn.ReLU()
    elif name.lower() == 'tanh': return nn.Tanh()
    elif name.lower() == 'sigmoid': return nn.Sigmoid()
    else: return nn.ReLU()

# ==============================================================================
# --- データ読み込み ---
# ==============================================================================
print("\n学習データを読み込んでいます...")
X_list, Y_list = [], []
for amp in train_amp:
    path = os.path.join(input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
    if not os.path.exists(path): continue
    df = pd.read_excel(path, engine='openpyxl')
    B, H = df['B'].values, df['H'].values
    for b_val, h_val in zip(B, H): X_list.append([amp, b_val]); Y_list.append([h_val])

if USE_AKIMA_DATA:
    try:
        df_akima = pd.read_excel(akima_excel_path, sheet_name='Interpolated Data')
        # フィルタリング等の処理（省略可、ここでは簡易的に）
        Hb_a, Bm_a = df_akima['amp_Hb'].values, df_akima['amp_Bm'].values
        for h, b in zip(Hb_a, Bm_a):
            X_list.append([b, b]); Y_list.append([h])
            X_list.append([b, -b]); Y_list.append([-h])
    except Exception as e: print(f"Akimaデータ読込スキップ: {e}")

X_train_raw, Y_train_raw = np.array(X_list), np.array(Y_list)

# スケーリング
scaler_X, scaler_Y = StandardScaler(), StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
Y_train_scaled = scaler_Y.fit_transform(Y_train_raw)

# ==============================================================================
# --- モデル学習 (Ensemble / Bagging) ---
# ==============================================================================
models = []
if PERFORM_TRAINING:
    print(f"\n🚀 {NUM_MODELS} 個のモデルのアンサンブル学習を開始します...")
    
    for i in range(NUM_MODELS):
        print(f"--- Model {i+1}/{NUM_MODELS} Training ---")
        
        # 1. ブートストラップサンプリング (重複を許したランダム抽出)
        indices = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
        X_boot = torch.FloatTensor(X_train_scaled[indices]).to(device)
        Y_boot = torch.FloatTensor(Y_train_scaled[indices]).to(device)
        
        # 2. モデル構築
        act_fn = get_activation_function(activation_func_str)
        model = FullyConnectedNN(2, 1, HIDDEN_LAYERS, act_fn).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = RMSELoss() if LossFunc == 'RMSE' else nn.MSELoss()
        
        train_loader = DataLoader(TensorDataset(X_boot, Y_boot), batch_size=BATCH_SIZE, shuffle=True)
        
        # 3. 学習ループ
        model.train()
        for epoch in range(EPOCHS):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
            if (epoch+1) % 500 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")
        
        # 4. 保存
        torch.save(model.to('cpu').state_dict(), f"{model_weights_base_path}_{i}.pth")
        models.append(model)
        
    with open(scaler_X_path, 'wb') as f: pickle.dump(scaler_X, f)
    with open(scaler_Y_path, 'wb') as f: pickle.dump(scaler_Y, f)
else:
    print("\n✅ 保存済みのアンサンブルモデルを読み込みます...")
    with open(scaler_X_path, 'rb') as f: scaler_X = pickle.load(f)
    with open(scaler_Y_path, 'rb') as f: scaler_Y = pickle.load(f)
    for i in range(NUM_MODELS):
        m_path = f"{model_weights_base_path}_{i}.pth"
        act_fn = get_activation_function(activation_func_str)
        model = FullyConnectedNN(2, 1, HIDDEN_LAYERS, act_fn)
        model.load_state_dict(torch.load(m_path))
        model.to(device).eval()
        models.append(model)

# ==============================================================================
# --- 推論と結果出力 ---
# ==============================================================================
print("\nアンサンブル推論と統計計算を実行中...")
pred_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
comparison_sheets_data = []

# 正解データの読み込み（既存ロジック）
df_truth_all = pd.read_excel(truth_data_path, sheet_name=f"{target_freq}Hz", header=1)
truth_data_blocks = [group.reset_index(drop=True) for _, group in df_truth_all.dropna(how='all').groupby(df_truth_all.isnull().all(axis=1).cumsum())]

plt.figure(figsize=(10, 8))
plt.title(f"NN Ensemble Regression (n={NUM_MODELS})")

for i, amp in enumerate(pred_amps):
    num_points = int(round(2 * amp / step)) + 1
    Breg = np.linspace(-amp, amp, num_points)
    X_pred = np.array([[amp, b] for b in Breg])
    X_pred_scaled = torch.FloatTensor(scaler_X.transform(X_pred)).to(device)
    
    # --- 全モデルの予測を取得 ---
    all_model_outputs = []
    with torch.no_grad():
        for m in models:
            m.eval()
            out = m(X_pred_scaled).cpu().numpy()
            all_model_outputs.append(scaler_Y.inverse_transform(out).flatten())
    
    all_model_outputs = np.array(all_model_outputs) # [NUM_MODELS, num_points]
    
    # --- 統計量の計算 (Mean, Std) ---
    H_mean = np.mean(all_model_outputs, axis=0)
    H_std = np.std(all_model_outputs, axis=0)
    
    # --- 正解データの取得 ---
    h_ref, b_ref = np.full_like(H_mean, np.nan), np.full_like(H_mean, np.nan)
    if i < len(truth_data_blocks):
        df_ref = truth_data_blocks[i]
        if len(df_ref) == len(Breg):
            h_ref, b_ref = df_ref['H_descending'].values, df_ref['B'].values

    # --- Excel用のデータフレーム作成 (F列以降に±1~3σ) ---
    df_comp = pd.DataFrame({
        'H_mean [A/m]': H_mean,
        'B_reg [T]': Breg,
        'H_ref [A/m]': h_ref,
        'B_ref [T]': b_ref,
        ' ': '',  # E列（スペーサー）
        'H_+1sigma': H_mean + H_std,
        'H_-1sigma': H_mean - H_std,
        'H_+2sigma': H_mean + 2*H_std,
        'H_-2sigma': H_mean - 2*H_std,
        'H_+3sigma': H_mean + 3*H_std,
        'H_-3sigma': H_mean - 3*H_std
    })
    comparison_sheets_data.append({'amp': amp, 'df': df_comp})

    # --- プロット ---
    color = 'blue' if amp in train_amp else 'red'
    plt.plot(H_mean, Breg, color=color, label=f'Mean {amp:.1f}T' if i % 4 == 0 else "")
    # 3σの範囲を塗りつぶし
    plt.fill_betweenx(Breg, H_mean - 3*H_std, H_mean + 3*H_std, color=color, alpha=0.1)

plt.xlabel("H [A/m]"); plt.ylabel("B [T]"); plt.grid(True)
plt.savefig(os.path.join(plot_output_dir, "ensemble_regression.png"))
plt.show()

# --- Excel保存 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
rmse_out_path = os.path.join(output_base, mat_name, f"{timestamp}_Ensemble_Summary_{mat_name}.xlsx")
os.makedirs(os.path.dirname(rmse_out_path), exist_ok=True)

with pd.ExcelWriter(rmse_out_path, engine='openpyxl') as writer:
    for item in comparison_sheets_data:
        sheet_name = f"{item['amp']:.2f}T"
        item['df'].to_excel(writer, sheet_name=sheet_name, index=False)
        # グラフ追加関数などの呼び出し（任意）
        
print(f"\n✅ 全ての処理が完了しました。結果を保存しました:\n{rmse_out_path}")