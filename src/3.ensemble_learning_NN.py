#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全結合型ニューラルネットワーク (NN) によるアンサンブル学習（バギング）
B-H ヒステリシス回帰スクリプト (GPRフォーマット互換・完全版)
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

# --- プロット設定 ---
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

# ==============================================================================
# --- ユーザー設定 ---
# ==============================================================================
NUM_MODELS = 10  # アンサンブルするモデルの数 (n)

# ==============================================================================
# --- 設定ファイルの読み込み ---
# ==============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
except NameError:
    script_dir = os.getcwd()
    base_dir = os.path.dirname(script_dir)

print(f"📂 プロジェクトルート: {base_dir}")
config_path = os.path.join(base_dir, "config", "1. NN.ini")

if not os.path.exists(config_path):
    print(f"🔴 エラー: 設定ファイルが見つかりません: {config_path}")
    exit()

config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

mat_name = config.get('settings', 'mat_name')
target_freq = config.getint('settings', 'target_freq')
PERFORM_TRAINING = config.getboolean('settings', 'PERFORM_TRAINING', fallback=True)
USE_GPU_SETTING = config.getboolean('settings', 'USE_GPU', fallback=True)

hidden_layers_str = config.get('architecture', 'HIDDEN_LAYERS')
HIDDEN_LAYERS = [int(x.strip()) for x in hidden_layers_str.split(',') if x.strip()]
activation_func_str = config.get('architecture', 'ACTIVATION_FUNC')

LEARNING_RATE = config.getfloat('training', 'LEARNING_RATE')
EPOCHS = config.getint('training', 'EPOCHS')
BATCH_SIZE = config.getint('training', 'BATCH_SIZE')
GRAD_CLIP = config.getfloat('training', 'GRAD_CLIP')

Bmtrain_min = config.getfloat('data', 'Bmtrain_min')
Bmtrain_max = config.getfloat('data', 'Bmtrain_max')
train_step = config.getfloat('data', 'train_step')
train_amp = list(np.round(np.arange(Bmtrain_min, Bmtrain_max + 1e-8, train_step), 1))
USE_AKIMA_DATA = config.getboolean('data', 'USE_AKIMA_DATA', fallback=True)

Bmreg_min = config.getfloat('regression', 'Bmreg_min')
Bmreg_max = config.getfloat('regression', 'Bmreg_max')
step = config.getfloat('regression', 'step')

# --- パス設定 ---
akima_excel_path = os.path.join(base_dir, "2.Normal Magnetization Curve Extraction Folder", "assets", "2.Akima spline interpolation", f"Bm-Hb Curve_akima_{mat_name}_50hz.xlsx")
input_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "6.Downsampling")
output_base = os.path.join(base_dir, "3.Answer", "NN_ensemble_results")
model_dir = os.path.join(base_dir, "3.Answer", "NN_models_ensemble")
truth_data_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "7.reference data")
plot_output_dir = os.path.join(output_base, mat_name, str(target_freq), "plots")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_output_dir, exist_ok=True)

model_weights_base = os.path.join(model_dir, f"weights_{mat_name}_{target_freq}hz")
scaler_X_path = os.path.join(model_dir, f"scaler_X_{mat_name}_{target_freq}hz.pkl")
scaler_Y_path = os.path.join(model_dir, f"scaler_Y_{mat_name}_{target_freq}hz.pkl")
truth_data_path = os.path.join(truth_data_base, f"summary_{mat_name}_{step}.xlsx")

# --- ファイル名生成 ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{timestamp}_Ensemble_Summary_{mat_name}_{target_freq}hz_NN.xlsx"
rmse_out_path = os.path.join(output_base, mat_name, str(target_freq), output_filename)

device = torch.device("cuda" if (USE_GPU_SETTING and torch.cuda.is_available()) else "cpu")
print(f"💻 使用デバイス: {device}")

# ==============================================================================
# --- クラス・関数定義 ---
# ==============================================================================
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self, yhat, y): return torch.sqrt(self.mse(yhat, y) + self.eps)

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

def create_info_df():
    info_data = {
        "項目": [
            "実行日時", "材料名", "対象周波数 (Hz)", "NN隠れ層", "NN活性化関数",
            "NN学習率", "NNエポック数", "NNバッチサイズ", "学習データ(振幅 T)", "Akimaデータ使用",
            "アンサンブルモデル数(n)", "結果出力ファイル名"
        ],
        "値": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mat_name, target_freq, str(HIDDEN_LAYERS), activation_func_str,
            LEARNING_RATE, EPOCHS, BATCH_SIZE, str(train_amp), str(USE_AKIMA_DATA),
            NUM_MODELS, output_filename
        ]
    }
    return pd.DataFrame(info_data)

# ==============================================================================
# --- データ読み込み ---
# ==============================================================================
X_list, Y_list = [], []
for amp in train_amp:
    path = os.path.join(input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
    if not os.path.exists(path): continue
    df = pd.read_excel(path, engine='openpyxl')
    for b, h in zip(df['B'].values, df['H'].values): X_list.append([amp, b]); Y_list.append([h])

if USE_AKIMA_DATA:
    try:
        df_akima = pd.read_excel(akima_excel_path, sheet_name='Interpolated Data')
        target_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
        df_akima_f = df_akima[np.round(df_akima['amp_Bm'], 2).isin(target_amps)]
        for h, b in zip(df_akima_f['amp_Hb'].values, df_akima_f['amp_Bm'].values):
            X_list.append([b, b]); Y_list.append([h])
            X_list.append([b, -b]); Y_list.append([-h])
    except: pass

X_train_raw, Y_train_raw = np.array(X_list), np.array(Y_list)
scaler_X, scaler_Y = StandardScaler(), StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
Y_train_scaled = scaler_Y.fit_transform(Y_train_raw)

# ==============================================================================
# --- 学習 ---
# ==============================================================================
models = []
if PERFORM_TRAINING:
    print(f"\n🚀 {NUM_MODELS} 個のモデルのバギングを開始します...")
    
    for i in range(NUM_MODELS):
        print(f"\n--- Model {i+1}/{NUM_MODELS} Training Start ---")
        
        indices = np.random.choice(len(X_train_scaled), len(X_train_scaled), replace=True)
        X_boot = torch.FloatTensor(X_train_scaled[indices]).to(device)
        Y_boot = torch.FloatTensor(Y_train_scaled[indices]).to(device)
        
        model = FullyConnectedNN(2, 1, HIDDEN_LAYERS, get_activation_function(activation_func_str)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_boot, Y_boot), batch_size=BATCH_SIZE, shuffle=True)
        
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0 # ★修正: 1エポックの合計Loss用
            
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(loader) # ★修正: エポック全体の平均Lossを計算
            
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"  Model {i+1} | Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.6f}")
        
        torch.save(model.to('cpu').state_dict(), f"{model_weights_base}_{i}.pth")
        models.append(model)
        print(f"✅ Model {i+1} Training Completed.")
        
    with open(scaler_X_path, 'wb') as f: pickle.dump(scaler_X, f)
    with open(scaler_Y_path, 'wb') as f: pickle.dump(scaler_Y, f)
else:
    print("\n✅ 保存済みのアンサンブルモデルを読み込みます...")
    with open(scaler_X_path, 'rb') as f: scaler_X = pickle.load(f)
    with open(scaler_Y_path, 'rb') as f: scaler_Y = pickle.load(f)
    for i in range(NUM_MODELS):
        m = FullyConnectedNN(2, 1, HIDDEN_LAYERS, get_activation_function(activation_func_str))
        m.load_state_dict(torch.load(f"{model_weights_base}_{i}.pth"))
        models.append(m.to(device))

# ==============================================================================
# --- 推論 & 統計計算 ---
# ==============================================================================
print("\nアンサンブル推論と統計計算を実行中...")
df_truth_all = pd.read_excel(truth_data_path, sheet_name=f"{target_freq}Hz", header=1)
truth_data_blocks = [group.reset_index(drop=True) for _, group in df_truth_all.dropna(how='all').groupby(df_truth_all.isnull().all(axis=1).cumsum())]

pred_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
rmse_results = []
comparison_sheets_data = []
smooth_variance_data = {}

for i, amp in enumerate(pred_amps):
    # --- 1. 個別振幅シート用の推論 ---
    num_points = int(round(2 * amp / step)) + 1
    Breg = np.linspace(-amp, amp, num_points)
    X_pred = torch.FloatTensor(scaler_X.transform(np.array([[amp, b] for b in Breg]))).to(device)
    
    all_preds = []
    with torch.no_grad():
        for m in models:
            m.eval()
            out = scaler_Y.inverse_transform(m(X_pred).cpu().numpy()).flatten()
            all_preds.append(out)
    
    all_preds = np.array(all_preds)
    H_mean = np.mean(all_preds, axis=0)
    H_var = np.var(all_preds, axis=0) # 分散
    H_std = np.std(all_preds, axis=0) # 標準偏差(1σ)

    # --- 2. RMSE計算 ---
    h_ref, b_ref = np.full_like(H_mean, np.nan), np.full_like(H_mean, np.nan)
    if i < len(truth_data_blocks):
        df_ref = truth_data_blocks[i]
        if len(df_ref) == len(Breg):
            h_ref, b_ref = df_ref['H_descending'].values, df_ref['B'].values
            rmse = np.sqrt(np.mean((h_ref - H_mean)**2))
            hb_pred = H_mean[-1]
            rmse_results.append({
                'Amplitude(T)': amp, 
                'RMSE(H_descending)': rmse, 
                'Hb[A/m]': hb_pred, 
                'RMSE/Hb': rmse/abs(hb_pred) if hb_pred!=0 else np.nan
            })

    # --- 3. データフレーム作成 (E〜H列の修正適用済み) ---
    df_sheet = pd.DataFrame({
        'H_mean [A/m]': H_mean, 
        'B_reg [T]': Breg, 
        'H_ref [A/m]': h_ref, 
        'B_ref [T]': b_ref,
        'H_pred_variance': H_var,      # E列: 分散
        'H_pred_1sigma': H_std,        # F列: 1σの大きさ
        'H_pred_2sigma': H_std * 2,    # G列: 2σの大きさ
        'H_pred_3sigma': H_std * 3     # H列: 3σの大きさ
    })
    comparison_sheets_data.append({'amp': amp, 'df': df_sheet})

    # --- 4. variance_summary用の滑らかな推論 (200分割) ---
    B_dense = np.linspace(-amp, amp, 200)
    X_dense = torch.FloatTensor(scaler_X.transform(np.array([[amp, b] for b in B_dense]))).to(device)
    dense_preds = []
    with torch.no_grad():
        for m in models:
            dense_out = scaler_Y.inverse_transform(m(X_dense).cpu().numpy()).flatten()
            dense_preds.append(dense_out)
    dense_std = np.std(np.array(dense_preds), axis=0)
    smooth_variance_data[f'B [T] (Bm={amp:.2f}T)'] = B_dense
    smooth_variance_data[f'1σ [A/m] (Bm={amp:.2f}T)'] = dense_std

# ==============================================================================
# --- Excel出力 ---
# ==============================================================================
os.makedirs(os.path.dirname(rmse_out_path), exist_ok=True)
print(f"\nExcelファイルへの書き込みを開始します...")

with pd.ExcelWriter(rmse_out_path, engine='openpyxl') as writer:
    # 1. Infoシート
    create_info_df().to_excel(writer, sheet_name='Info', index=False)
    # 2. RMSE_Summaryシート
    if rmse_results:
        pd.DataFrame(rmse_results).to_excel(writer, sheet_name='RMSE_Summary', index=False)
    # 3. 各振幅シート
    for item in comparison_sheets_data:
        item['df'].to_excel(writer, sheet_name=f"{item['amp']:.2f}T", index=False)
    # 4. variance_summaryシート
    pd.DataFrame(smooth_variance_data).to_excel(writer, sheet_name='variance_summary', index=False)

print(f"\n✅ 全工程完了。保存先:\n{rmse_out_path}")