#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN Ensemble Regression Script (Bagging)
- Excel Output compatible with GPR format
- Variance Summary implementation
- 個別モデルごとの詳細Excel出力機能を追加
- ★ 修正: Akimaデータをバギング対象から外し、常に100%学習データに含める
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

# Excelのグラフ描画用モジュールを追加
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties

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
individual_output_dir = os.path.join(output_base, mat_name, str(target_freq), f"{timestamp}_Individual_Models")
os.makedirs(individual_output_dir, exist_ok=True)

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

def create_info_df(is_individual=False, model_idx=None):
    info_data = {
        "項目": [
            "実行日時", "材料名", "対象周波数 (Hz)", "NN隠れ層", "NN活性化関数",
            "NN学習率", "NNエポック数", "NNバッチサイズ", "学習データ(振幅 T)", "Akimaデータ使用",
        ],
        "値": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mat_name, target_freq, str(HIDDEN_LAYERS), activation_func_str,
            LEARNING_RATE, EPOCHS, BATCH_SIZE, str(train_amp), str(USE_AKIMA_DATA),
        ]
    }
    
    if is_individual:
        info_data["項目"].extend(["アンサンブル内モデル番号", "結果出力ファイル名"])
        info_data["値"].extend([f"{model_idx+1} / {NUM_MODELS}", f"Model_{model_idx+1}_{mat_name}_{target_freq}hz.xlsx"])
    else:
        info_data["項目"].extend(["アンサンブルモデル数(n)", "結果出力ファイル名"])
        info_data["値"].extend([NUM_MODELS, output_filename])
        
    return pd.DataFrame(info_data)

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
# --- データ読み込み（通常データとAkimaデータを分離） ---
# ==============================================================================
X_std_list, Y_std_list = [], []
for amp in train_amp:
    path = os.path.join(input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
    if not os.path.exists(path): continue
    df = pd.read_excel(path, engine='openpyxl')
    for b, h in zip(df['B'].values, df['H'].values): 
        X_std_list.append([amp, b])
        Y_std_list.append([h])

X_aki_list, Y_aki_list = [], []
if USE_AKIMA_DATA:
    try:
        df_akima = pd.read_excel(akima_excel_path, sheet_name='Interpolated Data')
        target_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
        df_akima_f = df_akima[np.round(df_akima['amp_Bm'], 2).isin(target_amps)]
        for h, b in zip(df_akima_f['amp_Hb'].values, df_akima_f['amp_Bm'].values):
            X_aki_list.append([b, b]); Y_aki_list.append([h])
            X_aki_list.append([b, -b]); Y_aki_list.append([-h])
    except: pass

X_std_raw, Y_std_raw = np.array(X_std_list), np.array(Y_std_list)
X_aki_raw = np.array(X_aki_list) if len(X_aki_list) > 0 else np.empty((0, 2))
Y_aki_raw = np.array(Y_aki_list) if len(Y_aki_list) > 0 else np.empty((0, 1))

# --- スケーラーの準備（全データでFitさせて基準を統一する） ---
scaler_X, scaler_Y = StandardScaler(), StandardScaler()
if len(X_aki_raw) > 0:
    scaler_X.fit(np.vstack([X_std_raw, X_aki_raw]))
    scaler_Y.fit(np.vstack([Y_std_raw, Y_aki_raw]))
else:
    scaler_X.fit(X_std_raw)
    scaler_Y.fit(Y_std_raw)

X_std_scaled = scaler_X.transform(X_std_raw)
Y_std_scaled = scaler_Y.transform(Y_std_raw)

if len(X_aki_raw) > 0:
    X_aki_scaled = scaler_X.transform(X_aki_raw)
    Y_aki_scaled = scaler_Y.transform(Y_aki_raw)
else:
    X_aki_scaled, Y_aki_scaled = np.empty((0, 2)), np.empty((0, 1))

# ==============================================================================
# --- 学習 ---
# ==============================================================================
models = []
if PERFORM_TRAINING:
    print(f"\n🚀 {NUM_MODELS} 個のモデルのバギングを開始します...")
    
    for i in range(NUM_MODELS):
        print(f"\n--- Model {i+1}/{NUM_MODELS} Training Start ---")
        
        # 1. 通常データのみ、重複を許したブートストラップサンプリング
        indices = np.random.choice(len(X_std_scaled), len(X_std_scaled), replace=True)
        X_boot_std = X_std_scaled[indices]
        Y_boot_std = Y_std_scaled[indices]
        
        # 2. Akimaデータが存在する場合、サンプリングせずにそのまま結合 (100%確保)
        if len(X_aki_scaled) > 0:
            X_boot_np = np.vstack([X_boot_std, X_aki_scaled])
            Y_boot_np = np.vstack([Y_boot_std, Y_aki_scaled])
        else:
            X_boot_np = X_boot_std
            Y_boot_np = Y_boot_std
            
        X_boot = torch.FloatTensor(X_boot_np).to(device)
        Y_boot = torch.FloatTensor(Y_boot_np).to(device)
        
        model = FullyConnectedNN(2, 1, HIDDEN_LAYERS, get_activation_function(activation_func_str)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_boot, Y_boot), batch_size=BATCH_SIZE, shuffle=True)
        
        model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(loader)
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

rmse_results_ensemble = []
comparison_sheets_data_ensemble = []
smooth_variance_data = {}
individual_data = {m_idx: {'rmse_results': [], 'sheets_data': []} for m_idx in range(NUM_MODELS)}

for i, amp in enumerate(pred_amps):
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
    H_var = np.var(all_preds, axis=0)
    H_std = np.std(all_preds, axis=0)

    h_ref, b_ref = np.full_like(H_mean, np.nan), np.full_like(H_mean, np.nan)
    if i < len(truth_data_blocks):
        df_ref = truth_data_blocks[i]
        if len(df_ref) == len(Breg):
            h_ref, b_ref = df_ref['H_descending'].values, df_ref['B'].values
            
            rmse_ens = np.sqrt(np.mean((h_ref - H_mean)**2))
            hb_pred_ens = H_mean[-1]
            rmse_results_ensemble.append({
                'Amplitude(T)': amp, 'RMSE(H_descending)': rmse_ens, 'Hb[A/m]': hb_pred_ens, 'RMSE/Hb': rmse_ens/abs(hb_pred_ens) if hb_pred_ens!=0 else np.nan
            })
            
            for m_idx in range(NUM_MODELS):
                Hpred_ind = all_preds[m_idx]
                rmse_ind = np.sqrt(np.mean((h_ref - Hpred_ind)**2))
                hb_pred_ind = Hpred_ind[-1]
                individual_data[m_idx]['rmse_results'].append({
                    'Amplitude(T)': amp, 'RMSE(H_descending)': rmse_ind, 'Hb[A/m]': hb_pred_ind, 'RMSE/Hb': rmse_ind/abs(hb_pred_ind) if hb_pred_ind!=0 else np.nan
                })

    df_sheet_ens = pd.DataFrame({
        'H_mean [A/m]': H_mean, 'B_reg [T]': Breg, 'H_ref [A/m]': h_ref, 'B_ref [T]': b_ref,
        'H_pred_variance': H_var, 'H_pred_1sigma': H_std, 'H_pred_2sigma': H_std * 2, 'H_pred_3sigma': H_std * 3
    })
    comparison_sheets_data_ensemble.append({'amp': amp, 'df': df_sheet_ens})
    
    for m_idx in range(NUM_MODELS):
        df_sheet_ind = pd.DataFrame({
            'H_pred [A/m]': all_preds[m_idx], 'B_reg [T]': Breg, 'H_ref [A/m]': h_ref, 'B_ref [T]': b_ref
        })
        individual_data[m_idx]['sheets_data'].append({'amp': amp, 'df': df_sheet_ind})

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
print(f"\nExcelファイルへの書き込みを開始します...")

print(f"📂 個別モデルの出力先: {individual_output_dir}")
for m_idx in range(NUM_MODELS):
    ind_filename = f"Model_{m_idx+1}_{mat_name}_{target_freq}hz.xlsx"
    ind_filepath = os.path.join(individual_output_dir, ind_filename)
    
    with pd.ExcelWriter(ind_filepath, engine='openpyxl') as writer:
        create_info_df(is_individual=True, model_idx=m_idx).to_excel(writer, sheet_name='Info', index=False)
        if individual_data[m_idx]['rmse_results']:
            pd.DataFrame(individual_data[m_idx]['rmse_results']).to_excel(writer, sheet_name='RMSE_Summary', index=False)
        
        for item in individual_data[m_idx]['sheets_data']:
            sheet_name = f"{item['amp']:.2f}T"
            item['df'].to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            add_comparison_chart_to_sheet(ws, len(item['df']))
            
    if (m_idx + 1) % 5 == 0 or (m_idx + 1) == NUM_MODELS:
        print(f"  - Model {m_idx+1}/{NUM_MODELS} の出力を完了")

with pd.ExcelWriter(rmse_out_path, engine='openpyxl') as writer:
    create_info_df(is_individual=False).to_excel(writer, sheet_name='Info', index=False)
    if rmse_results_ensemble:
        pd.DataFrame(rmse_results_ensemble).to_excel(writer, sheet_name='RMSE_Summary', index=False)
    for item in comparison_sheets_data_ensemble:
        sheet_name = f"{item['amp']:.2f}T"
        item['df'].to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        add_comparison_chart_to_sheet(ws, len(item['df']))
    pd.DataFrame(smooth_variance_data).to_excel(writer, sheet_name='variance_summary', index=False)

print(f"\n✅ 全工程完了。全体アンサンブルの保存先:\n{rmse_out_path}")