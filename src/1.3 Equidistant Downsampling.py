#プロット形式の違い。
# -*- coding: utf-8 -*-
"""
ガウス過程回帰 (GPR) による B-H ヒステリシス回帰スクリプト
【v24：参照データのプロットスタイルを修正】
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
import japanize_matplotlib
from datetime import datetime
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
# ==============================================================================
# ユーザー設定箇所
# ==============================================================================

# --- 1. 動作モード設定 ---
PERFORM_TRAINING = True

# --- 2. モデルと学習データ、最適化の設定 ---
mat_name = "50A470"
target_freq = 20
kernel_type = "Matern52" 

Bmtrain_min = 0.1
Bmtrain_max = 1.8
train_step = 0.1
train_amp = list(np.round(np.arange(Bmtrain_min, Bmtrain_max + 1e-8, train_step), 1))

# 最適化手法を選択 ("lbfgsb" または "scg")
OPTIMIZER = "lbfgsb"
# 最適化の最大イテレーション回数
MAX_ITERS = 1000
# 最適化のリスタート回数
NUM_RESTARTS = 10


# --- 3. 回帰（予測）対象の設定 ---
Bmreg_min = 0.05
Bmreg_max = 1.8
step = 0.05


# --- パス設定 ---
script_dir = os.path.dirname(__file__)
base_dir = os.path.dirname(script_dir) # プロジェクトのルートディレクトリを取得
akima_excel_path = os.path.join(
    base_dir,
    "2.Normal Magnetization Curve Extraction Folder", "assets", "2.Akima spline interpolation",
    "Bm-Hb Curve_akima_50A470_50hz.xlsx"
)
input_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "6.2 Downsampling at s")#6.2 Downsampling at sは等間隔点数削減のフォルダ
output_base = os.path.join(base_dir, "3.3 Answer (Equidistant Downsampling)", "regression_results")
hyper_param_dir = os.path.join(base_dir, "3.3 Answer (Equidistant Downsampling)", "hyperparameters")
truth_data_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "7.reference data")
hyper_param_path = os.path.join(hyper_param_dir, f"hyperparamater_{mat_name}_{target_freq}hz_{kernel_type}.xlsx")
truth_data_path = os.path.join(truth_data_base, f"summary_{mat_name}_{step}.xlsx")


# ==============================================================================
# プログラム本体
# ==============================================================================

def create_info_df(amp_value=None):
    """Excelファイルに記載するメタデータを作成する関数"""
    info_data = {
        "項目": [
            "実行日時", "材料名", "対象周波数 (Hz)", "GPRカーネル", 
            "GPR最適化手法", "GPR最大イテレーション数", "最適化リスタート回数",
            "学習データ(振幅 T)", "Akima補間データ"
        ],
        "値": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            mat_name,
            target_freq,
            kernel_type,
            OPTIMIZER,
            MAX_ITERS,
            NUM_RESTARTS,
            str(train_amp),
            os.path.basename(akima_excel_path)
        ]
    }
    if amp_value is not None:
        info_data["項目"].append("回帰対象振幅 (T)")
        info_data["値"].append(f"{amp_value:.2f}")
    return pd.DataFrame(info_data)

def add_comparison_chart_to_sheet(ws, df_len):
    """ワークシートに比較グラフを追加する関数（デフォルトフォント版）"""
    chart = ScatterChart()
    chart.title = f"B-H Curve Comparison - {ws.title}"
    chart.x_axis.title = "H [A/m]"
    chart.y_axis.title = "B [T]"
    chart.style = 13
    max_row = df_len + 1
    x_pred = Reference(ws, min_col=1, min_row=2, max_row=max_row)
    y_pred = Reference(ws, min_col=2, min_row=2, max_row=max_row)
    series_pred = Series(y_pred, x_pred, title="GPR Regression")
    series_pred.marker.symbol = "none"
    series_pred.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill="FF0000", w=12700))
    x_ref = Reference(ws, min_col=3, min_row=2, max_row=max_row)
    y_ref = Reference(ws, min_col=4, min_row=2, max_row=max_row)
    series_ref = Series(y_ref, x_ref, title="Reference")
    series_ref.marker.symbol = "none"
    series_ref.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill="0000FF", w=12700))
    chart.series.append(series_pred)
    chart.series.append(series_ref)
    if chart.legend:
      chart.legend.position = 'r'
    ws.add_chart(chart, "F2")

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

print("\nGPRモデルの学習データを読み込んでいます...")
X_list, Y_list = [], []
X_list.append([0.0, 0.0]); Y_list.append([0.0])
for amp in train_amp:
    path = os.path.join(input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
    if not os.path.exists(path): continue
    df = pd.read_excel(path, engine='openpyxl')
    B, H = df['B'].values, df['H'].values
    for b_val, h_val in zip(B, H): X_list.append([B.max(), b_val]); Y_list.append([h_val])

target_regression_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
df_akima_full = pd.read_excel(akima_excel_path, sheet_name='Interpolated Data', engine='openpyxl')
df_akima_filtered = df_akima_full[np.round(df_akima_full['amp_Bm'], 2).isin(target_regression_amps)]
print(f"Akimaデータ: {len(df_akima_full)}点から、回帰範囲に合致する{len(df_akima_filtered)}点を学習データとして使用します。")
Hb_vals = df_akima_filtered['amp_Hb'].values
Bm_vals = df_akima_filtered['amp_Bm'].values
for Hb, Bm in zip(Hb_vals, Bm_vals):
    X_list.append([Bm, Bm]); Y_list.append([Hb])
    X_list.append([Bm, -Bm]); Y_list.append([-Hb])

X_train, Y_train = np.array(X_list), np.array(Y_list)
print("学習データの読み込みが完了しました。")

# --- 学習データプロット ---
print("\n学習データをプロットしています...")
plt.figure(figsize=(8, 6))
for amp in train_amp:
    path = os.path.join(input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
    if not os.path.exists(path): continue
    df = pd.read_excel(path, engine='openpyxl')
    plt.plot(df['H'], df['B'], marker='o', markersize=3, linestyle='-', label=f'{amp:.1f} T Loop', color='royalblue', alpha=0.4)
plt.scatter(0, 0, s=80, c='black', marker='o', zorder=6, label='Origin Point')
plt.scatter(Hb_vals, Bm_vals, s=80, c='red', marker='o', edgecolors='none', zorder=5, label='Akima Points (Used)')
plt.scatter(-Hb_vals, -Bm_vals, s=80, c='red', marker='o', edgecolors='none', zorder=5)
plt.title(f'学習データ点の分布 - {mat_name} {target_freq}Hz ({kernel_type})', fontsize=14)
plt.xlabel(r'$\it{H}$ [A/m]')
plt.ylabel(r'$\it{B}$ [T]')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- GPRモデル構築 ---
print("\nガウス過程回帰モデルを構築しています...")
kern = GPy.kern.Matern52(input_dim=2, ARD=False)
model = GPy.models.GPRegression(X_train, Y_train, kernel=kern, normalizer=True)

# --- 学習またはハイパーパラメータの読み込み ---
should_load_params = not PERFORM_TRAINING
settings_match = False
if should_load_params:
    try:
        saved_info_df = pd.read_excel(hyper_param_path, sheet_name='Info')
        saved_settings = pd.Series(saved_info_df.値.values, index=saved_info_df.項目).to_dict()
        if (str(saved_settings.get('材料名')) == str(mat_name) and
            int(saved_settings.get('対象周波数 (Hz)')) == int(target_freq) and
            str(saved_settings.get('GPRカーネル')) == str(kernel_type) and
            str(saved_settings.get('GPR最適化手法')).lower() == str(OPTIMIZER).lower() and
            int(saved_settings.get('GPR最大イテレーション数')) == int(MAX_ITERS) and
            int(saved_settings.get('最適化リスタート回数')) == int(NUM_RESTARTS) and
            str(saved_settings.get('学習データ(振幅 T)')) == str(train_amp)):
            settings_match = True
    except Exception:
        settings_match = False

if not settings_match:
    if should_load_params:
        print("\n⚠️  警告: 保存済みのハイパーパラメータと設定が異なるか、ファイルが存在しません。")
        print("   安全のため、モデルの再学習を強制的に実行します。")
    print("\nモデルの学習（ハイパーパラメータ最適化）を開始します...")
    model.optimize_restarts(
        num_restarts=NUM_RESTARTS, max_iters=MAX_ITERS,
        optimizer=OPTIMIZER, verbose=True, messages=True
    )
    print("学習が完了しました。")
    params = {'variance': model.kern.variance.item(), 'lengthscale': model.kern.lengthscale.item(), 'noise_var': model.likelihood.variance.item()}
    df_params = pd.DataFrame([params])
    info_df = create_info_df()
    os.makedirs(os.path.dirname(hyper_param_path), exist_ok=True)
    with pd.ExcelWriter(hyper_param_path, engine='openpyxl') as writer:
        df_params.to_excel(writer, sheet_name='Hyperparameters', index=False)
        info_df.to_excel(writer, sheet_name='Info', index=False)
    print(f"最適化したハイパーパラメータと設定情報を保存しました:\n {hyper_param_path}")
else:
    print(f"\n✅ 設定が一致したため、保存済みのハイパーパラメータを読み込みます:\n {hyper_param_path}")
    df_params = pd.read_excel(hyper_param_path, sheet_name='Hyperparameters')
    params = df_params.iloc[0].to_dict()
    model.kern.variance = params['variance']
    model.kern.lengthscale = params['lengthscale']
    model.likelihood.variance = params['noise_var']
    print("ハイパーパラメータの設定が完了しました。"); print(params)

# --- 結果プロット、Excel出力、およびRMSE計算 ---
print("\n回帰結果を計算し、出力しています...")
plt.figure(figsize=(10, 8))
plt.scatter(Hb_vals, Bm_vals, marker='x', c='k', s=50, zorder=3, label='Akima (Train)')
plt.scatter(-Hb_vals, -Bm_vals, marker='x', c='k', s=50, zorder=3, label='_nolegend_')
pred_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
rmse_results = []
comparison_sheets_data = []

# 凡例を一度だけ表示するためのフラグ
trained_label_added = False
predicted_label_added = False
ref_line_label_added = False
ref_dots_label_added = False

for i, amp in enumerate(pred_amps):
    num_points = int(round(2 * amp / step)) + 1
    Breg = np.linspace(-amp, amp, num_points)
    X_pred = np.array([[amp, b] for b in Breg])
    Hpred_means, _ = model.predict(X_pred)
    Hpred = Hpred_means.flatten()
    has_ref_data = False
    if i < len(truth_data_blocks):
        df_truth_loop = truth_data_blocks[i]
        if 'B' in df_truth_loop.columns and 'H_descending' in df_truth_loop.columns and np.allclose(Breg, df_truth_loop['B'].values):
            has_ref_data = True
            h_true_desc = df_truth_loop['H_descending'].values
            b_true = df_truth_loop['B'].values
            Hb_pred = Hpred[-1]
            rmse = np.sqrt(np.mean((h_true_desc - Hpred)**2))
            relative_rmse = rmse / Hb_pred if Hb_pred != 0 else np.nan
            rmse_results.append({'Amplitude (T)': amp, 'RMSE (H_descending)': rmse, 'Hb [A/m]': Hb_pred, 'RMSE/Hb': relative_rmse})
            print(f"   Bm = {amp:.2f}T, RMSE = {rmse:.4f}, Hb = {Hb_pred:.2f}, RMSE/Hb = {relative_rmse:.4%}")
            
            # ★★★ 修正箇所１：参照データのプロットを条件分岐 ★★★
            if amp not in train_amp:
                # 強調表示する振幅の場合、青い実線でプロット
                label = None
                if not ref_line_label_added:
                    label = 'Reference (Predicted)'
                    ref_line_label_added = True
                plt.plot(h_true_desc, b_true, color='blue', linestyle='-', alpha=0.8, label=label, zorder=1)
            else:
                # 学習振幅の場合、薄い色の点でプロット
                label = None
                if not ref_dots_label_added:
                    label = 'Reference (Trained)'
                    ref_dots_label_added = True
                plt.plot(h_true_desc, b_true, color='cyan', marker='.', linestyle='none', markersize=5, alpha=0.6, label=label, zorder=1)

            df_comp = pd.DataFrame({
                'H_pred [A/m]': Hpred, 'B_reg [T]': Breg,
                'H_ref [A/m]': h_true_desc, 'B_ref [T]': b_true
            })
            comparison_sheets_data.append({'amp': amp, 'df': df_comp})
        else:
             print(f"   Bm = {amp:.2f}T, 警告: 正解データとB軸の点が一致しないためRMSE計算と参照プロットをスキップします。")
    
    # GPRの予測結果のプロット
    if amp in train_amp:
        label = None
        if not trained_label_added:
            label = 'Trained Amplitude'
            trained_label_added = True
        plt.plot(Hpred, Breg, color='gray', linestyle='--', alpha=0.7, label=label, zorder=2)
    else:
        label = None
        if not predicted_label_added:
            label = 'Predicted Amplitude'
            predicted_label_added = True
        plt.plot(Hpred, Breg, color='red', linestyle='-', alpha=1.0, label=label, zorder=2)

# --- 最終結果の表示と保存 ---
plt.xlabel(r'$\it{H}$ [A/m]')
plt.ylabel(r'$\it{B}$ [T]')
#plt.title(f'GPR Regression vs Reference Data - {mat_name} {target_freq}Hz ({kernel_type})')
#plt.legend(loc='best'); 
plt.grid(True)
plt.show()

if rmse_results:
    print("\n" + "="*70)
    print("RMSE 計算結果サマリー")
    print("="*70)
    df_rmse = pd.DataFrame(rmse_results)
    print(df_rmse.to_string(index=False))

    final_output_dir = os.path.join(output_base, mat_name, str(target_freq), kernel_type)
    os.makedirs(final_output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_RMSE_summary_{mat_name}_{target_freq}hz_{kernel_type}.xlsx"
    rmse_out_path = os.path.join(final_output_dir, filename)

    if not PERFORM_TRAINING and settings_match:
        try:
            pattern = f"_RMSE_summary_{mat_name}_{target_freq}hz_{kernel_type}.xlsx"
            existing_files = [f for f in os.listdir(final_output_dir) if f.endswith(pattern)]
            if existing_files:
                existing_files.sort()
                latest_file_to_delete = existing_files[-1]
                path_to_delete = os.path.join(final_output_dir, latest_file_to_delete)
                print(f"\n🔄 既存の古いファイルを削除し、新しい日時に更新します:\n {path_to_delete}")
                os.remove(path_to_delete)
        except (FileNotFoundError, PermissionError) as e:
            print(f"\n⚠️ 既存ファイルの削除に失敗しました: {e}。新規ファイルとして作成を続行します。")

    print(f"\n結果をファイルに保存します:\n {rmse_out_path}")

    try:
        with pd.ExcelWriter(rmse_out_path, engine='openpyxl') as writer:
            info_df = create_info_df()
            info_df.to_excel(writer, sheet_name='Info', index=False)
            df_rmse.to_excel(writer, sheet_name='RMSE_Summary', index=False)
            for item in comparison_sheets_data:
                amp = item['amp']
                df_data = item['df']
                sheet_name = f"{amp:.2f}T"
                df_data.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                add_comparison_chart_to_sheet(ws, len(df_data))
        print(f"\n✅ 結果を保存しました。")
    except PermissionError:
        print(f"\n🔴 保存エラー: ファイルへのアクセスが拒否されました。'{os.path.basename(rmse_out_path)}'が開かれていないか確認してください。")
    except Exception as e:
        print(f"\n🔴 Excelファイルへの書き込み中に予期せぬエラーが発生しました: {e.__class__.__name__}: {e}")

print("\n全ての処理が完了しました。")