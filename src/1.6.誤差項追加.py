#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ガウス過程回帰 (GPR) による B-H ヒステリシス回帰スクリプト
【v42改：Akimaデータの読み込み修正版（Bm=0.05対応）】
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
from scipy.stats import norm

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

# ==============================================================================
# ユーザー設定箇所
# ==============================================================================
ENABLE_PLOTTING = True
PERFORM_TRAINING = True
USE_BAYESIAN_OPTIMIZATION = True
BO_ITERATIONS = 5
ACQUISITION_FUNCTION = 'UCB'
UCB_KAPPA = 50
mat_name = "50A470"
target_freq = 20
kernel_type = "Matern52" 
train_amp = list(np.round(np.arange(0.1, 2.0 + 1e-8, 0.05), 1))

USE_AKIMA_DATA = True

OPTIMIZER = "lbfgsb"
MAX_ITERS = 1000
NUM_RESTARTS = 10

Bmreg_min = 0.05
Bmreg_max = 1.8
step = 0.05
script_dir = os.path.dirname(__file__)
base_dir = os.path.dirname(script_dir) # プロジェクトのルートディレクトリを取得
akima_excel_path = os.path.join(
    base_dir, "2.Normal Magnetization Curve Extraction Folder", "assets", "2.Akima spline interpolation",
    f"Bm-Hb Curve_akima_{mat_name}_50hz.xlsx"
)
initial_input_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "6.Downsampling")
original_data_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "5.Ascending branch of the hysteresis loop")
output_base = os.path.join(base_dir, "4.Answer (Bayesian_optimization)", "regression_results")
hyper_param_dir = os.path.join(base_dir, "4.Answer (Bayesian_optimization)", "hyperparameters")
truth_data_base = os.path.join(base_dir, "1.Training Data Folder", "assets", "7.reference data")
hyper_param_path = os.path.join(hyper_param_dir, f"hyperparamater_{mat_name}_{target_freq}hz_{kernel_type}.xlsx")
truth_data_path = os.path.join(truth_data_base, f"summary_{mat_name}_{step}.xlsx")

# ==============================================================================
# プログラム本体
# ==============================================================================

def create_info_df(amp_value=None):
    info_data = {
        "項目": ["実行日時", "材料名", "対象周波数 (Hz)", "GPRカーネル", "GPR最適化手法", 
                "GPR最大イテレーション数", "最適化リスタート回数", "学習データ(振幅 T)", "Akimaデータ使用"],
        "値": [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mat_name, target_freq, kernel_type, 
               OPTIMIZER, MAX_ITERS, NUM_RESTARTS, str(train_amp), USE_AKIMA_DATA]
    }
    if USE_AKIMA_DATA:
        info_data["項目"].append("Akimaデータファイル")
        info_data["値"].append(os.path.basename(akima_excel_path))
    if amp_value is not None:
        info_data["項目"].append("回帰対象振幅 (T)")
        info_data["値"].append(f"{amp_value:.2f}")
    return pd.DataFrame(info_data)

def add_comparison_chart_to_sheet(ws, df_len):
    chart = ScatterChart()
    chart.title = f"B-H Curve Comparison - {ws.title}"
    chart.x_axis.title = "H [A/m]"; chart.y_axis.title = "B [T]"
    chart.style = 13
    max_row = df_len + 1
    x_pred, y_pred = Reference(ws, min_col=1, min_row=2, max_row=max_row), Reference(ws, min_col=2, min_row=2, max_row=max_row)
    series_pred = Series(y_pred, x_pred, title="GPR Regression")
    series_pred.marker.symbol = "none"
    series_pred.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill="FF0000", w=12700))
    x_ref, y_ref = Reference(ws, min_col=3, min_row=2, max_row=max_row), Reference(ws, min_col=4, min_row=2, max_row=max_row)
    series_ref = Series(y_ref, x_ref, title="Reference")
    series_ref.marker.symbol = "none"
    series_ref.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill="0000FF", w=12700))
    chart.series.extend([series_pred, series_ref])
    if chart.legend: chart.legend.position = 'r'
    ws.add_chart(chart, "F2")

def calculate_acquisition_function(model, X, f_max, original_data_cache, acq_type='UCB', kappa=0):
    mean, var = model.predict(X)
    std = np.sqrt(np.maximum(var, 1e-9))
    if acq_type.upper() == 'UCB':
        acq_values = std * kappa
        for i, x in enumerate(X):
            bm, b = x[0], x[1]
            if bm in train_amp:
                df_original = original_data_cache.get(bm, pd.DataFrame(columns=['B', 'H']))
                if not df_original.empty and 'H' in df_original.columns:
                    closest_idx = (df_original['B'] - b).abs().idxmin()
                    h_true = df_original.iloc[closest_idx]['H']
                    h_pred = mean[i]
                    local_error = abs(h_pred - h_true) if not np.isnan(h_true) else 0
                    acq_values[i] += local_error
        return acq_values
    elif acq_type.upper() == 'EI':
        z = (mean - f_max) / (std + 1e-9)
        return (mean - f_max) * norm.cdf(z) + std * norm.pdf(z)

def find_closest_point_in_original_data(b_target, df_original, current_b_in_model):
    if df_original.empty or 'B' not in df_original.columns or 'H' not in df_original.columns:
        return pd.Series({'B': b_target, 'H': np.nan})
    
    # ソートしてインデックスをリセット
    df_sorted = df_original.sort_values('B').reset_index(drop=True)
    closest_idx = (df_sorted['B'] - b_target).abs().idxmin()
    
    if closest_idx >= len(df_sorted) or closest_idx < 0:
        print(f"  -> 警告: closest_idx={closest_idx} が範囲外です。デフォルト値を返します。b_target={b_target}")
        return pd.Series({'B': b_target, 'H': np.nan})
    
    # 最も近い点を取得
    candidate_row = df_sorted.iloc[closest_idx]
    b_candidate = candidate_row['B']
    
    # モデルに既に含まれているか確認
    max_attempts = len(df_sorted)  # 無限ループを防ぐ
    attempt = 0
    while np.any(np.isclose(current_b_in_model, b_candidate, atol=1e-5)) and attempt < max_attempts:
        # 隣接する点を探索 (前方と後方を交互にチェック)
        next_idx = closest_idx + (-1 if attempt % 2 == 0 else 1) * (attempt // 2 + 1)
        if 0 <= next_idx < len(df_sorted):
            candidate_row = df_sorted.iloc[next_idx]
            b_candidate = candidate_row['B']
            attempt += 1
        else:
            print(f"  -> 警告: {attempt + 1}回目の隣接点探索で範囲外です。デフォルト値を返します。b_target={b_target}")
            return pd.Series({'B': b_target, 'H': np.nan})
    
    if pd.isna(candidate_row['H']):
        print(f"  -> 警告: b_ideal={b_candidate:.4f} の H 値が欠損しています。スキップします。")
        return pd.Series({'B': b_target, 'H': np.nan})
    
    return candidate_row

def plot_current_regression(model, title):
    if not ENABLE_PLOTTING: return
    pred_amps_plot = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
    plt.figure(figsize=(10, 8))
    for amp in pred_amps_plot:
        num_points = int(round(2 * amp / step)) + 1
        Breg = np.linspace(-amp, amp, num_points)
        X_pred = np.array([[amp, b] for b in Breg])
        Hpred_means, Hpred_variances = model.predict(X_pred)
        Hpred = Hpred_means.flatten()
        Hpred_std = np.sqrt(Hpred_variances).flatten()
        plt.fill_betweenx(Breg, Hpred - Hpred_std, Hpred + Hpred_std, color='blue', alpha=0.15)
        plt.plot(Hpred, Breg, color='red', linestyle='-')
    plt.scatter(0, 0, s=100, c='black', marker='o', zorder=15, label='Origin (0,0)')
    plt.title(title); plt.xlabel(r'$\it{H}$ [A/m]'); plt.ylabel(r'$\it{B}$ [T]')
    plt.grid(True); plt.show(); plt.close()

def plot_acquisition_surface_3d(model, f_max, original_data_cache, title):
    if not ENABLE_PLOTTING: return
    b_plot_space = np.linspace(-Bmreg_max, Bmreg_max, 50)
    bm_plot_space = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
    B_grid, Bm_grid = np.meshgrid(b_plot_space, bm_plot_space)
    X_grid = np.vstack([Bm_grid.ravel(), B_grid.ravel()]).T
    acq_grid = calculate_acquisition_function(model, X_grid, f_max, original_data_cache, ACQUISITION_FUNCTION, UCB_KAPPA).reshape(Bm_grid.shape)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(B_grid, Bm_grid, acq_grid, cmap='viridis', edgecolor='none')
    ax.set_title(title); ax.set_xlabel('B [T]'); ax.set_ylabel('Bm [T]'); ax.set_zlabel(f'{ACQUISITION_FUNCTION} Value')
    plt.show(); plt.close()

def plot_marginal_acquisition(model, f_max, original_data_cache, title, next_points):
    if not ENABLE_PLOTTING: return
    # 2D表面プロット
    b_plot_space = np.linspace(-Bmreg_max, Bmreg_max, 50)
    bm_plot_space = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
    B_grid, Bm_grid = np.meshgrid(b_plot_space, bm_plot_space)
    X_grid = np.vstack([Bm_grid.ravel(), B_grid.ravel()]).T
    acq_grid = calculate_acquisition_function(model, X_grid, f_max, original_data_cache, ACQUISITION_FUNCTION, UCB_KAPPA).reshape(Bm_grid.shape)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(B_grid, Bm_grid, acq_grid, cmap='viridis', edgecolor='none')
    
    # 指定されたBm=bm_regでのUCB関数を赤線で強調
    bm_reg = 0.5
    b_space_reg = np.linspace(-bm_reg, bm_reg, 100)
    X_reg = np.array([[bm_reg, b] for b in b_space_reg])
    acq_reg = calculate_acquisition_function(model, X_reg, f_max, original_data_cache, ACQUISITION_FUNCTION, UCB_KAPPA)
    ax.plot(b_space_reg, [bm_reg] * len(b_space_reg), acq_reg, color='red', linewidth=2, label=f'Bm = {bm_reg}T')
    
    ax.set_title(f'3D Acquisition Function Surface with Bm={bm_reg}T Highlighted')
    ax.set_xlabel('B [T]'); ax.set_ylabel('Bm [T]'); ax.set_zlabel(f'{ACQUISITION_FUNCTION} Value')
    ax.legend()
    plt.show(); plt.close()

    # 各振幅ごとの1D UCB関数プロット
    plt.figure(figsize=(12, 8))
    for amp in train_amp:
        b_space = np.linspace(-amp, amp, 100)
        X_space = np.array([[amp, b] for b in b_space])
        acq_values = calculate_acquisition_function(model, X_space, f_max, original_data_cache, ACQUISITION_FUNCTION, UCB_KAPPA)
        max_acq_idx = np.argmax(acq_values)
        max_acq_b = b_space[max_acq_idx]
        max_acq_value = acq_values[max_acq_idx]
        plt.plot(b_space, acq_values, label=f'Bm = {amp:.1f}T')
        plt.plot(max_acq_b, max_acq_value, 'ro', markersize=10, label=f'Max UCB ({amp:.1f}T) at B={max_acq_b:.2f}')
    plt.title('Marginalized UCB Function for Each Bm')
    plt.xlabel('B [T]'); plt.ylabel(f'{ACQUISITION_FUNCTION} Value')
    plt.grid(True); plt.legend()
    plt.show(); plt.close()

def plot_training_data_with_bo(initial_X, initial_Y, akima_points, bo_points_all, bo_points_current, title):
    if not ENABLE_PLOTTING: return
    plt.figure(figsize=(8, 6))
    df_initial = pd.DataFrame({'Bm': initial_X[:, 0], 'B': initial_X[:, 1], 'H': initial_Y.flatten()})
    is_origin = np.isclose(df_initial['Bm'], 0.0)
    is_akima = np.isclose(df_initial['Bm'], np.abs(df_initial['B']))
    is_loop = ~is_origin & ~is_akima
    df_loops, df_akima, df_origin = df_initial[is_loop], df_initial[is_akima], df_initial[is_origin]
    if not df_loops.empty:
        grouped_loops = df_loops.groupby('Bm')
        label_added = False
        for _, group_df in grouped_loops:
            group_df_sorted = group_df.sort_values(by='B')
            label = 'Initial Points' if not label_added else None
            plt.plot(group_df_sorted['H'], group_df_sorted['B'], marker='o', markersize=3, linestyle='-', color='royalblue', alpha=0.5, label=label)
            label_added = True
    if len(akima_points['Hb']) > 0:
        Hb_vals, Bm_vals = akima_points['Hb'], akima_points['Bm']
        plt.scatter(Hb_vals, Bm_vals, s=50, c='red', marker='o', edgecolors='none', zorder=5, label='Akima Points')
        plt.scatter(-Hb_vals, -Bm_vals, s=50, c='red', marker='o', edgecolors='none', zorder=5)
    if not df_origin.empty:
        plt.scatter(df_origin['H'], df_origin['B'], s=100, c='black', marker='o', zorder=15, label='Origin (0,0)')
    if bo_points_all:
        bo_X_all = np.array([p[0][0] for p in bo_points_all])
        bo_Y_all = np.array([p[1][0] for p in bo_points_all])
        plt.scatter(bo_Y_all.flatten(), bo_X_all[:, 1], s=60, c='green', marker='^', alpha=0.6, label='BO Added (Past)')
    if bo_points_current:
        bo_X_current = np.array([p[0][0] for p in bo_points_current])
        bo_Y_current = np.array([p[1][0] for p in bo_points_current])
        plt.scatter(bo_Y_current.flatten(), bo_X_current[:, 1], s=150, c='green', marker='*', edgecolors='black', zorder=10, label='BO Added (Current)')
    plt.title(title); plt.xlabel(r'$\it{H}$ [A/m]'); plt.ylabel(r'$\it{B}$ [T]')
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend(); plt.show(); plt.close()

def main():
    print("\n[ステップ1] GPRモデルの初期学習データを読み込んでいます...")
    X_list, Y_list = [], []
    X_list.append([0.0, 0.0]); Y_list.append([0.0])
    for amp in train_amp:
        path = os.path.join(initial_input_base, mat_name, str(target_freq), f"Bm{amp:.1f}hys_{target_freq}hz_reduct.xlsx")
        if not os.path.exists(path): continue
        try:
            df = pd.read_excel(path, header=0, usecols=[0, 1])
            df.columns = ['H', 'B']
            df.dropna(inplace=True)
            df['H'] = pd.to_numeric(df['H'], errors='coerce')
            df['B'] = pd.to_numeric(df['B'], errors='coerce')
            df.dropna(inplace=True)
            if df.empty: continue
            B, H = df['B'].values, df['H'].values
            for b_val, h_val in zip(B, H): 
                X_list.append([amp, b_val]); Y_list.append([h_val])
        except Exception as e:
            print(f"  -> 警告: ファイル {os.path.basename(path)} の読み込みに失敗: {e}")

    akima_points = {'Hb': np.array([]), 'Bm': np.array([])}
    if USE_AKIMA_DATA:
        print("Akimaデータを学習に追加します...")
        try:
            target_regression_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
            df_akima_full = pd.read_excel(akima_excel_path, sheet_name='Interpolated Data', engine='openpyxl')
            df_akima_filtered = df_akima_full[np.round(df_akima_full['amp_Bm'], 2).isin(target_regression_amps)]
            print(f"  -> {len(df_akima_full)}点から{len(df_akima_filtered)}点を学習データとして使用。")
            print(f"  -> フィルタリングされたamp_Bm値: {[round(x, 4) for x in df_akima_filtered['amp_Bm'].values]}")
            df_akima_filtered = df_akima_filtered.dropna(subset=['amp_Hb', 'amp_Bm'])
            akima_points['Hb'] = df_akima_filtered['amp_Hb'].values
            akima_points['Bm'] = df_akima_filtered['amp_Bm'].values
            print(f"  -> akima_points['Bm']値: {[round(x, 4) for x in akima_points['Bm']]}")
            for Hb, Bm in zip(akima_points['Hb'], akima_points['Bm']):
                print(f"  -> 追加データ: Bm={Bm:.4f}, Hb={Hb:.4f}")
                X_list.append([Bm, Bm]); Y_list.append([Hb])
                X_list.append([Bm, -Bm]); Y_list.append([-Hb])
        except Exception as e:
            print(f"  -> 🔴 警告: Akimaデータの読み込みに失敗: {e}")
    else:
        print("Akimaデータを学習に使用しません。")

    X_train_initial, Y_train_initial = np.array(X_list), np.array(Y_list)
    unique_indices = np.unique(X_train_initial, axis=0, return_index=True)[1]
    X_train_initial = X_train_initial[unique_indices]
    Y_train_initial = Y_train_initial[unique_indices]
    mask = ~np.any(np.isnan(X_train_initial), axis=1) & ~np.isnan(Y_train_initial).flatten()
    X_train_initial, Y_train_initial = X_train_initial[mask], Y_train_initial[mask]
    print(f"初期学習データ: {len(X_train_initial)} 点")
    bm_values = X_train_initial[:, 0]
    if np.any(np.isclose(bm_values, 0.05, atol=1e-2)):
        print("  -> Bm=0.05が学習データに含まれています。")
    else:
        print("  -> ⚠️ Bm=0.05が学習データに含まれていません。")

    if ENABLE_PLOTTING:
        print("\n[ステップ1.5] 初期学習データをプロットします...")
        plot_training_data_with_bo(X_train_initial, Y_train_initial, akima_points, [], [], title="Initial Training Data")

    print("\n[ステップ2] GPRモデルの初期学習を実行します...")
    kern = GPy.kern.Matern52(input_dim=2, ARD=False)
    model = GPy.models.GPRegression(X_train_initial.copy(), Y_train_initial.copy(), kernel=kern, normalizer=True)
    
    model.Gaussian_noise.variance.constrain_bounded(1e-6, 1e-2, warning=False)
    model.kern.lengthscale.constrain_bounded(0.01, 100.0, warning=False)
    model.kern.variance.constrain_bounded(1e-4, 1e4, warning=False)

    try:
        model.optimize_restarts(num_restarts=NUM_RESTARTS, max_iters=MAX_ITERS, optimizer=OPTIMIZER, robust=True, verbose=True, messages=True)
        print("✅ 初期学習が完了しました。")
    except Exception as e:
        print(f"🔴 初期学習中にエラーが発生しました: {e}")
        return

    if not USE_BAYESIAN_OPTIMIZATION:
        print("\nベイズ最適化はスキップされました。")
    else:
        print(f"\n[ステップ3] ベイズ最適化を開始します (反復回数: {BO_ITERATIONS}, 獲得関数: {ACQUISITION_FUNCTION})")
        original_data_cache = {}
        for amp in train_amp:
            path = os.path.join(original_data_base, mat_name, f"{target_freq}Hz", f"Bm{amp:.1f}hys_{target_freq}hz.xlsx")
            if os.path.exists(path):
                try:
                    df = pd.read_excel(path, header=0)
                    if 'B' in df.columns and 'H' in df.columns:
                        original_data_cache[amp] = df.dropna(subset=['B', 'H'])
                    else:
                        print(f"  -> 警告: {path} に 'B' または 'H' 列がありません。スキップします。")
                except Exception as e:
                    print(f"  -> 警告: ファイル {os.path.basename(path)} の読み込みに失敗: {e}")
        bo_added_points = []
        for i in range(BO_ITERATIONS):
            print(f"\n--- ベイズ最適化 イテレーション {i+1}/{BO_ITERATIONS} ---")
            plot_current_regression(model, f'Regression Result (Before BO Iteration {i+1})')
            f_max = model.Y.max()
            plot_acquisition_surface_3d(model, f_max, original_data_cache, f'Acquisition Function Surface (Iteration {i+1})')
            current_iteration_points, next_points_for_plot = [], {}
            new_points_added_count = 0
            for amp in train_amp:
                if amp not in original_data_cache or original_data_cache[amp].empty:
                    print(f"  -> 警告: {amp:.1f}T のオリジナルデータがありません。スキップします。")
                    continue
                df_original = original_data_cache[amp]
                current_b_in_model = model.X[np.isclose(model.X[:, 0], amp)][:, 1]
                df_search_space = df_original[~np.round(df_original['B'], 5).isin(np.round(current_b_in_model, 5))]
                if df_search_space.empty:
                    print(f"  -> 警告: {amp:.1f}T の df_search_space が空です。df_original 全体を使用します。")
                    df_search_space = df_original
                print(f"  -> デバッグ: amp={amp}, df_search_space 行数={len(df_search_space)}")
                b_continuous_space = np.linspace(df_original['B'].min(), df_original['B'].max(), 200)
                X_continuous_space = np.array([[amp, b] for b in b_continuous_space])
                acq_values_continuous = calculate_acquisition_function(model, X_continuous_space, f_max, original_data_cache, ACQUISITION_FUNCTION, UCB_KAPPA)
                b_ideal = b_continuous_space[np.argmax(acq_values_continuous)]
                closest_point_row = find_closest_point_in_original_data(b_ideal, df_search_space, current_b_in_model)
                if pd.isna(closest_point_row['H']):
                    print(f"  -> 警告: {amp:.1f}T で b_ideal={b_ideal:.4f} の H 値が欠損しています。スキップします。")
                    continue
                b_next, h_next = closest_point_row['B'], closest_point_row['H']
                X_new, Y_new = np.array([[amp, b_next]]), np.array([[h_next]])
                if not np.any(np.all(np.isclose(model.X, X_new, atol=1e-5), axis=1)):
                    current_iteration_points.append((X_new, Y_new))
                    new_points_added_count += 1
                    next_points_for_plot[amp] = b_next
                    print(f"  -> {amp:.1f}T ループの次の候補点: (B={b_next:.4f}, H={h_next:.4f})")
            plot_marginal_acquisition(model, f_max, original_data_cache, f'Marginalized Acquisition Function (Iteration {i+1})', next_points_for_plot)
            if new_points_added_count == 0:
                print("追加できる新しい点がありませんでした。ループを終了します。")
                break
            if ENABLE_PLOTTING:
                plot_training_data_with_bo(X_train_initial, Y_train_initial, akima_points, bo_added_points, current_iteration_points, f'Updated Training Data (Iteration {i+1})')
            for X_new, Y_new in current_iteration_points:
                model.set_XY(np.vstack([model.X, X_new]), np.vstack([model.Y, Y_new]))
            bo_added_points.extend(current_iteration_points)
            print(f"  -> {new_points_added_count} 点をモデルに追加。モデルを更新・再学習します...")
            try:
                model.optimize(optimizer=OPTIMIZER, max_iters=MAX_ITERS, messages=False)
            except Exception as e:
                print(f"🔴 ベイズ最適化中の再学習でエラーが発生しました: {e}")
                break
        print("\n✅ ベイズ最適化が完了しました。")
        print(f"最終的な学習データ点数: {len(model.X)}")

    print("\n[ステップ4] 最終モデルで結果を評価・保存します...")
    try:
        df_truth_all = pd.read_excel(truth_data_path, sheet_name=f"{target_freq}Hz", header=1)
        truth_data_blocks = [group.reset_index(drop=True) for _, group in df_truth_all.dropna(how='all').groupby(df_truth_all.isnull().all(axis=1).cumsum())]
        print(f"✅ 最終評価用の正解データを読み込みました。")
    except Exception as e:
        print(f"🔴 最終評価用の正解データ読み込みエラー: {e}")
        truth_data_blocks = []

    plt.figure(figsize=(10, 8))
    pred_amps = np.round(np.arange(Bmreg_min, Bmreg_max + 1e-8, step), 2)
    rmse_results, comparison_sheets_data = [], []
    for i, amp in enumerate(pred_amps):
        num_points = int(round(2 * amp / step)) + 1
        Breg = np.linspace(-amp, amp, num_points)
        X_pred = np.array([[amp, b] for b in Breg])
        Hpred_means, _ = model.predict(X_pred)
        Hpred = Hpred_means.flatten()
        plt.plot(Hpred, Breg, color='red', linestyle='-')
        if i < len(truth_data_blocks):
            df_truth_loop = truth_data_blocks[i]
            if 'B' in df_truth_loop.columns and 'H_descending' in df_truth_loop.columns and np.allclose(Breg, df_truth_loop['B'].values):
                h_true_desc = df_truth_loop['H_descending'].values
                Hb_pred = Hpred[-1]
                rmse = np.sqrt(np.mean((h_true_desc - Hpred)**2))
                relative_rmse = rmse / Hb_pred if Hb_pred != 0 else np.nan
                rmse_results.append({'Amplitude (T)': amp, 'RMSE (H_descending)': rmse, 'Hb [A/m]': Hb_pred, 'RMSE/Hb': relative_rmse})
                df_comp = pd.DataFrame({'H_pred [A/m]': Hpred, 'B_reg [T]': Breg, 'H_ref [A/m]': h_true_desc, 'B_ref [T]': df_truth_loop['B'].values})
                comparison_sheets_data.append({'amp': amp, 'df': df_comp})

    plt.title(f'Final GPR Regression after Bayesian Optimization\n({mat_name} {target_freq}Hz, {kernel_type})')
    plt.xlabel(r'$\it{H}$ [A/m]'); plt.ylabel(r'$\it{B}$ [T]')
    plt.grid(True)
    
    print("\n最終的な回帰結果をプロットします...")
    plt.show()
    plt.close()

    if rmse_results:
        print("\n" + "="*70)
        print("最終モデルのRMSE 計算結果サマリー")
        print("="*70)
        df_rmse = pd.DataFrame(rmse_results)
        print(df_rmse.to_string(index=False))
        final_output_dir = os.path.join(output_base, mat_name, str(target_freq), kernel_type)
        os.makedirs(final_output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_RMSE_summary_BO_{mat_name}_{target_freq}hz_{kernel_type}.xlsx"
        rmse_out_path = os.path.join(final_output_dir, filename)
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
            print(f"✅ Excelファイルを保存しました。")
        except PermissionError:
            print(f"🔴 保存エラー: ファイルへのアクセスが拒否されました。")
        except Exception as e:
            print(f"🔴 Excelファイルへの書き込み中に予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()