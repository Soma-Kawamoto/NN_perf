#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数のアンサンブル学習結果 (Excelファイル) を結合するスクリプト
- モデル数(n)、平均(μ)、分散(σ^2) から全体の平均と分散を正確に再計算します
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

# ==============================================================================
# ユーザー設定
# ==============================================================================
# 結合したいExcelファイルのパスをリストで指定してください（絶対パスでも相対パスでも可）
FILE_PATHS_TO_MERGE = [
    r"C:\path\to\your\results\20250414_100000_Ensemble_Summary_50A470_20hz_NN.xlsx", # 例: 20回の結果
    r"C:\path\to\your\results\20250414_120000_Ensemble_Summary_50A470_20hz_NN.xlsx"  # 例: 30回の結果
]

# 出力先のディレクトリ
OUTPUT_DIR = r"C:\path\to\your\results\Merged"

# ==============================================================================
# 結合の数学的ロジック（プールされた平均と分散）
# ==============================================================================
def merge_stats(n_list, mean_list, var_list):
    """
    複数の (n, 平均, 分散) から、全体の平均と分散を計算する
    """
    N_total = sum(n_list)
    
    # 全体平均の計算: (n1*μ1 + n2*μ2 + ...) / N_total
    mean_total = sum(n * m for n, m in zip(n_list, mean_list)) / N_total
    
    # 全体分散の計算: { n1*(σ1^2 + μ1^2) + n2*(σ2^2 + μ2^2) + ... } / N_total - μ_total^2
    sum_sq = sum(n * (v + m**2) for n, m, v in zip(n_list, mean_list, var_list))
    var_total = (sum_sq / N_total) - (mean_total**2)
    
    return N_total, mean_total, var_total

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    if not FILE_PATHS_TO_MERGE:
        print("エラー: 結合するファイルが指定されていません。")
        return

    print("🔍 結合するファイルを読み込んでいます...")
    all_data = []
    base_info = None

    for path in FILE_PATHS_TO_MERGE:
        if not os.path.exists(path):
            print(f"🔴 ファイルが見つかりません: {path}"); exit()
        
        xl = pd.ExcelFile(path, engine='openpyxl')
        info_df = xl.parse('Info')
        info_dict = pd.Series(info_df['値'].values, index=info_df['項目']).to_dict()
        
        n_models = int(info_dict.get('アンサンブルモデル数(n)', 0))
        if n_models == 0:
            print(f"🔴 エラー: {path} にモデル数(n)の情報がありません。"); exit()
            
        # 条件の一致確認（最初のファイルをベースとする）
        if base_info is None:
            base_info = info_dict
        else:
            # 結合してはいけない条件（材料、周波数、隠れ層、など）をチェック
            keys_to_check = ['材料名', '対象周波数 (Hz)', 'NN隠れ層', 'NN活性化関数', '学習データ(振幅 T)']
            for key in keys_to_check:
                if str(base_info.get(key)) != str(info_dict.get(key)):
                    print(f"🔴 エラー: 学習条件が異なるファイルは結合できません。({key} が不一致)")
                    exit()
                    
        # 全シートのデータをメモリに読み込む
        sheet_data = {sheet: xl.parse(sheet) for sheet in xl.sheet_names if sheet not in ['Info', 'RMSE_Summary']}
        all_data.append({'n': n_models, 'sheets': sheet_data})
        print(f"  - 読込完了: {os.path.basename(path)} (モデル数: {n_models})")

    # --- 結合処理 ---
    total_n = sum(d['n'] for d in all_data)
    print(f"\n🔄 データを結合しています... (合計モデル数: {total_n})")
    
    merged_sheets = {}
    rmse_results = []
    
    # 最初のファイルのシート名（Bm T）を基準にループ
    sheet_names = all_data[0]['sheets'].keys()
    
    for sheet_name in sheet_names:
        n_list = [d['n'] for d in all_data]
        
        if sheet_name == 'variance_summary':
            # variance_summaryシートの結合
            merged_df = pd.DataFrame()
            cols = all_data[0]['sheets'][sheet_name].columns
            for col in cols:
                if col.startswith('B [T]'):
                    merged_df[col] = all_data[0]['sheets'][sheet_name][col] # B座標はそのまま
                elif col.startswith('1σ'):
                    # 1σ から 分散(σ^2) に戻して結合し、再度 1σ にする
                    std_list = [d['sheets'][sheet_name][col].values for d in all_data]
                    var_list = [std**2 for std in std_list]
                    # 平均は計算上のゼロとする（分散のみの結合の場合の近似）
                    # 本来は平均のズレも考慮しますが、ここでは各モデルがほぼ同じ平均に収束していると仮定
                    _, _, merged_var = merge_stats(n_list, [0]*len(n_list), var_list) 
                    merged_df[col] = np.sqrt(merged_var)
            merged_sheets[sheet_name] = merged_df
            continue

        # --- 個別振幅シート (XX.XT) の結合 ---
        base_df = all_data[0]['sheets'][sheet_name]
        mean_list = [d['sheets'][sheet_name]['H_mean [A/m]'].values for d in all_data]
        var_list = [d['sheets'][sheet_name]['H_pred_variance'].values for d in all_data]
        
        _, merged_mean, merged_var = merge_stats(n_list, mean_list, var_list)
        merged_std = np.sqrt(merged_var)
        
        # 結合したデータフレームを作成
        merged_df = pd.DataFrame({
            'H_mean [A/m]': merged_mean,
            'B_reg [T]': base_df['B_reg [T]'],
            'H_ref [A/m]': base_df['H_ref [A/m]'],
            'B_ref [T]': base_df['B_ref [T]'],
            ' ': '',
            'H_pred_variance': merged_var,
            'H_pred_1sigma': merged_std,
            'H_pred_2sigma': merged_std * 2,
            'H_pred_3sigma': merged_std * 3
        })
        merged_sheets[sheet_name] = merged_df
        
        # --- RMSEの再計算 ---
        h_ref = merged_df['H_ref [A/m]'].values
        # NaNを除外して計算（参照データが存在する場合のみ）
        valid_idx = ~np.isnan(h_ref)
        if np.any(valid_idx):
            rmse = np.sqrt(np.mean((h_ref[valid_idx] - merged_mean[valid_idx])**2))
            hb_pred = merged_mean[-1]
            amp_val = float(sheet_name.replace('T', ''))
            rmse_results.append({
                'Amplitude(T)': amp_val, 
                'RMSE(H_descending)': rmse, 
                'Hb[A/m]': hb_pred, 
                'RMSE/Hb': rmse/abs(hb_pred) if hb_pred != 0 else np.nan
            })

    # --- 保存処理 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mat_name = base_info.get('材料名', 'Unknown')
    freq = base_info.get('対象周波数 (Hz)', 'Unknown')
    
    out_filename = f"{timestamp}_Merged_Ensemble(n={total_n})_{mat_name}_{freq}hz.xlsx"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    
    print(f"\n💾 結合データを保存しています: {out_filename}")
    
    # 新しいInfoデータの作成
    base_info['実行日時'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_info['アンサンブルモデル数(n)'] = total_n
    base_info['結果出力ファイル名'] = out_filename
    base_info['結合元ファイル'] = ", ".join([os.path.basename(p) for p in FILE_PATHS_TO_MERGE])
    new_info_df = pd.DataFrame({"項目": list(base_info.keys()), "値": list(base_info.values())})
    
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        new_info_df.to_excel(writer, sheet_name='Info', index=False)
        pd.DataFrame(rmse_results).to_excel(writer, sheet_name='RMSE_Summary', index=False)
        for sheet_name, df in merged_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"✅ 全ての結合処理が完了しました！\n出力先: {out_path}")

if __name__ == "__main__":
    main()