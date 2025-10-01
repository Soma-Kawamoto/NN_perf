#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ヒステリシスループから線形補間を用いてHcとBrを算出するスクリプト
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from datetime import datetime
import configparser

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

# ==============================================================================
# --- 設定ファイルの読み込み ---
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "8. hc_br_config.ini")
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

# ==============================================================================
# プログラム本体
# ==============================================================================

def calculate_hc_br(df: pd.DataFrame) -> tuple[float, float]:
    """
    DataFrameから線形補間を用いてHcとBrを算出する。
    """
    hc_val = np.nan
    br_val = np.nan

    # --- Br (H=0のときのB) の算出 ---
    df_pos_h = df[df['H'] >= 0]
    df_neg_h = df[df['H'] <= 0]
    
    if not df_pos_h.empty and not df_neg_h.empty:
        # H=0をまたぐ2点を取得
        p_pos = df_pos_h.sort_values(by='H').iloc[0] # Hが正で最小の点
        p_neg = df_neg_h.sort_values(by='H', ascending=False).iloc[0] # Hが負で最大の点
        
        # 2点間で線形補間してH=0のときのBを求める
        if p_neg['H'] != p_pos['H']:
            br_val = np.interp(0, [p_neg['H'], p_pos['H']], [p_neg['B'], p_pos['B']])
        else: # H=0の点そのもの
            br_val = p_pos['B']

    # --- Hc (B=0のときのH) の算出 ---
    df_pos_b = df[df['B'] >= 0]
    df_neg_b = df[df['B'] <= 0]

    if not df_pos_b.empty and not df_neg_b.empty:
        # B=0をまたぐ2点を取得
        p_pos = df_pos_b.sort_values(by='B').iloc[0] # Bが正で最小の点
        p_neg = df_neg_b.sort_values(by='B', ascending=False).iloc[0] # Bが負で最大の点

        # 2点間で線形補間してB=0のときのHを求める
        if p_neg['B'] != p_pos['B']:
            hc_val = np.interp(0, [p_neg['B'], p_pos['B']], [p_neg['H'], p_pos['H']])
        else: # B=0の点そのもの
            hc_val = p_pos['H']
            
    return hc_val, br_val

def main():
    """
    メインの実行関数
    """
    # --- 1. パスの設定 ---
    mat_name = config["settings"]["mat_name"]
    freq = int(config["settings"]["freq"])

    amp_min = float(config["settings"]["amp_min"])
    amp_max = float(config["settings"]["amp_max"])
    amp_step = float(config["settings"]["amp_step"])
    # スクリプトが置かれている 'src' フォルダの親フォルダを基準パスとする
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(script_dir)
    
    input_dir = os.path.join(base_dir, "assets", "5.Ascending branch of the hysteresis loop", mat_name, f"{freq}Hz")
    output_hc_dir = os.path.join(base_dir, "assets", "9.Hc", mat_name, f"{freq}Hz")
    output_br_dir = os.path.join(base_dir, "assets", "10.Br", mat_name, f"{freq}Hz")
    
    os.makedirs(output_hc_dir, exist_ok=True)
    os.makedirs(output_br_dir, exist_ok=True)
    
    print(f"入力フォルダ: {input_dir}")
    print(f"Hc出力フォルダ: {output_hc_dir}")
    print(f"Br出力フォルダ: {output_br_dir}")

    # --- 2. メインループ処理 ---
    amps = np.round(np.arange(amp_min, amp_max + 1e-8, amp_step), 2)
    hc_results = []
    br_results = []

    print("\n複数振幅のデータをプロット準備中...")
    fig, ax = plt.subplots(figsize=(10, 8))

    print("\n処理を開始します...")
    for amp in amps:
        file_name = f"Bm{amp}hys_{freq}hz.xlsx"
        file_path = os.path.join(input_dir, file_name)

        if not os.path.exists(file_path):
            print(f"-> ファイルが見つかりません: {file_name} (スキップ)")
            continue

        try:
            # --- データの読み込み ---
            df = pd.read_excel(file_path, header=0, usecols=[0, 1])
            df.columns = ['H', 'B']
            df.dropna(inplace=True)

            if df.empty:
                print(f"-> データが空です: {file_name} (スキップ)")
                continue
            
            print(f"-> 処理中: {file_name}")

            hc_val, br_val = calculate_hc_br(df)
            hc_results.append({'Bm': amp, 'Hc': hc_val})
            br_results.append({'Bm': amp, 'Br': br_val})

            # --- グラフへの描画 ---
            ax.plot(df['H'], df['B'], color='royalblue', linestyle='-', alpha=0.6)
            ax.plot(hc_val, 0, color='blue', marker='o', linestyle='None', markersize=12)
            ax.plot(0, br_val, color='green', marker='o', linestyle='None', markersize=12)

        except Exception as e:
            print(f"-> エラー発生: {file_name} で問題が発生しました: {e}")

    # --- グラフの最終的な整形と表示 ---
    ax.set_title(f'H-B Curves with Hc and Br points (Linear Interpolation)\n({mat_name} at {freq}Hz)')
    ax.set_xlabel('H [A/m]')
    ax.set_ylabel('B [T]')
    ax.grid(True, linestyle=':')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    print("\nグラフを表示します...")
    plt.show()

    # --- Excelファイルへの出力 ---
    if hc_results:
        df_hc = pd.DataFrame(hc_results)
        output_hc_path = os.path.join(output_hc_dir, f"Hc_{freq}hz.xlsx")
        df_hc.to_excel(output_hc_path, index=False)
        print(f"\n✅ Hcの計算結果を保存しました: {output_hc_path}")
    else:
        print("\n-> Hcの計算結果はありませんでした。")

    if br_results:
        df_br = pd.DataFrame(br_results)
        output_br_path = os.path.join(output_br_dir, f"Br_{freq}hz.xlsx")
        df_br.to_excel(output_br_path, index=False)
        print(f"✅ Brの計算結果を保存しました: {output_br_path}")
    else:
        print("\n-> Brの計算結果はありませんでした。")

    print("\n全ての処理が完了しました。")


if __name__ == '__main__':
    main()