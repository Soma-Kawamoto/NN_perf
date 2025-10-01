import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# ユーザー設定箇所
# ==============================================================================

# --- 確認したいデータの条件を指定 ---
MAT_NAME = "50A470"
TARGET_FREQ = 20  # 確認したい周波数
TARGET_AMP = 1.5  # 確認したい磁束密度の振幅

# ==============================================================================
# プログラム本体
# ==============================================================================

def find_data_file(base_path, mat, freq, amp, is_reduced_file):
    """
    指定された振幅のデータファイルを検索する。
    """
    if round(amp, 2) * 10 == int(round(amp, 2) * 10):
        amp_str = f"{amp:.1f}"
    else:
        amp_str = f"{amp:.2f}"
    
    base_filename = f"Bm{amp_str}hys_{freq}hz"
    
    if is_reduced_file:
        filename = f"{base_filename}_reduct.xlsx"
        path_to_check = os.path.join(base_path, mat, str(freq), filename)
    else:
        filename = f"{base_filename}.xlsx"
        path_to_check = os.path.join(base_path, mat, f"{freq}Hz", filename)

    if os.path.exists(path_to_check):
        return path_to_check
    else:
        return None

def main():
    """メイン実行関数"""
    print(f"■ プロットデータの確認を開始します。")
    print(f"  対象: {MAT_NAME}, 周波数: {TARGET_FREQ} Hz, 振幅: {TARGET_AMP} T")

    # --- 相対パス設定 ---
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.dirname(script_dir) # プロジェクトのルートディレクトリを取得
    original_data_path = os.path.join(base_dir, "1.Training Data Folder", "assets", "3.Fourier Transform Correction")
    reduced_data_path = os.path.join(base_dir, "1.Training Data Folder", "assets", "6.Downsampling")

    # --- ファイルの検索と読み込み ---
    original_file = find_data_file(original_data_path, MAT_NAME, TARGET_FREQ, TARGET_AMP, is_reduced_file=False)
    reduced_file = find_data_file(reduced_data_path, MAT_NAME, TARGET_FREQ, TARGET_AMP, is_reduced_file=True)

    if not original_file:
        print(f"❌ エラー: 元データファイルが見つかりません。")
        print(f"   検索したパスの例: {os.path.join(original_data_path, MAT_NAME, f'{TARGET_FREQ}Hz')}")
        return
        
    if not reduced_file:
        print(f"❌ エラー: 削減後データファイルが見つかりません。")
        print(f"   検索したパスの例: {os.path.join(reduced_data_path, MAT_NAME, str(TARGET_FREQ))}")
        return

    try:
        df_orig = pd.read_excel(original_file, header=0, usecols=[0, 1], names=['H', 'B'])
        df_red = pd.read_excel(reduced_file, header=0, usecols=[0, 1], names=['H', 'B'])

        # データクレンジング処理
        df_orig['H'] = pd.to_numeric(df_orig['H'], errors='coerce')
        df_orig['B'] = pd.to_numeric(df_orig['B'], errors='coerce')
        df_orig.dropna(inplace=True)

        df_red['H'] = pd.to_numeric(df_red['H'], errors='coerce')
        df_red['B'] = pd.to_numeric(df_red['B'], errors='coerce')
        df_red.dropna(inplace=True)

        print("✅ データの読み込みとクレンジングに成功しました。")
    except Exception as e:
        print(f"❌ エラー: ファイルの読み込み中に問題が発生しました: {e}")
        return

    if df_orig.empty or df_red.empty:
        if df_orig.empty:
            print("❌ エラー: 「元データ」のクレンジング後に有効なデータが残りませんでした。")
            print(f"   ファイルを確認してください: {original_file}")
        if df_red.empty:
            print("❌ エラー: 「削減後データ」のクレンジング後に有効なデータが残りませんでした。")
            print(f"   ファイルを確認してください: {reduced_file}")
        return

    # --- グラフの描画 ---
    h_orig, b_orig = df_orig['H'].values, df_orig['B'].values
    h_red, b_red = df_red['H'].values, df_red['B'].values
    
    plt.figure(figsize=(10, 8)) # グラフサイズを調整
    
    plt.plot(h_orig, b_orig, 'o-',color='red', markersize = 3, label=f'Original ({len(h_orig)} pts)')
    plt.plot(h_red, b_red, 'o-', color='royalblue', markersize=6, label=f'Downsampled ({len(h_red)} pts)')
    
    b_max_idx_red = np.argmax(b_red)
    b_min_idx_red = np.argmin(b_red)
    plt.scatter([h_red[b_max_idx_red], h_red[b_min_idx_red]],
                [b_red[b_max_idx_red], b_red[b_min_idx_red]],
                color='red', s=150, zorder=5, edgecolors='white', linewidth=1.5, label='Bmax/Bmin Points')
                
    plt.title(f'Downsampling Result for Bm={TARGET_AMP}T, {TARGET_FREQ}Hz', fontsize=16)
    plt.xlabel('H (A/m)', fontsize=12)
    plt.ylabel('B (T)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # ★★★ 修正箇所：Y軸の範囲を -1 から 1 に設定 ★★★
    plt.ylim(-1, 1)
    
    print("✅ グラフを生成しました。ウィンドウを閉じて終了してください。")
    plt.show()


if __name__ == '__main__':
    main()