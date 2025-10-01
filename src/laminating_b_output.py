#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正規磁化曲線を読み込み、線長に対して等分割した際の磁束密度Bの値を算出するスクリプト
【幾何学的アプローチ・全文版】
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
import japanize_matplotlib

# Matplotlibのフォント設定
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14

# ==============================================================================
# ユーザー設定箇所
# ==============================================================================

# --- 1. 対象データの設定 ---
MATERIAL_NAME = "50A470"
FREQ = 20  # Hz

# --- 2. 分割条件の設定 ---
# Bの最大値（この値の-Bmaxから+Bmaxの範囲で線長を計算）
TARGET_BM_MAX = 2.0  # T

# 分割数 (指定した範囲のカーブを何点に分割するか)
NUM_POINTS = 250

# --- 3. パス設定 ---
# このスクリプトの場所に基づいてプロジェクトルートを特定
# スクリプトが期待するフォルダ構成で実行してください
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
except NameError:
    # 対話型環境などで__file__が定義されていない場合のフォールバック
    script_dir = os.getcwd()
    base_dir = os.path.dirname(script_dir)


# 入力フォルダ: 生の正規磁化曲線データ
INPUT_FOLDER = os.path.join(
    base_dir, "2.Normal Magnetization Curve Extraction Folder", "assets", "1.Raw Normal Magnetization Curve"
)

# 出力フォルダ: 線長分割したBの値を保存する場所
OUTPUT_FOLDER = os.path.join(
    base_dir, "2.Normal Magnetization Curve Extraction Folder", "assets", "3.laminating normal magnetization curve_b"
)

# ==============================================================================
# 補助クラス
# ==============================================================================

class AkimaExtrapolator:
    """
    Akima1DInterpolatorをラップし、範囲外の入力に対して線形外挿を行うクラス。
    """
    def __init__(self, x, y):
        if not np.all(np.diff(x) > 0):
            # データがソートされていない場合、自動的にソートする
            sort_indices = np.argsort(x)
            x = x[sort_indices]
            y = y[sort_indices]

        self.interpolator = Akima1DInterpolator(x, y)
        self.x_min, self.x_max = x[0], x[-1]
        self.y_min, self.y_max = y[0], y[-1]

        if len(x) > 1:
            self.slope_min = (y[1] - y[0]) / (x[1] - x[0])
            self.slope_max = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            self.slope_min = 0
            self.slope_max = 0

    def __call__(self, x_new):
        """補間または外挿を実行する。"""
        x_new = np.asarray(x_new)
        y_new = np.empty_like(x_new, dtype=float)

        in_range = (x_new >= self.x_min) & (x_new <= self.x_max)
        below_range = x_new < self.x_min
        above_range = x_new > self.x_max

        y_new[in_range] = self.interpolator(x_new[in_range])
        y_new[below_range] = self.y_min + self.slope_min * (x_new[below_range] - self.x_min)
        y_new[above_range] = self.y_max + self.slope_max * (x_new[above_range] - self.x_max)
        return y_new

# ==============================================================================
# 幾何学的アプローチによる線長等分割関数
# ==============================================================================

def get_arc_length_points_geometric(interp_func, b_start, b_end, num_points):
    """
    幾何学的な逐次探索（二分法）を用いて、曲線の線長をロバストに等分割する。
    """
    print("✅【幾何学的アプローチ】による線長等分割を開始します。")
    start_time = time.time()

    # --- ヘルパー関数: 2点間の短い弧長を計算 ---
    def calculate_short_arc_length(b1, b2, h1, scale_r, num_steps=1000):
        if b1 == b2:
            return 0.0
        # b1からb2までの短い区間をさらに細かく分割して線長を近似計算
        b_segment = np.linspace(b1, b2, num_steps)
        h_segment = interp_func(b_segment)
        
        dh = np.diff(h_segment)
        db = np.diff(b_segment)
        
        lengths = np.sqrt(dh**2 + (db * scale_r)**2)
        return np.sum(lengths)

    # 1. 全体の線長を概算し、1区間の目標長（歩幅）を決定
    print("  - 全長の概算と歩幅の計算中...")
    b_dense = np.linspace(b_start, b_end, 20000)
    h_dense = interp_func(b_dense)
    h_total_range = h_dense.max() - h_dense.min()
    b_total_range = b_end - b_start
    total_scale_ratio = h_total_range / b_total_range
    
    total_dh = np.diff(h_dense)
    total_db = np.diff(b_dense)
    total_length = np.sum(np.sqrt(total_dh**2 + (total_db * total_scale_ratio)**2))
    
    segment_length = total_length / (num_points - 1)
    print(f"  - 概算全長: {total_length:.4f}, 目標の歩幅: {segment_length:.4f}")

    # 2. 逐次探索で各点を決定
    b_laminated = [b_start]
    current_b = b_start
    current_h = interp_func(current_b)

    for i in range(1, num_points - 1):
        # --- 次の点を二分法で探索 ---
        search_min_b = current_b
        search_max_b = b_end
        
        # 30回程度のループで十分な精度に収束する
        for _ in range(30): 
            mid_b = (search_min_b + search_max_b) / 2
            # 現在地から中間点までの弧長を計算
            arc_len_to_mid = calculate_short_arc_length(current_b, mid_b, current_h, total_scale_ratio)
            
            # 目標の歩幅と比較して探索範囲を半分に狭める
            if arc_len_to_mid < segment_length:
                search_min_b = mid_b
            else:
                search_max_b = mid_b
        
        # 収束した点を次の点として採用
        current_b = (search_min_b + search_max_b) / 2
        b_laminated.append(current_b)
        current_h = interp_func(current_b)

        if (i % 25) == 0:
             print(f"  - {i}/{num_points - 2} 点目を探索完了...")

    b_laminated.append(b_end)
    
    end_time = time.time()
    print(f"✅ 幾何学的アプローチ完了。処理時間: {end_time - start_time:.2f}秒")
    return np.array(b_laminated)


# ==============================================================================
# プログラム本体
# ==============================================================================

def main():
    """
    メイン処理
    """
    print("--- 正規磁化曲線の線長等分割処理を開始します ---")

    # --- 1. 入力ファイルの特定とデータ読み込み ---
    input_file = os.path.join(INPUT_FOLDER, f"Bm-Hb Curve_{MATERIAL_NAME}_{FREQ}hz.xlsx")
    if not os.path.exists(input_file):
        print(f"🔴 エラー: 入力ファイルが見つかりません: {input_file}")
        print("テスト用にダミーデータを生成します。")
        h_orig = np.array([0, 50, 100, 200, 500, 1000, 2000, 5000])
        b_orig = np.array([0, 0.5, 0.9, 1.2, 1.4, 1.5, 1.6, 1.7])
    else:
        try:
            df_raw = pd.read_excel(input_file, usecols=["Hb", "Bm"])
            print(f"✅ 入力ファイルを読み込みました: {os.path.basename(input_file)}")
            h_orig = df_raw["Hb"].values
            b_orig = df_raw["Bm"].values
        except Exception as e:
            print(f"🔴 エラー: ファイルの読み込み中にエラーが発生しました: {e}")
            return

    # --- 2. データ拡張 (原点追加 & 対称化) ---
    sort_indices = np.argsort(b_orig)
    h_sorted = h_orig[sort_indices]
    b_sorted = b_orig[sort_indices]
    
    h_extended = np.concatenate([-h_sorted[::-1], [0], h_sorted])
    b_extended = np.concatenate([-b_sorted[::-1], [0], b_sorted])
    print("✅ データを拡張しました (原点追加、対称化)。")

    # --- 3. 拡張データのプロットによる確認 ---
    plt.figure(figsize=(10, 8))
    plt.plot(h_extended, b_extended, 'o-', label="拡張データ (元データ + 原点 + 対称データ)", markersize=4)
    plt.title(f"拡張された正規磁化曲線 - {MATERIAL_NAME} ({FREQ}Hz)")
    plt.xlabel("H [A/m]")
    plt.ylabel("B [T]")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    print("✅ 拡張データをプロットします。ウィンドウを閉じて処理を続行してください。")
    plt.show()

    # --- 4. 秋間補間による連続化 ---
    try:
        interp_h_from_b = AkimaExtrapolator(b_extended, h_extended)
        print("✅ 秋間補間（線形外挿付き）によりデータを連続化しました。")
    except Exception as e:
        print(f"🔴 エラー: 補間器の作成に失敗しました: {e}")
        return

    # --- 5. 線長計算と等分割 (★★新しい幾何学的アプローチ関数を呼び出す★★) ---
    b_laminated = get_arc_length_points_geometric(
        interp_func=interp_h_from_b,
        b_start=-TARGET_BM_MAX,
        b_end=TARGET_BM_MAX,
        num_points=NUM_POINTS
    )

    # --- 6. 等分割した点のプロットによる確認 ---
    h_laminated = interp_h_from_b(b_laminated)

    # --- Plan B の点を計算 (Plan Aの点同士の幾何学的な中間点) ---
    h_laminated_plan_b = (h_laminated[:-1] + h_laminated[1:]) / 2
    b_laminated_plan_b = (b_laminated[:-1] + b_laminated[1:]) / 2

    # 補間曲線全体をプロットするための高密度データ
    b_dense_plot = np.linspace(b_laminated.min(), b_laminated.max(), 2000)
    h_dense_plot = interp_h_from_b(b_dense_plot)

    plt.figure(figsize=(18, 10))
    plt.plot(h_dense_plot, b_dense_plot, '-', label="補間曲線", color='cyan', zorder=1)
    plt.scatter(h_laminated, b_laminated, c='red', s=40, label=f"Plan A ({len(b_laminated)}点)", zorder=3)
    plt.scatter(h_laminated_plan_b, b_laminated_plan_b, c='green', s=40, marker='x', label=f"Plan B (幾何学的中間点) ({len(b_laminated_plan_b)}点)", zorder=2)
    plt.title(f"線長等分割の結果 (幾何学的アプローチ) - {MATERIAL_NAME} ({FREQ}Hz)")
    plt.xlabel("H [A/m]")
    plt.ylabel("B [T]")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    print("✅ 線長等分割した点 (Plan A & B) をプロットします。ウィンドウを閉じて処理を続行してください。")
    plt.show()

    # --- 7. 結果をExcelファイルに出力 (Plan A) ---
    df_output = pd.DataFrame({'b': b_laminated})
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_filename = f"planA_laminated_b_{MATERIAL_NAME}_{FREQ}hz_Bm{TARGET_BM_MAX}_pts{NUM_POINTS}.xlsx"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    try:
        df_output.to_excel(output_path, index=False)
        print(f"✅ Plan A の結果をExcelファイルに出力しました: {output_path}")
    except Exception as e:
        print(f"🔴 エラー: Plan A のExcelファイルへの書き込み中にエラーが発生しました: {e}")

    # --- 8. 結果をExcelファイルに出力 (Plan B) ---
    df_output_plan_b = pd.DataFrame({'b': b_laminated_plan_b})
    output_filename_plan_b = f"planB_laminated_b_{MATERIAL_NAME}_{FREQ}hz_Bm{TARGET_BM_MAX}_pts{NUM_POINTS - 1}.xlsx"
    output_path_plan_b = os.path.join(OUTPUT_FOLDER, output_filename_plan_b)
    try:
        df_output_plan_b.to_excel(output_path_plan_b, index=False)
        print(f"✅ Plan B の結果をExcelファイルに出力しました: {output_path_plan_b}")
    except Exception as e:
        print(f"🔴 エラー: Plan B のExcelファイルへの書き込み中にエラーが発生しました: {e}")

    print("--- 全ての処理が完了しました ---")


if __name__ == '__main__':
    main()