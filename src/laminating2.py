#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指定されたBmの範囲に対して、正規磁化曲線の線長等分割を交互の方式で実行し、
複数のExcelファイルとして出力するスクリプト。
"""
import os
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

# --- 2. 参照するマスターファイルの設定 ---
# laminating_b_output.py で、以下の設定で生成されたファイルをマスターとして使用します。
MASTER_BM_MAX = 2.0  # T
NUM_LAMINATE_POINTS = 240

# --- 3. 出力するBmの範囲と分割数の設定 ---
# 例: 1.51 から 1.6 までを10個に分割 -> BM_GEN_START=1.51, BM_GEN_END=1.6, BM_GEN_COUNT=10
BM_GEN_START = 1.81  # T
BM_GEN_END = 1.9    # T
BM_GEN_COUNT = 10

# --- 4. パス設定 ---
# このスクリプトの場所に基づいてプロジェクトルートを特定
script_dir = os.path.dirname(os.path.abspath(__file__))
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
            raise ValueError("入力xは単調増加である必要があります。")
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
# プログラム本体
# ==============================================================================

def main():
    """
    メイン処理
    """
    print("--- 正規磁化曲線の線長等分割処理（複数ファイル出力）を開始します ---")

    # --- 1. 入力ファイルの特定とデータ読み込み ---
    input_file = os.path.join(INPUT_FOLDER, f"Bm-Hb Curve_{MATERIAL_NAME}_{FREQ}hz.xlsx")
    if not os.path.exists(input_file):
        print(f"🔴 エラー: 入力ファイルが見つかりません: {input_file}")
        return

    try:
        df_raw = pd.read_excel(input_file, usecols=["Hb", "Bm"])
        print(f"✅ 入力ファイルを読み込みました: {os.path.basename(input_file)}")
    except Exception as e:
        print(f"🔴 エラー: ファイルの読み込み中にエラーが発生しました: {e}")
        return

    h_orig = df_raw["Hb"].values
    b_orig = df_raw["Bm"].values

    # --- 2. データ拡張 (原点追加 & 対称化) ---
    sort_indices = np.argsort(b_orig)
    h_sorted = h_orig[sort_indices]
    b_sorted = b_orig[sort_indices]
    h_extended = np.concatenate([-h_sorted[::-1], [0], h_sorted])
    b_extended = np.concatenate([-b_sorted[::-1], [0], b_sorted])
    print("✅ データを拡張しました (原点追加、対称化)。")

    # --- 3. 秋間補間による連続化 ---
    try:
        interp_h_from_b = AkimaExtrapolator(b_extended, h_extended)
        print("✅ 秋間補間（線形外挿付き）によりデータを連続化しました。")
    except Exception as e:
        print(f"🔴 エラー: 補間器の作成に失敗しました: {e}")
        return

    # --- 4. マスター点群の読み込み ---
    # laminating_b_output.pyで生成されたExcelファイルをマスター点群として読み込みます。
    print(f"\n--- マスター点群ファイルを読み込みます (Bm={MASTER_BM_MAX}T, Pts={NUM_LAMINATE_POINTS}) ---")

    # マスターファイルのファイル名を定義
    master_file_plan_a = f"planA_laminated_b_{MATERIAL_NAME}_{FREQ}hz_Bm{MASTER_BM_MAX}_pts{NUM_LAMINATE_POINTS}.xlsx"
    master_file_plan_b = f"planB_laminated_b_{MATERIAL_NAME}_{FREQ}hz_Bm{MASTER_BM_MAX}_pts{NUM_LAMINATE_POINTS}.xlsx"

    path_plan_a = os.path.join(OUTPUT_FOLDER, master_file_plan_a)
    path_plan_b = os.path.join(OUTPUT_FOLDER, master_file_plan_b)

    try:
        df_master_a = pd.read_excel(path_plan_a)
        b_master_plan_a = df_master_a['b'].values
        print(f"✅ Plan A マスターファイルを読み込みました: {master_file_plan_a} ({len(b_master_plan_a)}点)")

        df_master_b = pd.read_excel(path_plan_b)
        b_master_plan_b = df_master_b['b'].values
        print(f"✅ Plan B マスターファイルを読み込みました: {master_file_plan_b} ({len(b_master_plan_b)}点)")

    except FileNotFoundError:
        print(f"🔴 エラー: マスターファイルが見つかりません。")
        print(f"   laminating_b_output.py を実行して、以下のファイルを先に生成してください:")
        print(f"   - {path_plan_a}")
        print(f"   - {path_plan_b}")
        return
    except Exception as e:
        print(f"🔴 エラー: マスターファイルの読み込み中にエラーが発生しました: {e}")
        return


    # --- 5. メインループ: 指定されたBm範囲でファイルを生成 ---
    target_bm_list = np.linspace(BM_GEN_START, BM_GEN_END, BM_GEN_COUNT)
    print(f"\n✅ 以下の {len(target_bm_list)} 個の Bm 値でファイルを生成します:")
    print([round(val, 4) for val in target_bm_list])

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for i, target_bm_max in enumerate(target_bm_list):
        is_plan_a = (i % 2 == 0)
        plan_name = "planA" if is_plan_a else "planB"

        print(f"\n--- 処理中 ({i+1}/{len(target_bm_list)}): Bm = {target_bm_max:.4f} T ({plan_name}) ---")

        # --- 6. マスター点群からフィルタリングし、点群を生成 ---
        master_list = b_master_plan_a if is_plan_a else b_master_plan_b
        # マスターリストから target_bm_max の範囲内の点を抽出
        b_filtered = master_list[(master_list >= -target_bm_max) & (master_list <= target_bm_max)]
        # 端点 (-target_bm_max, +target_bm_max) を確実に追加し、重複を削除してソート
        b_output = np.unique(np.concatenate((b_filtered, [-target_bm_max, target_bm_max])))
        print(f"  - {plan_name}: マスター点群からフィルタリングし、{len(b_output)} 点を生成しました。")

        # --- 7. プロットによる確認 ---
        h_output = interp_h_from_b(b_output)

        # プロット用の背景曲線は target_bm_max の範囲で生成
        b_dense_plot = np.linspace(-target_bm_max, target_bm_max, 1000)
        h_dense_plot = interp_h_from_b(b_dense_plot)

        plt.figure(figsize=(8, 7))
        # 補間された元の曲線を薄い色でプロット
        plt.plot(h_dense_plot, b_dense_plot, '-', label="補間曲線", color='cyan', zorder=1)
        # 線長で等分割した点をプロット
        plt.scatter(h_output, b_output, c='red', s=25, label=f"{plan_name} 分割点 ({len(b_output)}点)", zorder=2)

        plt.title(f"線長等分割の結果 ({plan_name}) - Bm = {target_bm_max:.4f} T")
        plt.xlabel("H [A/m]")
        plt.ylabel("B [T]")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        print(f"  - 分割結果をプロットします。ウィンドウを閉じて次の処理に進んでください。")
        plt.show()

        # --- 8. 結果をExcelファイルに出力 ---
        df_output = pd.DataFrame({'b': b_output})

        # 出力ファイル名を作成
        output_filename = f"{plan_name}_laminated_b_{MATERIAL_NAME}_{FREQ}hz_Bm{target_bm_max:.2f}_pts{NUM_LAMINATE_POINTS}.xlsx"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Excelファイルに書き込むための情報シートを作成
        info_data = {
            "項目": [
                "生成日時", "材料名", "周波数 (Hz)", "最大磁束密度 Bm (T)",
                "分割方式", "総点数 (PlanA基準)", "出力点数"
            ],
            "値": [
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                MATERIAL_NAME, FREQ, f"{target_bm_max:.4f}",
                plan_name, NUM_LAMINATE_POINTS, len(b_output)
            ]
        }
        df_info = pd.DataFrame(info_data)

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_output.to_excel(writer, sheet_name='Data', index=False)
                df_info.to_excel(writer, sheet_name='Info', index=False)
            print(f"  ✅ 結果をExcelファイルに出力しました: {output_filename}")
        except Exception as e:
            print(f"  🔴 エラー: Excelファイルへの書き込み中にエラーが発生しました: {e}")

    print("\n--- 全ての処理が完了しました ---")


if __name__ == '__main__':
    main()
