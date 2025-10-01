import pandas as pd
import os
import numpy as np

# ★★★ 修正箇所：パス設定のロジックを全面的に変更 ★★★
# ----------------------------------------------------------------------------------
# 1. このスクリプト(.py)があるフォルダの場所を取得します。
#    この場所が、移動可能な GPR_perf フォルダのパスになります。
base_dir = os.path.dirname(__file__)

# 2. 入力と出力の「ルートフォルダ」を、上記 base_dir からの相対パスで定義します。
INPUT_DATA_ROOT = os.path.join(base_dir,  "1.raw_data", "50A470_高橋先生からもらったデータ")
OUTPUT_DATA_ROOT = os.path.join(base_dir, "2.extracting data(xlsx)")
# ----------------------------------------------------------------------------------


def excel_column_to_index(col_name):
    """
    Excelの列名（例: 'A', 'B', 'AA'）を0から始まるインデックスに変換します。
    """
    index = 0
    for char in col_name:
        index = index * 26 + (ord(char.upper()) - ord('A') + 1)
    return index - 1

def process_single_frequency(mat_name, freq, start_row, end_row, start_col_name, end_col_name, input_base_path, output_data_root):
    """
    単一の周波数に対するデータ抽出処理を行う関数。
    """
    # --- 1. パスとファイル名を、受け取った引数を使って動的に設定 ---
    freq_in_filename = f"{freq}Hz"
    filename_part1 = f"{mat_name}_ring_{freq_in_filename}_12.5mm"
    filename = filename_part1

    output_base_path = os.path.join(output_data_root, mat_name, f"{freq}Hz")
    input_file_path = os.path.join(input_base_path, f"{filename}.xls")

    # --- 2. 出力フォルダの作成 ---
    try:
        os.makedirs(output_base_path, exist_ok=True)
        print(f"✅ 出力先フォルダを準備しました: {output_base_path}")
    except OSError as e:
        print(f"❌ エラー: 出力フォルダ '{output_base_path}' の作成に失敗しました: {e}")
        return

    # --- 3. 入力Excelファイルの読み込み ---
    try:
        df = pd.read_excel(input_file_path, sheet_name='data', header=None)
        print(f"✅ 入力ファイルを読み込みました: {input_file_path}")
    except FileNotFoundError:
        print(f"❌ エラー: 入力ファイルが見つかりません。パスとファイル名を確認してください: {input_file_path}")
        return
    except Exception as e:
        print(f"❌ エラー: Excelファイルの読み込み中に問題が発生しました: {e}")
        return

    # --- 4. データ抽出とファイル保存のループ処理 ---
    start_col_index = excel_column_to_index(start_col_name)
    end_col_index = excel_column_to_index(end_col_name)
    
    current_amp = 0.05
    
    for col_idx in range(start_col_index, end_col_index + 1, 3):
        col1_idx, col2_idx = col_idx, col_idx + 1
        
        try:
            data_col1 = df.iloc[start_row - 1:end_row, col1_idx]
            data_col2 = df.iloc[start_row - 1:end_row, col2_idx]
        except IndexError:
            print(f"⚠️ 警告: 列インデックス ({col1_idx}, {col2_idx}) が範囲外です。処理をここで終了します。")
            break

        output_df = pd.DataFrame({'A': data_col2.values, 'B': data_col1.values})
        
        if round(current_amp, 2) * 10 == int(round(current_amp, 2) * 10):
            amp_str = f"{current_amp:.1f}"
        else:
            amp_str = f"{current_amp:.2f}"
        
        base_filename = f"Bm{amp_str}hys_{freq}hz"
        output_path = os.path.join(output_base_path, f"{base_filename}.xlsx")
        
        try:
            output_df.to_excel(output_path, header=False, index=False)
            print(f"📄 ファイルを保存しました: {os.path.basename(output_path)}")
        except Exception as e:
            print(f"❌ エラー: ファイル '{os.path.basename(output_path)}' の保存中に問題が発生しました: {e}")

        current_amp = round(current_amp + 0.05, 2)


if __name__ == "__main__":
    # --- ユーザー設定項目 ---
    mat_name = "50A470"
    freq_list = [20, 50, 100, 200, 400, 600, 800, 1000]
    start_row = 3
    end_row = 1026
    start_col_name = 'AJ'
    end_col_name = 'EW'
    # -------------------------

    print(f"処理を開始します。対象材料: {mat_name}, 対象周波数: {freq_list}")

    for freq in freq_list:
        print("\n" + "="*70)
        print(f"▶️  現在処理中の周波数: {freq} Hz")
        print("="*70)
        
        process_single_frequency(
            mat_name=mat_name, 
            freq=freq,
            start_row=start_row,
            end_row=end_row,
            start_col_name=start_col_name,
            end_col_name=end_col_name,
            input_base_path=INPUT_DATA_ROOT,
            output_data_root=OUTPUT_DATA_ROOT
        )

    print("\n🎉 全ての周波数の処理が完了しました。")