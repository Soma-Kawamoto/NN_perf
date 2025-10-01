import pandas as pd
import os
import numpy as np
import configparser

# --- パス設定と設定ファイル読み込み ---
# 1. このスクリプト(.py)があるフォルダの場所を取得します。
config_path = os.path.join(os.path.dirname(__file__), "..",  "config", "1.高橋先生からもらったデータのextracting(xlsx)_config.ini")
script_dir = os.path.dirname(__file__)
# 2. スクリプトが置かれている 'src' フォルダの親フォルダを基準パスとします。
base_dir = os.path.dirname(script_dir)

config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

INPUT_DATA_ROOT = os.path.join(base_dir, "assets", "1.raw_data", "50A470_高橋先生からもらったデータ")
OUTPUT_DATA_ROOT = os.path.join(base_dir, "assets", "2.extracting data(xlsx)")
# ----------------------------------------------------------------------------------

# --- 設定ファイルからパラメータを読み込む ---
mat_name = config["settings"]["mat_name"]
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')
freq_list = [int(f) for f in config["settings"]["freq_list"].split(",")]
start_row = int(config["settings"]["start_row"])
end_row = int(config["settings"]["end_row"])
start_col_name = config["settings"]["start_col_name"]
end_col_name = config["settings"]["end_col_name"]


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