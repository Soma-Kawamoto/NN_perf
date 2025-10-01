import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configparser
import io
try:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
except ImportError:
    print("【エラー】openpyxlライブラリが見つかりません。")
    print("コマンドプロンプトで `pip install openpyxl` を実行してインストールしてください。")
    exit()

# --- 設定ファイルの読み込み ---
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "1. Raw Normal Magnetization Curve_config.ini")
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

# --- 設定値の取得 ---
MATERIAL_NAME = config["settings"]["MATERIAL_NAME"]
FREQ_LIST = [int(f) for f in config["settings"]["FREQ_LIST"].split(",")]
AMP_MIN = float(config["settings"]["AMP_MIN"])
AMP_MAX = float(config["settings"]["AMP_MAX"])
AMP_STEP = float(config["settings"]["AMP_STEP"])

# ★★★ 修正箇所：相対パス設定 ★★★
# ----------------------------------------------------------------------
# 1. このスクリプト(.py)があるフォルダの場所を取得
#    (.../GPR_perf/2.Normal Magnetization Curve Extraction Folder)
script_dir = os.path.dirname(__file__)

# 2. 2階層上のフォルダ(GPR_perf)のパスを取得
gpr_perf_dir = os.path.dirname(os.path.dirname(script_dir))

# 3. 入力フォルダと出力フォルダのパスを構築
INPUT_BASE_FOLDER = os.path.join(gpr_perf_dir, "1.Training Data Folder", "assets", "3.Fourier Transform Correction") #input
OUTPUT_FOLDER = os.path.join(gpr_perf_dir, "2.Normal Magnetization Curve Extraction Folder", "assets", "1.Raw Normal Magnetization Curve") #output
# ----------------------------------------------------------------------


# ==============================================================================
# プログラム本体
# ==============================================================================

def find_input_file(directory, amp, freq):
    """
    指定された振幅の入力ファイルを探す。
    """
    if round(amp, 2) * 10 == int(round(amp, 2) * 10):
        amp_str = f"{amp:.1f}"
    else:
        amp_str = f"{amp:.2f}"
    
    filename = f"Bm{amp_str}hys_{freq}hz.xlsx"
    filepath = os.path.join(directory, filename)
    
    return filepath if os.path.exists(filepath) else None


def calculate_hysteresis_area(h_data, b_data):
    """シューレースの公式を用いてヒステリシスループの面積を計算する。"""
    return 0.5 * np.abs(np.dot(h_data, np.roll(b_data, -1)) - np.dot(b_data, np.roll(h_data, -1)))

def extract_bmax_record_from_file(filepath, amp):
    """
    単一のExcelファイルを読み込み、Bmaxと対応するHbを抽出してレコードを返す。
    """
    try:
        # データ読み込みと処理
        df = pd.read_excel(filepath, header=None, names=['H', 'B'])
        
        H_col = df['H']
        B_col = df['B']
        
        idx = B_col.idxmax()
        
        print(f"  Processing: {os.path.basename(filepath)} -> Bmax={B_col[idx]:.4f} at H={H_col[idx]:.4f}")
        return [amp, H_col[idx], B_col[idx]]

    except ValueError:
        print(f"Warning: No valid numeric data in -> {os.path.basename(filepath)}")
        return None
    except Exception as e:
        print(f"Error processing file: {os.path.basename(filepath)}, Error: {e}")
        return None

def main():
    """メイン処理を実行する関数"""
    print(f"■ Material: {MATERIAL_NAME}, Normal Magnetization Curve Extraction Start.")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for freq in FREQ_LIST:
        print("\n" + "="*60)
        print(f"Processing Frequency: {freq} Hz")
        print("="*60)
        
        input_dir_for_freq = os.path.join(INPUT_BASE_FOLDER, MATERIAL_NAME, f"{freq}Hz")
        if not os.path.isdir(input_dir_for_freq):
            print(f"Warning: Input directory not found, skipping freq {freq}Hz -> {input_dir_for_freq}")
            continue

        amps = np.round(np.arange(AMP_MIN, AMP_MAX + 1e-8, AMP_STEP), 2)
        records = []

        for amp in amps:
            fpath = find_input_file(input_dir_for_freq, amp, freq)
            if fpath is None:
                continue

            record = extract_bmax_record_from_file(fpath, amp)
            if record:
                records.append(record)

        if not records:
            print(f"No valid data processed for frequency {freq}Hz.")
            continue
            
        df_out = pd.DataFrame(records, columns=["Amplitude", "Hb", "Bm"])
        df_out.sort_values("Amplitude", inplace=True)

        # === Excelワークブック作成 (周波数ごとに1ファイル) ===
        wb = Workbook()
        ws1 = wb.active
        ws1.title = "Bm-Hb Data"
        ws1.append(["Amplitude", "Hb", "Bm"])
        for row in df_out.itertuples(index=False):
            ws1.append(list(row))

        # シート2: グラフ
        ws2 = wb.create_sheet("Bm-Hb Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df_out["Hb"], df_out["Bm"], marker='o', linestyle='-')
        
        ax.set_xlabel("H (A/m)")
        ax.set_ylabel("B (T)")
        ax.set_title(f"Normal Magnetization Curve - {MATERIAL_NAME} ({freq}Hz)")
        ax.grid(True)
        
        img_stream = io.BytesIO()
        fig.savefig(img_stream, format='png', dpi=150)
        plt.close(fig)
        img_stream.seek(0)

        img = XLImage(img_stream)
        img.anchor = "A1"
        ws2.add_image(img)

        output_file_path = os.path.join(OUTPUT_FOLDER, f"Bm-Hb Curve_{MATERIAL_NAME}_{freq}hz.xlsx")
        try:
            wb.save(output_file_path)
            print(f"\nOutput complete: {output_file_path}")
        except Exception as e:
            print(f"\nError saving file: {output_file_path}, Error: {e}")

if __name__ == "__main__":
    main()