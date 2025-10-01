import os
import numpy as np
import pandas as pd
from openpyxl.chart import ScatterChart, Reference, Series
import configparser

# ====================================================================
# 補助関数
# ====================================================================

def format_bm_string(bm_in: float) -> str:
    """振幅Bmを文字列化"""
    if round(bm_in, 2) * 10 == int(round(bm_in, 2) * 10):
        return f"{bm_in:.1f}"
    else:
        return f"{bm_in:.2f}"

def check_monotonical(data: np.ndarray) -> bool:
    """データが単調増加/減少でない場合Trueを返す"""
    if data.size < 2: return False
    sign_changes = np.sum(np.diff(np.sign(np.diff(data))) != 0)
    return sign_changes > 2

def process_waveform_file(filepath: str, N: int):
    """単一のExcelファイルを読み込み、フーリエ補正した波形を返す"""
    try:
        df = pd.read_excel(filepath, header=None)
        raw_data = df.to_numpy()
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"  -> 🔴 Error reading file: {os.path.basename(filepath)}, Error: {e}")
        return None, None
    
    if raw_data.shape[0] != N or raw_data.shape[1] != 2:
        print(f"  -> 🔴 Error: Unexpected data shape {raw_data.shape} in {os.path.basename(filepath)}. Expected ({N}, 2).")
        return None, None
            
    h_data, b_data = (raw_data[:, 0], raw_data[:, 1])

    b_fft, h_fft = np.fft.rfft(b_data), np.fft.rfft(h_data)
    order = N // 2
    
    while True:
        b_fft_filtered, h_fft_filtered = np.zeros_like(b_fft), np.zeros_like(h_fft)
        indices = np.arange(1, order, 2)
        if indices.size > 0:
            b_fft_filtered[indices], h_fft_filtered[indices] = b_fft[indices], h_fft[indices]
        
        b_reconstructed, h_reconstructed = np.fft.irfft(b_fft_filtered, n=N), np.fft.irfft(h_fft_filtered, n=N)
        
        if not check_monotonical(b_reconstructed):
            break
        
        order -= 2
        if order < 1:
            print(f"  -> ⚠️ Warning: Fourier order reduction reached limit for {os.path.basename(filepath)}.")
            break
            
    return b_reconstructed, h_reconstructed

# ====================================================================
# メインの処理ロジック
# ====================================================================
def run_processing_for_step_size(material_name, freq_array, step_size, input_base_folder, output_base_folder, b_max, n_points):
    """
    指定された単一のSTEP_SIZEに対して、全周波数の処理を実行する関数。
    """
    # 出力フォルダが存在しない場合に作成する
    os.makedirs(output_base_folder, exist_ok=True)

    output_filename = os.path.join(output_base_folder, f"summary_{material_name}_{step_size}.xlsx")
    try:
        excel_writer = pd.ExcelWriter(output_filename, engine='openpyxl')
        print(f"✅ 出力ファイルを開きました: {output_filename}")
    except Exception as e:
        print(f"🔴 Error: Failed to create output Excel file at {output_filename}. Check permissions. Error: {e}")
        return

    for frequency in freq_array:
        print(f"\n====== Processing frequency: {frequency} Hz ======")
        input_folder_path = os.path.join(input_base_folder, material_name, f"{frequency}Hz")
        
        if not os.path.isdir(input_folder_path):
            print(f"-> ⚠️ Input directory not found, skipping: {input_folder_path}")
            continue
        
        all_results_for_freq = []
        
        num_amplitudes = int(round(b_max / step_size))
        amplitude_range = np.linspace(step_size, b_max, num_amplitudes)
        num_decimals = len(str(step_size).split('.')[-1]) if '.' in str(step_size) else 0
        amplitude_range = np.round(amplitude_range, num_decimals)

        for Bm in amplitude_range:
            bm_str = format_bm_string(Bm)
            filename = f"Bm{bm_str}hys_{frequency}hz.xlsx"
            filepath = os.path.join(input_folder_path, filename)
            
            b_wave, h_wave = process_waveform_file(filepath, n_points)
            
            if b_wave is None:
                continue

            num_points_interp = int(round(2 * Bm / step_size)) + 1
            Bd_target = np.linspace(-Bm, Bm, num_points_interp)
            Bd_target = np.round(Bd_target, num_decimals + 1)
            
            min_idx, max_idx = np.argmin(b_wave), np.argmax(b_wave)
            b_rolled, h_rolled = np.roll(b_wave, -min_idx), np.roll(h_wave, -min_idx)
            
            max_idx_rolled = (max_idx - min_idx + n_points) % n_points

            b_asc, h_asc = b_rolled[:max_idx_rolled + 1], h_rolled[:max_idx_rolled + 1]
            b_desc, h_desc = b_rolled[max_idx_rolled:], h_rolled[max_idx_rolled:]
            
            _, unique_indices_asc = np.unique(b_asc, return_index=True)
            _, unique_indices_desc = np.unique(b_desc, return_index=True)

            b_asc_unique, h_asc_unique = b_asc[np.sort(unique_indices_asc)], h_asc[np.sort(unique_indices_asc)]
            b_desc_unique, h_desc_unique = b_desc[np.sort(unique_indices_desc)], h_desc[np.sort(unique_indices_desc)]
            
            Hu_interp = np.interp(Bd_target, b_asc_unique, h_asc_unique)
            Hd_interp = np.interp(Bd_target, np.flip(b_desc_unique), np.flip(h_desc_unique))
            
            df_loop = pd.DataFrame({'B': Bd_target, 'H_ascending': Hu_interp, 'H_descending': Hd_interp})
            all_results_for_freq.append(df_loop)

        if all_results_for_freq:
            final_display_list = []
            for df in all_results_for_freq:
                final_display_list.append(df)
                final_display_list.append(pd.DataFrame([np.nan] * len(df.columns), index=df.columns).T)
            
            if final_display_list:
                final_df = pd.concat(final_display_list, ignore_index=True).iloc[:-1]
                sheet_name = f"{frequency}Hz"
                
                header_df = pd.DataFrame([[len(amplitude_range), b_max, step_size]])
                header_df.to_excel(excel_writer, sheet_name=sheet_name, index=False, header=False, startrow=0)
                final_df.to_excel(excel_writer, sheet_name=sheet_name, index=False, header=True, startrow=1)
                
                print(f"✅ Data for {frequency}Hz written to Excel sheet.")
    
    try:
        excel_writer.close()
        print(f"\n🎉 Excel file saved successfully: {output_filename}")
    except Exception as e:
        print(f"🔴 Error: Failed to save Excel file. Error: {e}")


def main():
    """メイン実行関数"""

    # --- 設定ファイルの読み込み ---
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "6.Reference data_config.ini")
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    # --- ユーザー設定箇所 ---
    MATERIAL_NAME = config["settings"]["MATERIAL_NAME"]
    B_MAX = float(config["settings"]["B_MAX"])
    FREQ_ARRAY = [int(f) for f in config["settings"]["FREQ_ARRAY"].split(",")]
    N_POINTS = int(config["settings"]["N_POINTS"])
    STEP_SIZES_TO_PROCESS = [float(s) for s in config["settings"]["STEP_SIZES_TO_PROCESS"].split(",")]
    # ★★★ 修正箇所：相対パス設定 ★★★
    # ----------------------------------------------------------------------
    # 1. このスクリプト(.py)があるフォルダの場所を取得
    script_dir = os.path.dirname(__file__)
    # 2. スクリプトが置かれている 'src' フォルダの親フォルダを基準パスとします。
    base_dir = os.path.dirname(script_dir)

    INPUT_BASE_FOLDER = os.path.join(base_dir, "assets", "3.Fourier Transform Correction")
    OUTPUT_BASE_FOLDER = os.path.join(base_dir, "assets", "7.reference data")
    # ----------------------------------------------------------------------
    
    for step in STEP_SIZES_TO_PROCESS:
        print("\n" + "#"*80)
        print(f"# Processing for STEP_SIZE = {step}")
        print("#"*80)
        
        run_processing_for_step_size(
            material_name=MATERIAL_NAME,
            freq_array=FREQ_ARRAY,
            step_size=step,
            input_base_folder=INPUT_BASE_FOLDER,
            output_base_folder=OUTPUT_BASE_FOLDER,
            b_max=B_MAX,
            n_points=N_POINTS
        )
    
    print("\nAll processing finished.")
    
if __name__ == "__main__":
    main()