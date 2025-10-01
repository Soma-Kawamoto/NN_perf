import numpy as np
import pandas as pd
import os
import configparser

def process_hst_b_signals(h_signal, b_signal, N=1024, high_freq_cutoff_order=None):
    """
    磁界Hと磁束密度Bの信号に対してフーリエ変換を行い、
    直流成分、偶数次高調波、および指定次数以上の高調波を除去し、
    逆フーリエ変換で再構成します。
    """
    processed_signals = []
    for original_signal in [h_signal, b_signal]:
        if len(original_signal) != N:
            print(f"      -> 警告: 信号長 {len(original_signal)} が期待値 {N} と異なり、スキップします。")
            return None, None
        
        coeffs = np.fft.rfft(original_signal)
        
        if len(coeffs) > 0:
            coeffs[0] = 0.0
        
        for k_harmonic_order in range(2, (N // 2) + 1, 2):
            if k_harmonic_order < len(coeffs):
                coeffs[k_harmonic_order] = 0.0
        
        if high_freq_cutoff_order is not None:
            if high_freq_cutoff_order >= 0 and high_freq_cutoff_order < len(coeffs):
                coeffs[high_freq_cutoff_order:] = 0.0
                
        reconstructed_signal = np.fft.irfft(coeffs, n=N)
        processed_signals.append(reconstructed_signal)
        
    return processed_signals[0], processed_signals[1]

def main():
    """
    メインの実行関数。
    ユーザー設定に基づき、複数の周波数に対してフーリエ変換補正をバッチ処理します。

    """
    # --- ユーザー設定項目 ---
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "2.fourier_transform_config.ini")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    mat_name = config["settings"]["mat_name"]
    frequencies = [int(f) for f in config["settings"]["frequencies"].split(",")]
    N_points = int(config["settings"]["N_points"])
    HARMONIC_CUTOFF_ORDER = int(config["settings"]["HARMONIC_CUTOFF_ORDER"])
    # -------------------------

    # ★★★ 修正箇所：相対パス設定 ★★★
    # ----------------------------------------------------------------------
    # 1. このスクリプト(.py)があるフォルダの場所を取得します。
    script_dir = os.path.dirname(__file__)
    # 2. スクリプトが置かれている 'src' フォルダの親フォルダを基準パスとします。
    base_dir = os.path.dirname(script_dir)

    base_input_root = os.path.join(base_dir, "assets","2.extracting data(xlsx)")
    base_output_root = os.path.join(base_dir, "assets","3.Fourier Transform Correction")
    # ----------------------------------------------------------------------

    total_files_processed_all_freqs = 0
    
    print(f"処理を開始します。対象: {mat_name}, 周波数: {frequencies}")
    if HARMONIC_CUTOFF_ORDER is not None:
        print(f"フーリエ変換フィルタ: {HARMONIC_CUTOFF_ORDER}次以上の高調波を除去します。")

    # --- 各周波数に対する繰り返し処理 ---
    for frequency in frequencies:
        print("\n" + "="*70)
        print(f"▶️  周波数: {frequency} Hz の処理を開始します...")
        print("="*70)

        input_dir = os.path.join(base_input_root, mat_name, f"{frequency}Hz")
        output_dir = os.path.join(base_output_root, mat_name,f"{frequency}Hz")

        if not os.path.isdir(input_dir):
            print(f"❌ 入力フォルダが見つかりません: {input_dir}")
            print(f"   周波数 {frequency} Hz の処理をスキップします。")
            continue

        os.makedirs(output_dir, exist_ok=True)

        files_in_dir = os.listdir(input_dir)
        xlsx_files = [f for f in files_in_dir if f.endswith('.xlsx')]

        if not xlsx_files:
            print(f"-> 処理対象のExcelファイル (.xlsx) が見つかりませんでした。")
            continue
        
        print(f"-> '{os.path.basename(input_dir)}' 内の {len(xlsx_files)} 個のファイルを処理します...")
        files_processed_this_freq = 0

        # --- 各ファイルに対する繰り返し処理 ---
        for filename in xlsx_files:
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)

            try:
                data = pd.read_excel(input_filepath, header=None)

                if data.shape[0] != N_points or data.shape[1] != 2:
                    print(f"   -> 警告: ファイル {filename} の形状 ({data.shape}) が期待値 ({N_points}, 2) と異なります。スキップします。")
                    continue
                
                h_original = data.iloc[:, 0].values
                b_original = data.iloc[:, 1].values

                h_processed, b_processed = process_hst_b_signals(
                    h_original, b_original, N=N_points,
                    high_freq_cutoff_order=HARMONIC_CUTOFF_ORDER
                )
                
                if h_processed is None:
                    continue

                output_df = pd.DataFrame(np.column_stack((h_processed, b_processed)))
                output_df.to_excel(output_filepath, header=False, index=False)
                
                files_processed_this_freq += 1

            except Exception as e :
                print(f"   ❌ エラー: ファイル {filename} の処理中に問題が発生しました: {e}")
        
        print(f"--- 周波数 {frequency} Hz の処理結果 ---")
        print(f"   処理されたファイル数: {files_processed_this_freq}")
        total_files_processed_all_freqs += files_processed_this_freq

    print("\n" + "="*70)
    print("🎉 全ての周波数の処理が完了しました。")
    print(f"   合計処理ファイル数: {total_files_processed_all_freqs}")
    print("="*70)


if __name__ == "__main__":
    main()