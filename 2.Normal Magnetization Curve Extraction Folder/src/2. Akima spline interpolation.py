import os
import sys
import numpy as np
import pandas as pd
import configparser
from scipy.interpolate import Akima1DInterpolator
# xlsxwriterライブラリが必要です (pip install xlsxwriter)
try:
    import xlsxwriter
except ImportError:
    print("【エラー】xlsxwriterライブラリが見つかりません。")
    print("コマンドプロンプトで `pip install xlsxwriter` を実行してインストールしてください。")
    exit()


# ==============================================================================
# ユーザー設定箇所
# ==============================================================================

# --- 設定ファイルの読み込み ---
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "2. Akima spline interpolation_config.ini")
config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

# --- 設定値の取得 ---
MATERIAL_NAME = config["settings"]["MATERIAL_NAME"]
FREQ_LIST = [int(f) for f in config["settings"]["FREQ_LIST"].split(",")]
INTERPOLATION_METHOD = config["settings"]["INTERPOLATION_METHOD"]

# ★★★ 修正箇所：相対パス設定 ★★★
# ----------------------------------------------------------------------
# 1. このスクリプト(.py)があるフォルダの場所を取得
#    (.../GPR_perf/2.Normal Magnetization Curve Extraction Folder)
script_dir = os.path.dirname(__file__)

# 2. 親フォルダ(GPR_perf)のパスを取得
gpr_perf_dir = os.path.dirname(os.path.dirname(script_dir))

# 3. 入力フォルダと出力フォルダのパスを構築
INPUT_FOLDER = os.path.join(gpr_perf_dir, "2.Normal Magnetization Curve Extraction Folder", "assets", "1.Raw Normal Magnetization Curve")
OUTPUT_FOLDER = os.path.join(gpr_perf_dir, "2.Normal Magnetization Curve Extraction Folder", "assets", "2.Akima spline interpolation")
# ----------------------------------------------------------------------


# ==============================================================================
# プログラム本体
# ==============================================================================

def get_interpolator(x: np.ndarray, y: np.ndarray, method: str) -> Akima1DInterpolator:
    """x→y の補間関数を返す。"""
    if method.lower() == 'akima':
        # Bm_origを基準にHb_origを補間するための関数を作成
        # np.argsortでデータをBmの昇順に並べ替えることが重要
        sort_indices = np.argsort(x)
        return Akima1DInterpolator(x[sort_indices], y[sort_indices])
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def process_single_frequency(mat_name, freq, method, input_dir, output_dir):
    """単一の周波数に対して秋間補間を実行し、結果をExcelに出力する関数。"""
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(input_dir, f'Bm-Hb Curve_{mat_name}_{freq}hz.xlsx')

    if not os.path.isfile(input_file):
        print(f"Warning: Input file not found, skipping -> {input_file}")
        return

    try:
        # 'Bm-Hb Data'シートを読み込む
        df = pd.read_excel(input_file, sheet_name='Bm-Hb Data', header=0)
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return

    # データを抽出
    amp_Bm_orig = df.iloc[:, 0].to_numpy() # 元の振幅点 (補間後の評価点として使用)
    Hb_orig = df.iloc[:, 1].to_numpy()     # 元のHb (補間対象Y)
    Bm_orig = df.iloc[:, 2].to_numpy()     # 元のBm (補間対象X)

    # 補間を実行
    try:
        interp_func = get_interpolator(Bm_orig, Hb_orig, method=method)
        # 元の振幅点(amp_Bm_orig)におけるHbの値を補間して求める
        Hb_interpolated = interp_func(amp_Bm_orig)
    except Exception as e:
        print(f"Error during interpolation for freq {freq}Hz: {e}")
        return

    # 結果をDataFrameに格納
    df_out = pd.DataFrame({
        'amp_Hb': Hb_interpolated,
        'amp_Bm': amp_Bm_orig,
        'Hb': Hb_orig,
        'Bm': Bm_orig
    })
    df_out.sort_values('amp_Bm', inplace=True)

    # 出力ファイル名 (.xlsx)
    outfile = os.path.join(output_dir, f'Bm-Hb Curve_{method}_{mat_name}_{freq}hz.xlsx')
    
    # xlsxwriterを使ってデータとグラフを書き込む
    with pd.ExcelWriter(outfile, engine='xlsxwriter') as writer:
        df_out.to_excel(writer, sheet_name='Interpolated Data', index=False)
        
        # シート2: グラフ
        wb = writer.book
        ws = wb.add_worksheet('Plot')
        chart = wb.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})

        n = len(df_out)
        # 元のデータ系列
        chart.add_series({
            'name':       'Original Data',
            'categories': ['Interpolated Data', 1, 3, n, 3],  # Original_Bm (D列)
            'values':     ['Interpolated Data', 1, 2, n, 2],  # Original_Hb (C列)
            'marker':     {'type': 'circle', 'size': 6},
            'line':       {'none': True},
        })
        # 補間系列
        chart.add_series({
            'name':       f'{method.capitalize()} Interpolation',
            'categories': ['Interpolated Data', 1, 1, n, 1],  # amp_Bm (B列)
            'values':     ['Interpolated Data', 1, 0, n, 0],  # amp_Hb (A列)
            'marker':     {'type': 'none'},
            'line':       {'width': 1.5},
        })
        chart.set_title({'name': f'Normal Magnetization Curve ({mat_name} {freq}Hz)'})
        chart.set_x_axis({'name': 'H (A/m)'})
        chart.set_y_axis({'name': 'B (T)'})
        ws.insert_chart('B2', chart, {'x_scale': 1.5, 'y_scale': 1.5})

    print(f"Output complete: {outfile}")


def main():
    """メイン実行関数"""
    print(f"■ Material: {MATERIAL_NAME}, Akima Interpolation Start.")
    
    for freq in FREQ_LIST:
        print("\n" + "="*60)
        print(f"Processing Frequency: {freq} Hz")
        print("="*60)
        
        process_single_frequency(
            mat_name=MATERIAL_NAME,
            freq=freq,
            method=INTERPOLATION_METHOD,
            input_dir=INPUT_FOLDER,
            output_dir=OUTPUT_FOLDER
        )

if __name__ == '__main__':
    main()