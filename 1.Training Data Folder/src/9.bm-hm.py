import configparser
import pandas as pd  # pdとしてインポートするのが一般的
import numpy as np
from pathlib import Path

# --- 初期設定 ---
base_dir = Path(__file__).parent.parent
config_path = Path("/home/soma/NN_perf/1.Training Data Folder/config/9.bm_hm_config.ini")
# 文字列ではなくPathオブジェクトにする
save_dir = Path("/home/soma/NN_perf/1.Training Data Folder/assets/11.bm_hm")
save_dir.mkdir(parents=True, exist_ok=True) # フォルダがなければ作成

config = configparser.ConfigParser()
config.read(config_path, encoding='utf-8')

material = config.get('EXPERIMENT', 'material')
freq_listing = config.get('EXPERIMENT', 'freq_list')
freq_list = list(map(int, freq_listing.split()))
amp_start = config.getfloat('EXPERIMENT', 'amp_start')
amp_end = config.getfloat('EXPERIMENT', 'amp_end')

amp_list = np.arange(amp_start, amp_end + 0.01, 0.05)

# --- 処理ループ ---
for freq in freq_list:
    # ✅ 周波数ごとにリストをリセット
    bm_hm_list = []
    
    for amp in amp_list:
        amp = round(amp, 2) # ✅ 誤差対策
        
        file_path = (
            base_dir / "assets" / "3.Fourier Transform Correction" /
            material / f"{freq}Hz" / f"Bm{amp}hys_{freq}hz.xlsx"
        )
        
        if file_path.exists():
            try:
                # header=None を忘れずに
                df = pd.read_excel(file_path, header=None, usecols=[0, 1])
                h_data = df.iloc[:, 0].values
                b_data = df.iloc[:, 1].values # ✅ インデックスは 1
                
                # 正の側の最大値（$H_m, B_m$）を取得
                h_m = np.max(h_data)
                b_m = np.max(b_data)
                bm_hm_list.append([h_m, b_m])
                
                print(f"✅ 処理中: {file_path.name}")

            except Exception as e:
                print(f"⚠️ 読み込みエラー ({file_path.name}): {e}")

    # --- 周波数ループの最後で保存 ---
    if bm_hm_list:
        df_output = pd.DataFrame(bm_hm_list, columns=["Hm", "Bm"])
        # ✅ save_dir（ディレクトリ）を壊さずに個別のファイルパスを作る
        file_save_path = save_dir / f"{freq}hz_bm_hm.xlsx"
        df_output.to_excel(file_save_path, index=False)
        print(f"✨ 保存完了：{file_save_path.name}")
    else:
        print(f"ℹ️ {freq}Hz のデータは存在しませんでした")