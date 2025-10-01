import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family']='Times New Roman'
plt.rcParams['font.size']=20
# --- 1. 活性化関数の定義 ---
def relu(x):
    """ReLU（正規化線形ユニット）関数を計算します。"""
    return np.maximum(0, x)

def sigmoid(x):
    """シグモイド関数を計算します。"""
    return 1 / (1 + np.exp(-x))

# TanhはNumPyに np.tanhとして組み込まれています

# --- 2. プロット用データの生成 ---
# -5から5までの範囲でxの値を生成
x = np.linspace(-5, 5, 200)

# 各関数に対応するyの値を計算
y_relu = relu(x)
y_sigmoid = sigmoid(x)
y_tanh = np.tanh(x)

# --- 3. プロットの作成とカスタマイズ ---
# 描画領域を作成
plt.figure(figsize=(10, 7))

# 各関数を異なる色とラベルでプロット
plt.plot(x, y_relu, label='ReLU', color='#3498db', linewidth=3)
plt.plot(x, y_sigmoid, label='Sigmoid', color='#e74c3c', linewidth=3)
plt.plot(x, y_tanh, label='Tanh', color='#2ecc71', linewidth=3)

# 【変更点①】軸のスタイルを調整
ax = plt.gca()  # 現在の軸設定を取得

# 上と右の枠線を非表示に
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 下と左の枠線を原点(0,0)を通るように移動
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')

# グリッド線は非表示のまま
plt.grid(False)

# 【変更点②】凡例のフォントをTimes New Romanに設定
# フォントのプロパティを定義
font_props = FontProperties(family='Times New Roman', style='normal', size=30)

# 定義したフォントプロパティを凡例に適用
plt.legend(prop=font_props)

# 最終的なプロットを表示
plt.show()