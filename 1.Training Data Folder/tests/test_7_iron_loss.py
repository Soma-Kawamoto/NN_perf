import pytest
import numpy as np

# conftest.pyで定義された 'module7' と 'module7_2' フィクスチャを使用します。

def test_calculate_hysteresis_area(module7):
    """シューレース公式による面積計算をテスト"""
    # 単位正方形 (面積1)
    h_square = np.array([0, 1, 1, 0])
    b_square = np.array([0, 0, 1, 1])
    area = module7.calculate_hysteresis_area(h_square, b_square)
    assert area == pytest.approx(1.0)
    
    # 縦2, 横3の長方形 (面積6)
    h_rect = np.array([1, 4, 4, 1])
    b_rect = np.array([2, 2, 4, 4])
    area_rect = module7.calculate_hysteresis_area(h_rect, b_rect)
    assert area_rect == pytest.approx(6.0)

    # 単純なひし形ループ
    h_loop = np.array([0, 100, 0, -100])
    b_loop = np.array([-0.5, 0, 0.5, 0])
    # 面積 = 2 * (三角形の面積) = 2 * (1/2 * 底辺 * 高さ) = 100 * 1.0 = 100
    area_loop = module7.calculate_hysteresis_area(h_loop, b_loop)
    assert area_loop == pytest.approx(100.0)

def test_hysteresis_area_in_variant_script(module7, module7_2):
    """派生スクリプトでも面積計算ロジックが同じであることを確認"""
    h_square = np.array([0, 1, 1, 0])
    b_square = np.array([0, 0, 1, 1])
    area1 = module7.calculate_hysteresis_area(h_square, b_square)
    area2 = module7_2.calculate_hysteresis_area(h_square, b_square)
    assert area1 == area2