import pytest
import numpy as np
import pandas as pd

# conftest.pyで定義された 'module3' フィクスチャを使用します。

def test_fit_quadratic(module3):
    """3点を通る二次関数の係数計算をテスト"""
    # y = 2x^2 + 3x + 1
    points = [(0, 1), (1, 6), (2, 15)]
    a, b, c = module3.fit_quadratic(points)
    np.testing.assert_allclose([a, b, c], [2.0, 3.0, 1.0], atol=1e-9)

    # y = x^2
    points = [(-1, 1), (0, 0), (1, 1)]
    a, b, c = module3.fit_quadratic(points)
    np.testing.assert_allclose([a, b, c], [1.0, 0.0, 0.0], atol=1e-9)

def test_make_table(module3):
    """係数とBm値からDataFrameを正しく生成できるかテスト"""
    a, b, c = 2.0, 3.0, 1.0
    bm_values = np.array([0, 1, 2])
    df = module3.make_table(a, b, c, bm_values)
    
    expected_reducted = a * bm_values**2 + b * bm_values + c
    expected_df = pd.DataFrame({
        'B_m': bm_values,
        'Reduced_Points': expected_reducted
    })
    
    pd.testing.assert_frame_equal(df, expected_df)