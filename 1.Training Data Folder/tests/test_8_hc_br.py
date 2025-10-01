import pytest
import numpy as np
import pandas as pd

# conftest.pyで定義された 'module8' フィクスチャを使用します。

def test_calculate_hc_br_symmetric(module8):
    """対称なループでのHc, Br計算をテスト"""
    h = np.array([-100, 100])
    b = np.array([-1.0, 1.0])
    df = pd.DataFrame({'H': h, 'B': b})
    
    hc, br = module8.calculate_hc_br(df)
    
    # H=0のときB=0, B=0のときH=0になるはず
    assert br == pytest.approx(0.0)
    assert hc == pytest.approx(0.0)

def test_calculate_hc_br_asymmetric(module8):
    """非対称なループでのHc, Br計算をテスト"""
    h = np.array([-100, -10, 10, 200, 100, -100])
    b = np.array([-0.5, 0.5, 0.6, 1.0, -0.2, -0.5])
    df = pd.DataFrame({'H': h, 'B': b})
    
    hc, br = module8.calculate_hc_br(df)
    
    # Br (H=0の補間): Hが-10(B=0.5)と10(B=0.6)の間 -> Bは0.55になるはず
    assert br == pytest.approx(0.55)
    
    # Hc (B=0の補間): Bが-0.2(H=100)と0.5(H=-10)の間 -> Hは68.571...になるはず
    expected_hc = 100 + (0 - (-0.2)) * (-10 - 100) / (0.5 - (-0.2))
    assert hc == pytest.approx(expected_hc)

def test_calculate_hc_br_exact_zero(module8):
    """H=0またはB=0の点がデータに正確に存在する場合をテスト"""
    # H=0の点が存在
    h_br = np.array([-10, 0, 10])
    b_br = np.array([-0.5, 0.8, 1.0]) # Br = 0.8
    df_br = pd.DataFrame({'H': h_br, 'B': b_br})
    _, br = module8.calculate_hc_br(df_br)
    assert br == pytest.approx(0.8)

    # B=0の点が存在
    h_hc = np.array([-50, 20, 100])
    b_hc = np.array([-1.0, 0, 1.0]) # Hc = 20
    df_hc = pd.DataFrame({'H': h_hc, 'B': b_hc})
    hc, _ = module8.calculate_hc_br(df_hc)
    assert hc == pytest.approx(20)