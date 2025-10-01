import pytest
import numpy as np

# conftest.pyで定義された 'module5' フィクスチャを使用します。

def test_reduce_points_basic(module5):
    """基本的な点削減ロジックをテスト"""
    h = np.arange(20, dtype=float)
    # np.linspaceのデフォルトはendpoint=Trueのため、2*piが含まれる
    b = np.sin(np.linspace(0, 2 * np.pi, 20))
    # Bmax at idx=5, Bmin at idx=14
    
    # 必須点: 0(始点), 19(終点), 5(Bmax), 14(Bmin) -> 4点
    # 内部点として5点保持する
    h_red, b_red = module5.reduce_points(h, b, interior_keep=5)
    
    # 期待される合計点数 = 4 (必須) + 5 (内部) = 9
    assert len(h_red) == 9
    
    # 必須点が保持されているか確認
    assert h[0] in h_red
    assert h[19] in h_red
    assert h[5] in h_red
    assert h[14] in h_red

def test_reduce_points_ceil(module5):
    """interior_keepが正しく切り上げられるかテスト"""
    h = np.arange(30, dtype=float)
    b = np.sin(np.linspace(0, 2 * np.pi, 30))
    # Bmax at idx=7, Bmin at idx=22
    
    # interior_keep=5.1 は 6 として扱われるべき
    h_red, b_red = module5.reduce_points(h, b, interior_keep=5.1)
    
    # 期待される合計点数 = 4 (必須) + 6 (内部) = 10
    assert len(h_red) == 10

def test_reduce_points_not_enough_available(module5):
    """利用可能な点より多くの点を要求した場合のテスト"""
    h = np.arange(10, dtype=float)
    b = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1])
    # 必須点: 0(始点/Bmin), 9(終点), 5(Bmax) -> 3点
    # 利用可能な内部点: 10 - 3 = 7点
    
    # 10点の内部点を要求するが、7点しかないので全て選択されるべき
    h_red, b_red = module5.reduce_points(h, b, interior_keep=10)
    
    # 期待される合計点数 = 3 (必須) + 7 (利用可能な全て) = 10
    assert len(h_red) == 10

def test_reduce_points_small_array(module5):
    """4点未満の配列が与えられた場合のテスト"""
    h = np.array([1, 2, 3])
    b = np.array([0, 1, 0])
    h_red, b_red = module5.reduce_points(h, b, interior_keep=5)
    
    # 元の配列がそのまま返されるべき
    np.testing.assert_array_equal(h, h_red)
    np.testing.assert_array_equal(b, b_red)