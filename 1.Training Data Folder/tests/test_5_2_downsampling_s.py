import pytest
import numpy as np

# conftest.pyで定義された 'module5_2' フィクスチャを使用します。

@pytest.fixture
def circle_loop_data():
    """テスト用の標準的な円形ループを準備するフィクスチャ"""
    M = 100
    t = np.linspace(0, 2 * np.pi, M, endpoint=False)
    h = np.cos(t)
    b = np.sin(t)
    # Bmax at t=pi/2 (idx=25), Bmin at t=3pi/2 (idx=75)
    return h, b, M

def test_reduce_by_arclength_basic(module5_2, circle_loop_data):
    """基本的な線長ベースの削減をテスト"""
    h, b, M = circle_loop_data
    # interior_keep=3 は、BmaxとBminの間に3つの点をサンプリングすることを意味する
    h_red, b_red = module5_2.reduce_points_by_arclength(h, b, interior_keep=3)

    # 必須点: 0(始点), 99(終点), 25(Bmax), 75(Bmin)
    assert h[0] in h_red
    assert h[99] in h_red
    assert h[25] in h_red
    assert h[75] in h_red
    
    # 期待される合計点数 = 4 (必須) + 3 (サンプリング) = 7 (重複がなければ)
    assert len(h_red) <= 7
    assert len(h_red) >= 4 # 少なくとも必須点は含まれる

def test_reduce_by_arclength_zero_interior(module5_2, circle_loop_data):
    """interior_keep=0 の場合、必須点のみが返されることをテスト"""
    h, b, M = circle_loop_data
    h_red, b_red = module5_2.reduce_points_by_arclength(h, b, interior_keep=0)
    
    # 必須点 {0, 25, 75, 99} の4点のみになるはず
    assert len(h_red) == 4
    
    expected_h = h[[0, 25, 75, 99]]
    # ソートして比較
    np.testing.assert_allclose(np.sort(h_red), np.sort(expected_h))

def test_reduce_by_arclength_wrap_around(module5_2, circle_loop_data):
    """上側カーブが配列の終端をまたぐ場合の削減をテスト"""
    h, b, M = circle_loop_data
    h_rolled = np.roll(h, 50)
    b_rolled = np.roll(b, 50)
    # Bmax at idx=75, Bmin at idx=25
    
    h_red, b_red = module5_2.reduce_points_by_arclength(h_rolled, b_rolled, interior_keep=3)
    
    # 必須点: 0, 99, 75(Bmax), 25(Bmin)
    assert h_rolled[0] in h_red
    assert h_rolled[99] in h_red
    assert h_rolled[75] in h_red
    assert h_rolled[25] in h_red
    assert len(h_red) <= 7

def test_reduce_by_arclength_flat_line(module5_2):
    """Bが平坦な直線の場合にエラーなく動作するかテスト"""
    h = np.linspace(0, 10, 100)
    b = np.ones(100) * 5.0 # Bが一定
    
    # ゼロ除算エラーが発生しないことを確認
    h_red, b_red = module5_2.reduce_points_by_arclength(h, b, interior_keep=5)
    
    # Bmax/Bminが同じインデックスになるため、必須点は始点と終点のみ
    assert len(h_red) == 2
    np.testing.assert_array_equal(h_red, h[[0, 99]])