import pytest
import numpy as np

# conftest.pyで定義された 'module4' フィクスチャを使用します。

@pytest.fixture
def loop_data():
    """テスト用の単純なループデータを作成するフィクスチャ"""
    M = 100
    t = np.linspace(0, 2 * np.pi, M, endpoint=False)
    b = np.sin(t)
    # 位相をずらしてループ形状にする
    h = np.sin(t - np.pi / 4)
    return h, b, M

def test_extract_upper_branch_normal_case(module4, loop_data):
    """BmaxのインデックスがBminより小さい通常のケースをテスト"""
    h, b, M = loop_data
    # Bの最大点は t=pi/2 (idx=25), 最小点は t=3pi/2 (idx=75)
    i_max = np.argmax(b)
    i_min = np.argmin(b)
    assert i_max == 25
    assert i_min == 75
    assert i_max < i_min

    h_branch, b_branch = module4.extract_upper_branch(h, b)

    # 期待されるインデックスは 25 から 75 まで
    expected_indices = np.arange(i_max, i_min + 1)
    np.testing.assert_array_equal(h_branch, h[expected_indices])
    np.testing.assert_array_equal(b_branch, b[expected_indices])
    
    # 抽出されたデータの始点と終点がBmaxとBminになっているか確認
    assert b_branch[0] == pytest.approx(np.max(b))
    assert b_branch[-1] == pytest.approx(np.min(b))

def test_extract_upper_branch_wrap_around_case(module4, loop_data):
    """BmaxのインデックスがBminより大きく、配列の終端をまたぐケースをテスト"""
    h, b, M = loop_data
    # データを50個ずらして、Bmaxが後半、Bminが前半に来るようにする
    b_rolled = np.roll(b, 50)
    h_rolled = np.roll(h, 50)
    
    i_max = np.argmax(b_rolled) # 元の25+50=75あたり
    i_min = np.argmin(b_rolled) # 元の75+50-100=25あたり
    assert i_max == 75
    assert i_min == 25
    assert i_max > i_min

    h_branch, b_branch = module4.extract_upper_branch(h_rolled, b_rolled)
    
    # 期待されるインデックスは [75, 76, ..., 99, 0, 1, ..., 25]
    expected_indices = np.concatenate([np.arange(i_max, M), np.arange(0, i_min + 1)])
    np.testing.assert_array_equal(h_branch, h_rolled[expected_indices])
    np.testing.assert_array_equal(b_branch, b_rolled[expected_indices])

    # 抽出されたデータの始点と終点がBmaxとBminになっているか確認
    assert b_branch[0] == pytest.approx(np.max(b_rolled))
    assert b_branch[-1] == pytest.approx(np.min(b_rolled))
