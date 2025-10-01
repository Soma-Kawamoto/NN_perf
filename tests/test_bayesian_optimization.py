import pytest
import numpy as np
import pandas as pd

# conftest.pyで定義されたフィクスチャを使用

@pytest.fixture
def mock_gpr_model(mocker):
    """ダミーのGPRモデルを返すフィクスチャ"""
    model = mocker.MagicMock()
    # model.predict(X) が (mean, variance) のタプルを返すように設定
    def predict_dummy(X):
        n_points = X.shape[0]
        mean = np.sin(X[:, 1]).reshape(-1, 1)  # Bに依存するようなダミーの平均
        variance = np.ones((n_points, 1)) * 0.1 # ダミーの分散
        return mean, variance
    model.predict.side_effect = predict_dummy
    return model

def test_calculate_acquisition_function_ucb(bayesian_opt_module, mock_gpr_model):
    """獲得関数(UCB)が正しく標準偏差を返すかテスト"""
    X_test = np.array([[1.0, 0.5], [1.0, 0.8]])
    f_max = 1.0 # UCBでは未使用
    
    # UCBは標準偏差 * kappa を返す
    # model.predictは分散0.1を返すので、標準偏差はsqrt(0.1)
    expected_ucb = np.sqrt(0.1)
    
    # kappa=1 (デフォルト) の場合
    ucb_values = bayesian_opt_module.calculate_acquisition_function(mock_gpr_model, X_test, f_max, acq_type='UCB', kappa=1.0)
    
    assert ucb_values.shape == (2, 1)
    np.testing.assert_allclose(ucb_values, expected_ucb)

def test_find_closest_point_in_original_data(bayesian_opt_module):
    """
    理想的なB値に最も近い点を元のデータセットから正しく見つけられるかテスト
    """
    df_original = pd.DataFrame({
        'B': [0.1, 0.2, 0.3, 0.4, 0.5],
        'H': [10,  20,  30,  40,  50]
    })

    # ケース1: 理想点(0.32)に最も近いのは0.3
    b_ideal_1 = 0.32
    closest_row_1 = bayesian_opt_module.find_closest_point_in_original_data(b_ideal_1, df_original)
    assert closest_row_1['B'] == 0.3
    assert closest_row_1['H'] == 30

    # ケース2: 理想点(0.48)に最も近いのは0.5
    b_ideal_2 = 0.48
    closest_row_2 = bayesian_opt_module.find_closest_point_in_original_data(b_ideal_2, df_original)
    assert closest_row_2['B'] == 0.5
    assert closest_row_2['H'] == 50
