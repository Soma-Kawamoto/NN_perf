import pytest
import torch
import torch.nn as nn
import numpy as np
import configparser
import os

# テスト対象のスクリプトからクラスや関数をインポート
# sys.path.appendを追加して、srcフォルダ内のモジュールを読み込めるようにする
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src import nn_script_for_test as nn_script

# ==============================================================================
# 1. 補助関数のテスト
# ==============================================================================

@pytest.mark.parametrize("name, expected_class", [
    ("relu", nn.ReLU),
    ("ReLU", nn.ReLU),
    ("tanh", nn.Tanh),
    ("sigmoid", nn.Sigmoid),
])
def test_get_activation_function_valid(name, expected_class):
    """
    get_activation_functionが有効な文字列に対して正しいクラスを返すことをテストする。
    """
    activation_func = nn_script.get_activation_function(name)
    assert isinstance(activation_func, expected_class)

def test_get_activation_function_invalid():
    """
    get_activation_functionが無効な文字列に対してValueErrorを発生させることをテストする。
    """
    with pytest.raises(ValueError, match="未対応の活性化関数です: invalid_func"):
        nn_script.get_activation_function("invalid_func")

def test_rmseloss():
    """
    RMSELossクラスが正しくRMSEを計算することをテストする。
    """
    loss_func = nn_script.RMSELoss()
    y_hat = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.5, 2.5, 3.5])
    # 期待値: sqrt( ( (1.0-1.5)^2 + (2.0-2.5)^2 + (3.0-3.5)^2 ) / 3 )
    # = sqrt( (0.25 + 0.25 + 0.25) / 3 ) = sqrt(0.25) = 0.5
    expected_loss = 0.5
    calculated_loss = loss_func(y_hat, y_true).item()
    assert np.isclose(calculated_loss, expected_loss)

# ==============================================================================
# 2. モデルクラスのテスト
# ==============================================================================

def test_fullyconnectednn_creation():
    """
    FullyConnectedNNモデルが指定された構造で正しく構築されることをテストする。
    """
    input_size = 2
    output_size = 1
    hidden_layers = [64, 32]
    activation_func = nn.ReLU()

    model = nn_script.FullyConnectedNN(input_size, output_size, hidden_layers, activation_func)

    # ネットワークの層の数を確認
    # Linear -> ReLU -> Linear -> ReLU -> Linear
    assert len(model.network) == len(hidden_layers) * 2 + 1

    # 各層のサイズとタイプを確認
    assert isinstance(model.network[0], nn.Linear) and model.network[0].in_features == 2 and model.network[0].out_features == 64
    assert isinstance(model.network[1], nn.ReLU)
    assert isinstance(model.network[2], nn.Linear) and model.network[2].in_features == 64 and model.network[2].out_features == 32
    assert isinstance(model.network[3], nn.ReLU)
    assert isinstance(model.network[4], nn.Linear) and model.network[4].in_features == 32 and model.network[4].out_features == 1

    # フォワードパスがエラーなく実行できることを確認
    dummy_input = torch.randn(10, input_size) # バッチサイズ10
    output = model(dummy_input)
    assert output.shape == (10, output_size)

# ==============================================================================
# 3. 設定ファイル読み込みのテスト
# ==============================================================================

@pytest.fixture
def mock_config_file(monkeypatch):
    """
    configparserの読み込みをモックし、テスト用の設定値を返すfixture。
    """
    config = configparser.ConfigParser()
    config['settings'] = {'PERFORM_TRAINING': 'False', 'mat_name': 'TestMat', 'target_freq': '60'}
    config['architecture'] = {'HIDDEN_LAYERS': '128, 64', 'ACTIVATION_FUNC': 'Tanh'}
    config['training'] = {'LEARNING_RATE': '0.01', 'EPOCHS': '50', 'BATCH_SIZE': '16', 'GRAD_CLIP': '0.5', 'LOSS_FUNC': 'RMSE'}
    config['data'] = {'Bmtrain_min': '0.2', 'Bmtrain_max': '1.5', 'train_step': '0.2'}
    config['regression'] = {'Bmreg_min': '0.1', 'Bmreg_max': '1.6', 'step': '0.1'}

    # configparser.ConfigParser.read を何もしないように上書き
    def mock_read(*args, **kwargs):
        pass
    monkeypatch.setattr(configparser.ConfigParser, "read", mock_read)

    # スクリプト内で configparser.ConfigParser() が呼ばれたら、
    # このテスト用に作成したconfigオブジェクトを返すように設定
    monkeypatch.setattr(configparser, "ConfigParser", lambda: config)

def test_config_loading(mock_config_file):
    """
    モックされた設定ファイルから値が正しく読み込まれることをテストする。
    """
    # このテストでは、実際のファイルは読み込まれず、mock_config_fileが使われる
    assert not nn_script.PERFORM_TRAINING
    assert nn_script.mat_name == 'TestMat'
    assert nn_script.target_freq == 60
    assert nn_script.HIDDEN_LAYERS == [128, 64]
    assert nn_script.activation_func_str == 'Tanh'
    assert nn_script.LEARNING_RATE == 0.01
    assert nn_script.EPOCHS == 50
    assert nn_script.LossFunc == 'RMSE'
    assert np.isclose(nn_script.Bmtrain_min, 0.2)
    assert np.isclose(nn_script.Bmreg_max, 1.6)
