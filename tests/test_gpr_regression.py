import pytest
import pandas as pd
import numpy as np
import configparser

# conftest.pyで定義されたフィクスチャを使用

@pytest.fixture
def mock_config():
    """テスト用のダミーconfigparserオブジェクトを作成するフィクスチャ"""
    config = configparser.ConfigParser()
    config['settings'] = {
        'PERFORM_TRAINING': 'True',
        'mat_name': 'test_material',
        'target_freq': '50',
        'kernel_type': 'Matern52',
        'Bmtrain_min': '0.5',
        'Bmtrain_max': '1.5',
        'train_step': '1.0',
        'USE_AKIMA_DATA': 'True',
        'OPTIMIZER': 'lbfgsb',
        'MAX_ITERS': '10',
        'NUM_RESTARTS': '1',
        'Bmreg_min': '0.1',
        'Bmreg_max': '1.8',
        'step': '0.1'
    }
    return config

def test_main_flow(gpr_regression_module, mocker, mock_config):
    """
    スクリプトのメイン処理フローがエラーなく実行されるかをテストする。
    ファイルI/OとGPRモデルの学習・予測をモック化する。
    """
    # --- Arrange (準備) ---

    # 1. configparserのモック
    mocker.patch('configparser.ConfigParser.read')
    mocker.patch('configparser.ConfigParser', return_value=mock_config)

    # 2. ファイル読み込み (pandas.read_excel) のモック
    # 正解データ読み込み用のダミーDataFrame
    df_truth = pd.DataFrame({
        'B': np.linspace(-1.0, 1.0, 21),
        'H_descending': np.linspace(-100, 100, 21)
    })
    # 学習データ読み込み用のダミーDataFrame
    df_train = pd.DataFrame({'H': [10, 20], 'B': [0.1, 0.2]})
    # Akimaデータ読み込み用のダミーDataFrame
    df_akima = pd.DataFrame({'amp_Bm': [0.1, 0.2], 'amp_Hb': [10, 20]})

    # read_excelが呼ばれるたびに、事前に定義したDataFrameを順番に返すように設定
    mock_read_excel = mocker.patch('pandas.read_excel', side_effect=[
        df_truth,  # 1回目: 正解データ
        df_train,  # 2回目: 学習データ (Bm=0.5)
        df_train,  # 3回目: 学習データ (Bm=1.5)
        df_akima,  # 4回目: Akimaデータ
    ])

    # 3. GPyモデルのモック
    mock_gpr_model = mocker.MagicMock()
    # model.predict()がダミーの平均と分散を返すように設定
    dummy_mean = np.random.rand(21, 1)
    dummy_var = np.random.rand(21, 1)
    mock_gpr_model.predict.return_value = (dummy_mean, dummy_var)
    
    # GPy.models.GPRegressionがこのモックオブジェクトを返すように設定
    # モジュール内で 'import GPy' されているため、そのモジュール名をターゲットにする
    mock_gpy_class = mocker.patch(f'{gpr_regression_module.__name__}.GPy.models.GPRegression', return_value=mock_gpr_model)

    # 4. ファイル書き込みとプロットのモック
    mocker.patch('pandas.DataFrame.to_excel')
    mocker.patch('pandas.ExcelWriter')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('os.makedirs')
    mocker.patch('os.path.exists', return_value=True) # 全てのファイルが存在することにする

    # --- Act (実行) ---
    # スクリプトのmain関数を実行。例外が発生しなければ成功とみなす。
    try:
        gpr_regression_module.main()
    except Exception as e:
        pytest.fail(f"スクリプトのメイン処理中に予期せぬ例外が発生しました: {e}")

    # --- Assert (検証) ---    
    # GPy.models.GPRegression が呼び出されたことを確認
    mock_gpy_class.assert_called_once()
    
    # GPRモデルが正しい学習データで初期化されたか検証 (引数の確認)
    # 初期学習データは (0,0), Bm=0.5の2点, Bm=1.5の2点, Akimaの4点 = 9点
    # X_trainの形状が (9, 2) であることを確認
    call_args, _ = mock_gpy_class.call_args
    X_train_passed_to_model = call_args[0]
    assert X_train_passed_to_model.shape[0] == 9

    # モデルの最適化が呼ばれたか検証
    mock_gpr_model.optimize_restarts.assert_called_once()