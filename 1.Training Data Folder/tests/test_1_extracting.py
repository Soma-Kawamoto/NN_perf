import pytest
import pandas as pd
import numpy as np
import os

# conftest.pyで定義された 'module1' フィクスチャを使用します。

def test_excel_column_to_index(module1):
    """Excel列名を0ベースのインデックスに正しく変換できるかテスト"""
    assert module1.excel_column_to_index('A') == 0
    assert module1.excel_column_to_index('Z') == 25
    assert module1.excel_column_to_index('AA') == 26
    assert module1.excel_column_to_index('AZ') == 51
    assert module1.excel_column_to_index('IV') == 255

def test_process_single_frequency_success(module1, mocker):
    """単一の周波数処理が正常に動作するかのテスト"""
    # --- 準備 (Arrange) ---
    mock_makedirs = mocker.patch('os.makedirs')
    mock_read_excel = mocker.patch('pandas.read_excel')
    mock_to_excel = mocker.patch('pandas.DataFrame.to_excel')

    # pd.read_excelが返すダミーのDataFrameを準備
    mock_df = pd.DataFrame(np.arange(10000).reshape(100, 100))
    mock_read_excel.return_value = mock_df

    # テスト用のパス設定
    fake_input_path = "/fake/input"
    fake_output_path = "/fake/output"
    mat_name = "test_mat"
    freq = 50

    # --- 実行 (Act) ---
    module1.process_single_frequency(
        mat_name=mat_name,
        freq=freq,
        start_row=2,
        end_row=101,
        start_col_name='C', # index 2
        end_col_name='H', # index 7
        input_base_path=fake_input_path,
        output_data_root=fake_output_path
    )

    # --- 検証 (Assert) ---
    # フォルダ作成が呼ばれたか (os.path.joinでOS依存のパスを生成)
    expected_output_dir = os.path.join(fake_output_path, mat_name, f"{freq}Hz")
    mock_makedirs.assert_called_with(expected_output_dir, exist_ok=True)
    
    # Excel読み込みが呼ばれたか (os.path.joinでOS依存のパスを生成)
    expected_input_file = os.path.join(fake_input_path, f"{mat_name}_ring_{freq}Hz_12.5mm.xls")
    mock_read_excel.assert_called_with(expected_input_file, sheet_name='data', header=None)
    
    # C列(2)からH列(7)まで3列おきに処理されるので、col_idx = 2, 5 の2回ループするはず
    assert mock_to_excel.call_count == 2
    
    # 1回目のExcel書き出しを検証
    call_args_1, call_kwargs_1 = mock_to_excel.call_args_list[0]
    expected_output_file_1 = os.path.join(expected_output_dir, "Bm0.05hys_50hz.xlsx")
    assert call_args_1[0] == expected_output_file_1
    assert not call_kwargs_1['header']
    assert not call_kwargs_1['index']
    
    # 2回目のExcel書き出しを検証
    call_args_2, call_kwargs_2 = mock_to_excel.call_args_list[1]
    expected_output_file_2 = os.path.join(expected_output_dir, "Bm0.1hys_50hz.xlsx")
    assert call_args_2[0] == expected_output_file_2

def test_process_single_frequency_file_not_found(module1, mocker, capsys):
    """入力ファイルが見つからない場合のエラー処理をテスト"""
    # --- 準備 (Arrange) ---
    mocker.patch('os.makedirs')
    mocker.patch('pandas.read_excel', side_effect=FileNotFoundError)

    # --- 実行 (Act) ---
    module1.process_single_frequency(
        mat_name="test_mat", freq=50, start_row=2, end_row=101,
        start_col_name='C', end_col_name='E',
        input_base_path="/fake/input", output_data_root="/fake/output"
    )
    
    # --- 検証 (Assert) ---
    # エラーメッセージが出力されることを確認
    captured = capsys.readouterr()
    assert "エラー: 入力ファイルが見つかりません" in captured.out

def test_process_single_frequency_index_error(module1, mocker, capsys):
    """列インデックスが範囲外になった場合の警告をテスト"""
    # --- 準備 (Arrange) ---
    mocker.patch('os.makedirs')
    # 5列しかないDataFrameを返すように設定
    mock_df = pd.DataFrame(np.arange(500).reshape(100, 5))
    mocker.patch('pandas.read_excel', return_value=mock_df)

    # --- 実行 (Act) ---
    # I列(idx=8)まで処理しようとする -> col_idx=6のループでIndexErrorが発生するはず
    module1.process_single_frequency(
        mat_name="test_mat", freq=50, start_row=2, end_row=101,
        start_col_name='A', end_col_name='I',
        input_base_path="/fake/input", output_data_root="/fake/output"
    )
    
    # --- 検証 (Assert) ---
    # 警告メッセージが出力されることを確認
    captured = capsys.readouterr()
    assert "警告: 列インデックス" in captured.out