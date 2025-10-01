import pytest
import os
import pandas as pd
import numpy as np

# conftest.pyで定義したフィクスチャ 'module1' を使用します。
# これにより、テスト対象のスクリプトが 'module1' という名前でインポートされます。

def test_find_input_file(module1, tmp_path):
    """find_input_file関数が正しくファイルを検索できるかテストします。"""
    test_dir = tmp_path
    # テスト用のダミーファイルを作成
    (test_dir / "Bm1.0hys_50hz.xlsx").touch()
    (test_dir / "Bm0.85hys_50hz.xlsx").touch()

    # ケース1: 振幅が整数.0 の場合
    path1 = module1.find_input_file(str(test_dir), 1.0, 50)
    assert os.path.basename(path1) == "Bm1.0hys_50hz.xlsx"

    # ケース2: 振幅が小数点以下2桁の場合
    path2 = module1.find_input_file(str(test_dir), 0.85, 50)
    assert os.path.basename(path2) == "Bm0.85hys_50hz.xlsx"

    # ケース3: ファイルが存在しない場合
    path_none = module1.find_input_file(str(test_dir), 2.0, 50)
    assert path_none is None

def test_calculate_hysteresis_area(module1):
    """シューレース公式による面積計算をテストします。"""
    # 単位正方形 (面積1)
    h_square = np.array([0, 1, 1, 0])
    b_square = np.array([0, 0, 1, 1])
    area = module1.calculate_hysteresis_area(h_square, b_square)
    assert np.isclose(area, 1.0)

    # 単純なひし形ループ (面積100)
    h_loop = np.array([0, 100, 0, -100])
    b_loop = np.array([-0.5, 0, 0.5, 0])
    area_loop = module1.calculate_hysteresis_area(h_loop, b_loop)
    assert np.isclose(area_loop, 100.0)

def test_extract_bmax_record_from_file(module1, mocker):
    """extract_bmax_record_from_file関数が正しくBmaxとHbを抽出するかテストします。"""
    # pd.read_excelをモック化し、テスト用のDataFrameを返すように設定
    mock_df = pd.DataFrame({
        'H': [10, 20, 30, 40, 35, 25],
        'B': [0.1, 0.5, 1.2, 0.8, 0.3, 0.0]
    })
    # Bの最大値は1.2 (index=2), その時のHは30
    mock_read_excel = mocker.patch('pandas.read_excel', return_value=mock_df)

    # テスト対象の関数を実行
    record = module1.extract_bmax_record_from_file("dummy_path.xlsx", amp=1.0)

    # BmaxとHbが正しく抽出できているか検証
    extracted_amp, extracted_h, extracted_b = record

    mock_read_excel.assert_called_once_with("dummy_path.xlsx", header=None, names=['H', 'B'])
    assert np.isclose(extracted_amp, 1.0)
    assert np.isclose(extracted_b, 1.2)
    assert np.isclose(extracted_h, 30)

def test_extract_bmax_record_from_file_handles_exception(module1, mocker, capsys):
    """
    extract_bmax_record_from_fileが例外を適切に処理するかテストします。
    (e.g., Excelファイルに数値でないデータが含まれている場合)
    """
    # pd.read_excelがValueErrorを発生させるようにモック化
    mocker.patch('pandas.read_excel', side_effect=ValueError("Invalid data in file"))

    # テスト対象の関数を実行
    record = module1.extract_bmax_record_from_file("dummy_error_path.xlsx", amp=1.0)

    # 関数がNoneを返すことを確認
    assert record is None

    # 標準出力に警告メッセージが表示されることを確認
    captured = capsys.readouterr()
    assert "Warning: No valid numeric data" in captured.out