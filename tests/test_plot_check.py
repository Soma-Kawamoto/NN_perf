import pytest
import os

# conftest.pyで定義されたフィクスチャを使用

def test_find_data_file(plot_check_module, tmp_path):
    """
    find_data_file関数が、ファイル名の小数点以下の桁数を正しく処理して
    ファイルを検索できるかテストする。
    """
    # --- Arrange (準備) ---
    # is_reduced_file=False のテスト用ディレクトリとファイル
    original_dir = tmp_path / "original" / "test_mat" / "50Hz"
    original_dir.mkdir(parents=True)
    (original_dir / "Bm1.0hys_50hz.xlsx").touch()
    (original_dir / "Bm0.85hys_50hz.xlsx").touch()

    # is_reduced_file=True のテスト用ディレクトリとファイル
    reduced_dir = tmp_path / "reduced" / "test_mat" / "50"
    reduced_dir.mkdir(parents=True)
    (reduced_dir / "Bm1.5hys_50hz_reduct.xlsx").touch()

    # --- Act & Assert (実行と検証) ---

    # ケース1: 元データ、振幅が整数.0
    path1 = plot_check_module.find_data_file(str(tmp_path / "original"), "test_mat", 50, 1.0, is_reduced_file=False)
    assert os.path.basename(path1) == "Bm1.0hys_50hz.xlsx"

    # ケース2: 元データ、振幅が小数点以下2桁
    path2 = plot_check_module.find_data_file(str(tmp_path / "original"), "test_mat", 50, 0.85, is_reduced_file=False)
    assert os.path.basename(path2) == "Bm0.85hys_50hz.xlsx"

    # ケース3: 削減後データ
    path3 = plot_check_module.find_data_file(str(tmp_path / "reduced"), "test_mat", 50, 1.5, is_reduced_file=True)
    assert os.path.basename(path3) == "Bm1.5hys_50hz_reduct.xlsx"

    # ケース4: 存在しないファイル
    path_none = plot_check_module.find_data_file(str(tmp_path / "original"), "test_mat", 50, 9.9, is_reduced_file=False)
    assert path_none is None