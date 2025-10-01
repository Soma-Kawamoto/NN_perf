import pytest
import numpy as np
import pandas as pd
import os

# conftest.pyで定義したフィクスチャ 'module2' を使用します。

def test_get_interpolator(module2):
    """get_interpolator関数が正しく補間オブジェクトを生成するかテストします。"""
    # 単純な二次関数 y = x^2
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])

    interp_func = module2.get_interpolator(x, y, method='akima')

    # ケース1: 既知の点での評価
    assert np.isclose(interp_func(2.0), 4.0)

    # ケース2: 補間点での評価 (Akimaは区分3次多項式なので、二次関数を完全に再現はしませんが近い値になるはず)
    assert np.isclose(interp_func(1.5), 2.25, atol=0.1)

    # ケース3: ソートされていない入力でも動作するかテスト
    x_unsorted = np.array([2, 0, 4, 1, 3])
    y_unsorted = np.array([4, 0, 16, 1, 9])
    interp_func_unsorted = module2.get_interpolator(x_unsorted, y_unsorted, method='akima')
    assert np.isclose(interp_func_unsorted(1.5), interp_func(1.5))

    # ケース4: 不明なメソッドでエラーが発生するか
    with pytest.raises(ValueError):
        module2.get_interpolator(x, y, method='unknown_method')

def test_process_single_frequency(module2, tmp_path, mocker):
    """process_single_frequency関数がファイルI/Oと補間を正しく行うかテストします。"""
    # --- 準備 (Arrange) ---
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # ダミーの入力Excelファイルを作成
    df_in = pd.DataFrame({"Amplitude": [0.1, 0.2, 0.3], "Hb": [10, 20, 30], "Bm": [0.1, 0.2, 0.3]})
    input_file = input_dir / "Bm-Hb Curve_test_50hz.xlsx"
    df_in.to_excel(input_file, sheet_name="Bm-Hb Data", index=False)

    # ファイル書き込み関連をモック化
    mock_excel_writer = mocker.patch('pandas.ExcelWriter')
    mock_to_excel = mocker.patch('pandas.DataFrame.to_excel', autospec=True)

    # --- 実行 (Act) ---
    module2.process_single_frequency(
        mat_name="test",
        freq=50,
        method="akima",
        input_dir=str(input_dir),
        output_dir=str(output_dir)
    )

    # --- 検証 (Assert) ---
    # ExcelWriterが正しい出力パスで呼び出されたか
    expected_outfile = os.path.join(str(output_dir), 'Bm-Hb Curve_akima_test_50hz.xlsx')
    mock_excel_writer.assert_called_with(expected_outfile, engine='xlsxwriter')

    # to_excelに渡されたDataFrameの内容を検証
    # autospec=Trueにより、第1引数(self)がDataFrameインスタンスとしてキャプチャされる
    mock_to_excel.assert_called_once()
    df_out = mock_to_excel.call_args.args[0]

    # DataFrameの構造と内容を検証
    assert 'amp_Hb' in df_out.columns
    assert 'amp_Bm' in df_out.columns
    assert 'Hb' in df_out.columns
    assert 'Bm' in df_out.columns
    assert np.allclose(df_out['amp_Hb'], df_in['Hb'])
    assert np.allclose(df_out['amp_Bm'], df_in['Amplitude'])