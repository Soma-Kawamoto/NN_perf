import pytest
import numpy as np
import pandas as pd

# conftest.pyで定義された 'module6' フィクスチャを使用します。

def test_format_bm_string(module6):
    """Bm値から文字列へのフォーマットをテスト"""
    assert module6.format_bm_string(1.0) == "1.0"
    assert module6.format_bm_string(1.5) == "1.5"
    assert module6.format_bm_string(0.05) == "0.05"
    assert module6.format_bm_string(0.15) == "0.15"

def test_check_monotonical(module6):
    """単調性のチェック機能をテスト"""
    # 単調な信号 (頂点が1つ)
    assert not module6.check_monotonical(np.array([1, 2, 3, 4, 3, 2, 1]))
    assert not module6.check_monotonical(np.sin(np.linspace(0, np.pi, 100)))
    
    # 単調でない信号 (頂点が複数)
    t = np.linspace(0, np.pi, 100)
    non_monotonic_signal = np.sin(t) + 0.1 * np.sin(10 * t)
    assert module6.check_monotonical(non_monotonic_signal)
    
    # エッジケース
    assert not module6.check_monotonical(np.array([1, 2]))
    assert not module6.check_monotonical(np.array([1]))

def test_process_waveform_file(module6, mocker):
    """波形処理とフーリエフィルタリングのロジックをテスト"""
    mock_read_excel = mocker.patch('pandas.read_excel')
    N = 100
    t = np.linspace(0, 1, N, endpoint=False)
    
    # 高次高調波を含む、単調でない信号を作成
    b_wave_orig = np.sin(2 * np.pi * t) + 0.1 * np.sin(2 * np.pi * 11 * t) # 1次 + 11次
    h_wave_orig = np.cos(2 * np.pi * t) + 0.1 * np.cos(2 * np.pi * 11 * t)
    
    mock_df = pd.DataFrame(np.column_stack([h_wave_orig, b_wave_orig]))
    mock_read_excel.return_value = mock_df
    
    b_recon, h_recon = module6.process_waveform_file("fake_path.xlsx", N)
    
    # 再構成された信号は単調になっているはず
    assert not module6.check_monotonical(b_recon)
    
    # 高次成分(11次)が除去されているか確認
    b_fft = np.fft.rfft(b_recon)
    assert np.abs(b_fft[11]) == pytest.approx(0.0, abs=1e-5)
    assert np.abs(b_fft[1]) > 0.1 # 基本波は残っている