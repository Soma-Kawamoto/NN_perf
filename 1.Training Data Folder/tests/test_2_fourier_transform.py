import pytest
import numpy as np

# conftest.pyで定義された 'module2' フィクスチャを使用します。

@pytest.fixture
def base_signal():
    """テスト用の基本信号を準備するフィクスチャ"""
    N = 1024
    t = np.linspace(0, 1, N, endpoint=False)
    # 直流(1.0), 奇数次(1,3,5), 偶数次(2,4) を含む信号
    signal = (1.0 + 
              2.0 * np.sin(2 * np.pi * 1 * t) + 
              0.5 * np.sin(2 * np.pi * 2 * t) + 
              1.0 * np.sin(2 * np.pi * 3 * t) +
              0.2 * np.sin(2 * np.pi * 4 * t) +
              0.8 * np.sin(2 * np.pi * 5 * t))
    return signal, N

def test_process_hst_b_signals_removes_dc_and_even_harmonics(module2, base_signal):
    """直流成分と偶数次高調波が除去されることをテスト"""
    test_signal, N = base_signal
    h_proc, b_proc = module2.process_hst_b_signals(test_signal, test_signal, N=N)

    # --- 検証 ---
    assert h_proc is not None
    assert b_proc is not None

    # 直流成分がゼロになっているか
    assert np.mean(h_proc) == pytest.approx(0.0, abs=1e-6)
    assert np.mean(b_proc) == pytest.approx(0.0, abs=1e-6)

    # 周波数領域で確認
    h_coeffs = np.fft.rfft(h_proc)
    
    # 偶数次高調波がゼロになっているか (インデックス 2, 4, 6...)
    for k_harmonic in range(2, 10, 2):
        assert np.abs(h_coeffs[k_harmonic]) == pytest.approx(0.0, abs=1e-6)
    
    # 奇数次高調波が残っているか (インデックス 1, 3, 5...)
    for k_harmonic in range(1, 6, 2):
        assert np.abs(h_coeffs[k_harmonic]) > 1e-3

def test_process_hst_b_signals_with_high_freq_cutoff(module2, base_signal):
    """高周波カットオフが正しく機能するかテスト"""
    test_signal, N = base_signal
    # 5次以上の高調波をカットオフ
    h_proc, b_proc = module2.process_hst_b_signals(
        test_signal, test_signal, N=N, high_freq_cutoff_order=5
    )
    
    # --- 検証 ---
    h_coeffs = np.fft.rfft(h_proc)
    
    # 1次と3次は残っている
    assert np.abs(h_coeffs[1]) > 1e-3
    assert np.abs(h_coeffs[3]) > 1e-3
    
    # 5次以降はゼロになっている
    assert np.abs(h_coeffs[5]) == pytest.approx(0.0, abs=1e-6)
    assert np.abs(h_coeffs[7]) == pytest.approx(0.0, abs=1e-6)

def test_process_hst_b_signals_wrong_length(module2, base_signal):
    """信号長がNと異なる場合にNoneを返すかテスト"""
    test_signal, N = base_signal
    short_signal = test_signal[:512]
    h_proc, b_proc = module2.process_hst_b_signals(short_signal, short_signal, N=N)
    assert h_proc is None
    assert b_proc is None