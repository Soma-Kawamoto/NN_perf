import pytest
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch

# このconftest.pyがある'tests'フォルダの親ディレクトリ(GPR_perf)を取得
project_root = os.path.dirname(os.path.abspath(__file__))
# 'src'フォルダの絶対パスを構築
src_path = os.path.join(project_root, 'src')
# Pythonの検索パスの先頭に'src'フォルダを追加
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def import_script_from_src(script_name, module_name_override=None):
    """srcフォルダから指定されたスクリプトをモジュールとしてインポートするヘルパー関数"""
    script_path = os.path.join(src_path, script_name)
    module_name = module_name_override or os.path.splitext(script_name)[0].replace('. ', '_').replace(' ', '_')

    if not os.path.exists(script_path):
        pytest.skip(f"Test target script not found, skipping: {script_path}")

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {script_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    # GUI関連や時間のかかるライブラリをモック化
    # これにより、テスト実行中にウィンドウが表示されたり、不要な処理が走るのを防ぐ
    mocked_modules = {
        'matplotlib.pyplot': MagicMock(),
        'japanize_matplotlib': MagicMock(),
    }
    with patch.dict('sys.modules', mocked_modules):
        spec.loader.exec_module(module)
    
    return module

@pytest.fixture(scope="session")
def gpr_regression_module():
    """1. GPR Regression.py をモジュールとして提供するフィクスチャ"""
    return import_script_from_src("1. GPR Regression.py", "gpr_regression")

@pytest.fixture(scope="session")
def gpr_reg_sigma_module():
    """1.1.1. GPR Reg +-σ.py をモジュールとして提供するフィクスチャ"""
    return import_script_from_src("1.1.1. GPR Reg +-σ.py", "gpr_reg_sigma")

@pytest.fixture(scope="session")
def gpr_regression_1_2_module():
    """1.2.GPR Regression.py をモジュールとして提供するフィクスチャ"""
    return import_script_from_src("1.2.GPR Regression.py", "gpr_regression_1_2")

@pytest.fixture(scope="session")
def equidistant_ds_module():
    """1.3 Equidistant Downsampling.py をモジュールとして提供するフィクスチャ"""
    return import_script_from_src("1.3 Equidistant Downsampling.py", "equidistant_ds")

@pytest.fixture(scope="session")
def bayesian_opt_module():
    """1.4. bayesian_optimization.py をモジュールとして提供するフィクスチャ"""
    return import_script_from_src("1.4. bayesian_optimization.py", "bayesian_opt")

@pytest.fixture(scope="session")
def module_1_5():
    """1.5.py をモジュールとして提供するフィクスチャ"""
    return import_script_from_src("1.5.py", "module_1_5")

@pytest.fixture(scope="session")
def module_1_6():
    """1.6.誤差項追加.py をモジュールとして提供するフィクスチャ"""
    return import_script_from_src("1.6.誤差項追加.py", "module_1_6")

@pytest.fixture(scope="session")
def plot_check_module():
    """plot_check.py をモジュールとして提供するフィクスチャ"""
    return import_script_from_src("plot_check.py", "plot_check")