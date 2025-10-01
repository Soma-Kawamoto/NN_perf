import pytest
import sys
import os
import importlib.util
from unittest.mock import MagicMock, patch

# このconftest.pyがある'tests'フォルダの親ディレクトリを取得
# (.../1.Training Data Folder/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 'src'フォルダの絶対パスを構築
src_path = os.path.join(project_root, 'src')
# Pythonの検索パスの先頭に'src'フォルダを追加
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def import_script(script_name, module_name_override=None):
    """srcフォルダから指定されたスクリプトをモジュールとしてインポートするヘルパー関数"""
    script_path = os.path.join(src_path, script_name)
    # スクリプト名からモジュール名を生成 (例: '1. script.py' -> 'script_1')
    module_name = module_name_override or os.path.splitext(script_name)[0].replace('. ', '_').replace(' ', '_').replace('(', '').replace(')', '')

    # モジュールが既に読み込まれているか確認
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module {module_name} from {script_path}")
    module = importlib.util.module_from_spec(spec)
    
    # exec_moduleの前にsys.modulesに登録する
    sys.modules[module_name] = module

    # インポート時にGUIウィンドウが表示されるのを防ぐため、依存ライブラリをモック化
    with patch.dict('sys.modules', {'matplotlib.pyplot': MagicMock()}):
        spec.loader.exec_module(module)
    
    return module

@pytest.fixture(scope="session")
def module1():
    return import_script("1.高橋先生からもらったデータのextracting(xlsx).py", "module1")

@pytest.fixture(scope="session")
def module2():
    return import_script("2.Fourier Transform Correction.py", "module2")

@pytest.fixture(scope="session")
def module3():
    return import_script("3.how_reduct.py", "module3")

@pytest.fixture(scope="session")
def module4():
    return import_script("4.Ascending branch of the hysteresis loop.py", "module4")

@pytest.fixture(scope="session")
def module5():
    return import_script("5.Downsampling.py", "module5")

@pytest.fixture(scope="session")
def module5_2():
    return import_script("5.2.Downsampling at s.py", "module5_2")

@pytest.fixture(scope="session")
def module6():
    return import_script("6.Reference data.py", "module6")

@pytest.fixture(scope="session")
def module7():
    return import_script("7.Iron loss to pptx.py", "module7")

@pytest.fixture(scope="session")
def module7_2():
    return import_script("7.2.Iron loss to pptx 縦軸統一.py", "module7_2")

@pytest.fixture(scope="session")
def module8():
    return import_script("8.Hc_Br.py", "module8")