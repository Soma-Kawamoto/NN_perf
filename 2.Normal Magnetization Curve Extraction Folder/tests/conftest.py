# pytestを使用。
#! py -m pytest "2.Normal Magnetization Curve Extraction Folder/tests/"
import pytest
import sys
import os
import importlib.util
from unittest.mock import MagicMock

# このconftest.pyがある'tests'フォルダの親ディレクトリを取得
# (.../2.Normal Magnetization Curve Extraction Folder/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 'src'フォルダの絶対パスを構築
src_path = os.path.join(project_root, 'src')
# Pythonの検索パスの先頭に'src'フォルダを追加
sys.path.insert(0, src_path)

def import_script(script_name):
    """srcフォルダから指定されたスクリプトをモジュールとしてインポートするヘルパー関数"""
    script_path = os.path.join(src_path, script_name)
    # スクリプト名からモジュール名を生成 (例: '1. script.py' -> '1_script')
    module_name = os.path.splitext(script_name)[0].replace('. ', '_').replace(' ', '_')
    
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    
    # インポート時にGUIウィンドウが表示されるのを防ぐため、matplotlib.pyplotをモック化
    sys.modules['matplotlib.pyplot'] = MagicMock()
    
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="session")
def module1():
    """1. Raw Normal Magnetization Curve.py をモジュールとして提供するフィクスチャ"""
    return import_script("1. Raw Normal Magnetization Curve.py")

@pytest.fixture(scope="session")
def module2():
    """2. Akima spline interpolation.py をモジュールとして提供するフィクスチャ"""
    return import_script("2. Akima spline interpolation.py")