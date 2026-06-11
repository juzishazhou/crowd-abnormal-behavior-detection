#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试主检测脚本可以正常 import。

Usage:
    python tests/test_imports.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_import_fall_detection():
    """测试 fall_detection.py 可以被 import"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fall_detection",
        PROJECT_ROOT / "scripts" / "fall_detection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print("✓ scripts/fall_detection.py import OK")
        return True
    except ImportError as e:
        print(f"✗ scripts/fall_detection.py import failed (missing dep?): {e}")
        return False
    except Exception as e:
        print(f"✗ scripts/fall_detection.py import failed: {e}")
        return False


def test_import_running_detection():
    """测试 running_detection.py 可以被 import"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "running_detection",
        PROJECT_ROOT / "scripts" / "running_detection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print("✓ scripts/running_detection.py import OK")
        return True
    except ImportError as e:
        print(f"✗ scripts/running_detection.py import failed (missing dep?): {e}")
        return False
    except Exception as e:
        print(f"✗ scripts/running_detection.py import failed: {e}")
        return False


def test_import_intrusion_detection():
    """测试 intrusion_detection.py 可以被 import"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "intrusion_detection",
        PROJECT_ROOT / "scripts" / "intrusion_detection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print("✓ scripts/intrusion_detection.py import OK")
        return True
    except ImportError as e:
        print(f"✗ scripts/intrusion_detection.py import failed (missing dep?): {e}")
        return False
    except Exception as e:
        print(f"✗ scripts/intrusion_detection.py import failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing imports...")
    print("-" * 40)

    results = [
        test_import_fall_detection(),
        test_import_running_detection(),
        test_import_intrusion_detection(),
    ]

    print("-" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\n⚠️  Some imports failed — missing dependencies?")
        print("   Install: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✓ All imports OK!")
        sys.exit(0)
