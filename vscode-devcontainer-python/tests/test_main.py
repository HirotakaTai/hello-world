"""Tests for main module."""

import pytest

from main import main


def test_main_function_exists():
    """Test that main function exists and is callable."""
    assert callable(main)


def test_main_function_runs():
    """Test that main function runs without error."""
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised {e} unexpectedly!")
