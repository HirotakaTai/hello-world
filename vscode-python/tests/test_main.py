"""Test for hello_world function."""

from src.main import hello_world


def test_hello_world() -> None:
    """Test that hello_world() returns 'Hello, World!'."""
    assert hello_world() == "Hello, World!"
