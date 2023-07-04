"""An example of unit tests based on pytest."""

from .demo import add


def test_add() -> None:
    """Test add."""
    assert add([1, 2, 3]) == 6
