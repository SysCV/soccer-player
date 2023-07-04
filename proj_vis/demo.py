"""Demo file for python code style.

You should delete this after adding your python code and unit tests.
"""

from typing import List


def add(numbers: List[float]) -> float:
    """Adding two numbers."""
    return sum(numbers)


def main() -> None:
    """Main function.

    This is necessary if you want to run this file in command line.
    """
    print(add([1, 2, 3]))


if __name__ == "__main__":
    # Always use a main function. Don't write the code at the file level.
    main()
