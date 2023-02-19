"""Common utils."""

from pathlib import Path
from typing import Union


def get_abs_path(path: Union[str, Path], fname: str):
    """Get absolute path of a file.

    Absolute path of a file relative to the file from where it
    is referenced.
    """
    return (Path(fname).parent.absolute() / path).absolute()
