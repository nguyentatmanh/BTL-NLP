"""Compatibility package for the flattened source layout.

This exposes subpackages such as ``viet_qa.api`` and ``viet_qa.models``
from the parent ``src`` directory without forcing a larger refactor.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]
