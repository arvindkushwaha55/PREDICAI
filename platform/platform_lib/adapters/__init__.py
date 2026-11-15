"""Dataset adapters — the only code that knows each dataset's file layout."""
from .base import DatasetAdapter
from .ai4i import AI4IAdapter
from .secom import SECOMAdapter
from .cmapss import CMapssAdapter

__all__ = ["DatasetAdapter", "AI4IAdapter", "SECOMAdapter", "CMapssAdapter"]
