from __future__ import annotations

from typing import Mapping

from rich.console import Console
from rich.table import Table

_CONSOLE = Console()


def print_kv_table(title: str, values: Mapping[str, object]) -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(str(key), str(value))
    _CONSOLE.print(table)


def print_status(message: str) -> None:
    """Print a concise user-facing status line outside the verbose logger."""
    _CONSOLE.print(message)
