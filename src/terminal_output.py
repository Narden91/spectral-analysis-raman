from __future__ import annotations

import builtins
import logging
from typing import Mapping

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table

    _CONSOLE: Console | None = Console()
except Exception:
    _CONSOLE = None


def print_kv_table(title: str, values: Mapping[str, object]) -> None:
    if _CONSOLE is not None:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Key", style="bold")
        table.add_column("Value")
        for key, value in values.items():
            table.add_row(str(key), str(value))
        _CONSOLE.print(table)
        return

    logger.info("%s", title)
    for key, value in values.items():
        logger.info("  - %s: %s", key, value)


def print_status(message: str) -> None:
    """Emit a concise user-facing status line outside the verbose logger."""
    if _CONSOLE is not None:
        _CONSOLE.print(message)
        return
    builtins.print(message)
