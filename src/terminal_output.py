from __future__ import annotations

import logging
from typing import Iterator, Mapping, Sequence, TypeVar

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.progress import track
    from rich.table import Table

    _CONSOLE: Console | None = Console()
except Exception:
    _CONSOLE = None

_T = TypeVar("_T")


def rich_enabled() -> bool:
    return _CONSOLE is not None


def print_phase(title: str, subtitle: str | None = None) -> None:
    if _CONSOLE is not None:
        _CONSOLE.rule(f"[bold cyan]{title}[/bold cyan]")
        if subtitle:
            _CONSOLE.print(subtitle, style="dim")
        return

    logger.info("=== %s ===", title)
    if subtitle:
        logger.info("%s", subtitle)


def print_step(message: str) -> None:
    if _CONSOLE is not None:
        _CONSOLE.print(f"[bold green]>>[/bold green] {message}")
        return

    logger.info(">> %s", message)


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


def print_artifact_table(title: str, artifacts: Mapping[str, str]) -> None:
    if _CONSOLE is not None:
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Artifact", style="bold")
        table.add_column("Path")
        for name, path in artifacts.items():
            table.add_row(str(name), str(path))
        _CONSOLE.print(table)
        return

    logger.info("%s", title)
    for name, path in artifacts.items():
        logger.info("  - %s: %s", name, path)


class _BufferingLogHandler(logging.Handler):
    """Capture log records emitted while a rich ``track`` is live.

    The handler installs itself on the root logger when ``progress_iter``
    enters the rich branch, and removes itself on exit. Captured records
    are flushed through ``console.log`` after the corresponding iteration
    step yields, so per-fold ``logger.info`` lines land cleanly under the
    bar instead of overwriting it.

    The handler's level is ``NOTSET`` so it inherits the root logger's
    effective level at install time — that way it doesn't accidentally
    raise the bar from WARNING to INFO and start swallowing warnings that
    weren't visible before.
    """

    def __init__(self, console: Console) -> None:
        super().__init__(level=logging.NOTSET)
        self._console = console
        self._buffer: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(record)

    def flush_now(self) -> None:
        records, self._buffer = self._buffer, []
        for record in records:
            try:
                msg = self.format(record)
            except Exception:
                msg = record.getMessage()
            self._console.log(msg)


def progress_iter(
    items: Sequence[_T],
    description: str,
    transient: bool = True,
) -> Iterator[_T]:
    """Yield items with a progress indicator.

    Parameters
    ----------
    items:
        Sequence to iterate over.
    description:
        Bar prefix.
    transient:
        Only used with rich. When ``True`` (default for short loops), the
        bar is removed from the terminal when iteration finishes so that
        per-iteration ``logger.info`` lines land cleanly below it. Set to
        ``False`` for long-running outer loops where you want a persistent
        visual reference.
    """
    if _CONSOLE is not None:
        handler = _BufferingLogHandler(_CONSOLE)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            for item in track(
                items,
                description=description,
                console=_CONSOLE,
                transient=transient,
            ):
                yield item
                handler.flush_now()
        finally:
            try:
                root_logger.removeHandler(handler)
            except ValueError:
                pass
            handler.flush_now()
        return

    total = len(items)
    for idx, item in enumerate(items, start=1):
        if idx == 1 or idx == total or idx % 10 == 0:
            logger.info("%s: %d/%d", description, idx, total)
        yield item
