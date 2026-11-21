"""PySide6 graphical interface."""

from __future__ import annotations

__all__ = ["main", "MainWindow"]


def __getattr__(name):
    # Imported lazily so that the CLI and tests never need Qt installed.
    if name in __all__:
        from .app import MainWindow, main

        return {"main": main, "MainWindow": MainWindow}[name]
    raise AttributeError(name)
