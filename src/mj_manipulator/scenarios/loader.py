# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Scenario discovery and loading.

A scenario is a Python module with a ``scene = {...}`` dict and optional
user-facing functions. This module finds scenario files on disk and
imports them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def discover(search_dirs: list[Path]) -> dict[str, Path]:
    """Find scenario files in the given directories.

    A scenario file is any non-dunder, non-underscore ``*.py`` whose
    source contains a top-level ``scene =`` assignment. The content
    check lets scenarios live alongside helper modules in the same
    directory without polluting the picker.

    Args:
        search_dirs: Directories to search for scenarios.

    Returns:
        Mapping of scenario name (file stem) to its path.
    """
    found: dict[str, Path] = {}
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.py")):
            if p.name.startswith("_"):
                continue
            if not _looks_like_scenario(p):
                continue
            if p.stem not in found:
                found[p.stem] = p
    return found


def _looks_like_scenario(path: Path) -> bool:
    """Cheap check: does the file have a top-level ``scene =`` assignment?

    Reads the file textually (no import) so we can filter without
    executing potentially-expensive scenario code at discovery time.
    Only matches unindented ``scene =`` lines (module-level assignment).
    """
    try:
        text = path.read_text()
    except OSError:
        return False
    for line in text.splitlines():
        # Require no indent (module-level assignment)
        if line.startswith(" ") or line.startswith("\t"):
            continue
        code = line.split("#", 1)[0]  # strip comment
        if code.startswith("scene") and "=" in code:
            return True
    return False


def describe(path: Path) -> str:
    """Extract the one-line description from a scenario file's docstring.

    Reads the first triple-quoted docstring without importing the module
    (so we can list scenarios without triggering side effects).

    Returns:
        The first line of the module docstring, or the file stem if no
        docstring is found.
    """
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    quote = line[:3]
                    content = line[3:]
                    if content.endswith(quote):
                        return content[:-3].strip()
                    return content.strip()
                if line and not line.startswith("#"):
                    break
    except OSError:
        pass
    return path.stem


def load(name_or_path: str, search_dirs: list[Path] | None = None) -> ModuleType:
    """Load a scenario by name or file path.

    If ``name_or_path`` is a file path, load that file directly.
    Otherwise, look up the name in ``search_dirs``.

    Args:
        name_or_path: Scenario name (stem) or path to a ``.py`` file.
        search_dirs: Directories to search when resolving a bare name.

    Returns:
        The imported scenario module.

    Raises:
        ValueError: If the scenario can't be found.
    """
    path = Path(name_or_path)
    if path.is_file():
        return _load_file(path)

    if search_dirs:
        available = discover(search_dirs)
        if name_or_path in available:
            return _load_file(available[name_or_path])

    names = list(discover(search_dirs or []).keys())
    raise ValueError(f"Scenario '{name_or_path}' not found. Available: {', '.join(names) if names else 'none'}")


def _load_file(path: Path) -> ModuleType:
    """Load a Python file as a module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load scenario from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_interactive(search_dirs: list[Path]) -> ModuleType | None:
    """Present a numbered picker and load the user's choice.

    Returns the loaded scenario module, or ``None`` if the user bails out.
    """
    scenarios = discover(search_dirs)
    items = list(scenarios.items())

    if not items:
        print("No scenarios found.")
        return None

    print("\nWhat scenario would you like?\n")
    for i, (name, path) in enumerate(items, 1):
        print(f"  {i}. {describe(path)}")
    print()

    choice = input("> ").strip()
    try:
        idx = int(choice) - 1
    except ValueError:
        return None
    if 0 <= idx < len(items):
        _, path = items[idx]
        return _load_file(path)
    return None
