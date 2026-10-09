from __future__ import annotations

import ast

from datadec.models.strict_check import REFERENCE_SCRIPT


def _imported_modules(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_reference_script_exists_outside_the_package() -> None:
    assert REFERENCE_SCRIPT.is_file()
    assert "scripts" in REFERENCE_SCRIPT.parts
    assert "datadec" not in REFERENCE_SCRIPT.parts[:-1][-2:]


def test_reference_script_imports_nothing_from_datadec() -> None:
    tree = ast.parse(REFERENCE_SCRIPT.read_text())
    assert "datadec" not in _imported_modules(tree)
