"""The coupling rule, made checkable.

The coefficient must not depend on any particular scoring model. That is a
design commitment, so it gets a test rather than a paragraph: the core module
imports numpy and nothing else, and the scoring package is reachable only
through the optional plausible-value layer.
"""

import ast
import importlib.metadata
import pathlib

import prerelation
from prerelation import core

CORE = pathlib.Path(core.__file__)
PKG = pathlib.Path(prerelation.__file__).parent


def top_level_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_core_imports_numpy_only():
    assert top_level_imports(CORE) <= {"numpy", "__future__"}


def test_scoring_package_is_only_reached_from_the_pv_layer():
    offenders = []
    for path in sorted(PKG.glob("*.py")):
        if path.name == "pv.py":
            continue
        text = path.read_text(encoding="utf-8")
        # A prose mention is fine; an import is not.
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                if any(a.name.startswith("cogtraitmodel") for a in node.names):
                    offenders.append(path.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("cogtraitmodel"):
                    offenders.append(path.name)
    assert offenders == []


def test_pv_imports_the_scoring_package_lazily():
    """Importing prerelation must not pull in the optional dependency."""
    text = pathlib.Path(PKG / "pv.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = [a.name for a in getattr(node, "names", [])]
            if module.startswith("cogtraitmodel") or any(
                n.startswith("cogtraitmodel") for n in names
            ):
                # must sit inside a function body, not at module level
                assert node.col_offset > 0, "cogtraitmodel imported at module level"


def test_version_matches_the_installed_distribution():
    """Compare the package against the BUILD, not against a third copy of the
    number. The previous form asserted equality with the literal "0.2.0" and so
    stayed green while pyproject.toml moved to 0.3.0 and then 0.3.1."""
    assert prerelation.__version__ == importlib.metadata.version("prerelation")
