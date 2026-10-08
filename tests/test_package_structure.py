"""Keep package boundaries acyclic, including lazy and type-only imports."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src/stavellum"
ALLOWED = {
    "domain": set(),
    "graphics": {"domain"},
    "importers": {"domain"},
    "engraving": {"domain", "graphics"},
    "presentation": {"domain", "graphics", "engraving"},
    "rendering": {"domain", "graphics", "presentation"},
    "exporting": {"domain", "graphics", "presentation", "rendering"},
    "ui": {"domain", "graphics", "importers", "engraving", "presentation",
           "rendering", "exporting", "demo"},
}


def dependencies():
    modules = {}
    for path in ROOT.rglob("*.py"):
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(["stavellum", *parts])] = path
    edges = {name: set() for name in modules}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                        if node.level else node.module or "")
                candidates = [base + "." + alias.name if base + "." + alias.name in modules
                              else base for alias in node.names]
            else:
                continue
            edges[name].update(candidate for candidate in candidates if candidate in modules)
    return edges


def test_package_root_contains_only_entrypoints_and_demo():
    assert {path.name for path in ROOT.glob("*.py")} == {
        "__init__.py", "__main__.py", "cli.py", "demo.py",
    }


def test_all_imports_follow_responsibility_boundaries():
    for name, targets in dependencies().items():
        source = name.split(".")[1] if "." in name else ""
        if source not in ALLOWED:
            continue
        for target in targets:
            group = target.split(".")[1] if "." in target else ""
            assert not group or group == source or group in ALLOWED[source], f"{name} -> {target}"


def test_module_graph_has_no_cycles_including_lazy_and_type_imports():
    edges = dependencies()
    visited = set()

    def visit(name, stack):
        assert name not in stack, " -> ".join([*stack, name])
        if name in visited:
            return
        for target in sorted(edges[name]):
            visit(target, [*stack, name])
        visited.add(name)

    for name in sorted(edges):
        visit(name, [])
