"""Plain assert-based tests for the dialogs in widgets.py.

Reads the source rather than building the dialogs, so it runs without a
display. A ``command=self._handler`` naming a method the class lacks raises
inside the dialog's ``__init__``, before ``wait_window``: the window stays
open, but its caller never sees the result.

Run with: python tests/test_widgets.py
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILURES = []


def check(name, cond):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        FAILURES.append(name)


def self_commands(cls):
    """Names of the self attributes passed as ``command=`` anywhere in *cls*."""
    names = set()
    for node in ast.walk(cls):
        if not isinstance(node, ast.keyword) or node.arg != 'command':
            continue
        value = node.value
        if (isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == 'self'):
            names.add(value.attr)
    return names


def self_attributes(cls):
    """Names of the methods and instance attributes *cls* defines."""
    names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(cls):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and node.value.id == 'self'):
            names.add(node.attr)
    return names


with open(os.path.join(ROOT, 'widgets.py'), encoding='utf-8') as f:
    tree = ast.parse(f.read())

print("button commands")
classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
check("found dialog classes", len(classes) > 5)
for cls in classes:
    missing = sorted(self_commands(cls) - self_attributes(cls))
    check(f"{cls.name}: every command exists{missing or ''}", not missing)


if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All tests passed.")
