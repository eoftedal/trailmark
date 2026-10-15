"""Top-level code and anonymous callbacks in JavaScript/TypeScript modules.

The JavaScript and TypeScript parsers extract named functions, methods,
and functions assigned to a variable. In a typical Express module

    const app = express()
    app.get('/product/:id', (req, res) => { ... getProductById(id) ... })
    app.listen(3000)

that alone extracts nothing: the route handler is an anonymous callback
and the wiring lives in top-level statements. This pass runs after the
regular visitor and

- turns each function expression (and object-literal method) found in a
  top-level statement into a function node ``module:<label>@<line>``
  (e.g. ``server:app.get@6``), or ``module:default`` for an anonymous
  default export, tagged with ``anonymous``, ``callback_of`` (the callee
  it is passed to), ``route`` (a leading string argument, e.g.
  ``/product/:id``), and ``registered_on_factory`` when the callee's
  receiver was assigned from a call like ``express()`` / ``Router()`` /
  ``require('express')()`` at top level;
- records calls made directly by top-level statements as calls from the
  module node.

Calls inside a callback belong to that callback (nested callbacks
included, matching how calls inside named functions are attributed).
Functions the regular visitor already extracts (top-level
``const f = () => ...`` and ``f = function () {}``) are left alone.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from tree_sitter import Node

from trailmark.models.graph import CodeGraph
from trailmark.models.nodes import Attribute
from trailmark.parsers._common import extract_call_name, node_text

AddCallback = Callable[[Node, str, str, tuple[Attribute, ...]], None]
"""``(func_node, func_id, name, attributes)``: create a function node and its call edges."""

AddModuleCalls = Callable[[list[tuple[str, Node]]], None]
"""Add call edges from the module node."""

_FUNCTION_EXPR_TYPES = frozenset(
    {"arrow_function", "function", "function_expression", "generator_function"}
)

# Function expression types the regular visitor extracts when assigned to
# a name (it does not extract generator expressions).
_VISITOR_EXPR_TYPES = frozenset({"arrow_function", "function", "function_expression"})

_DECLARATION_TYPES = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "import_statement",
        "ambient_declaration",
        "module",
        "internal_module",
    }
)

_VARIABLE_DECL_TYPES = frozenset({"lexical_declaration", "variable_declaration"})

_STRING_TYPES = frozenset({"string", "template_string"})

_WHITESPACE = re.compile(r"\s+")

_REQUIRE_MEMBER = re.compile(r"^require\((['\"`])([^'\"`]+)\1\)\.(.+)$")


def extract_top_level_code(
    root: Node,
    module_id: str,
    graph: CodeGraph,
    add_callback: AddCallback,
    add_module_calls: AddModuleCalls,
) -> None:
    """Extract anonymous callbacks and module-level calls from ``root``.

    Uses an explicit stack instead of recursion to avoid stack overflow
    on deeply nested ASTs from untrusted source files.
    """
    factories = _factory_assignments(root)
    module_calls: list[tuple[str, Node]] = []
    for statement in _top_level_statements(root):
        stack: list[Node] = [statement]
        while stack:
            node = stack.pop()
            if node.type in _FUNCTION_EXPR_TYPES or _is_object_method(node):
                if not _extracted_by_visitor(node):
                    _add_callback(node, module_id, graph, factories, add_callback)
                # The callback (or the already-extracted function) owns
                # everything inside it.
                continue
            if node.type in ("class", "class_declaration"):
                continue
            if node.type == "call_expression":
                name = extract_call_name(node)
                if name:
                    module_calls.append((name, node))
            stack.extend(reversed(node.children))
    if module_calls:
        add_module_calls(module_calls)


def _top_level_statements(root: Node) -> list[Node]:
    """Top-level statements that may contain code, unwrapping ``export``."""
    out: list[Node] = []
    for child in root.named_children:
        if child.type == "export_statement":
            out.extend(
                inner for inner in child.named_children if inner.type not in _DECLARATION_TYPES
            )
        elif child.type not in _DECLARATION_TYPES and child.type != "comment":
            out.append(child)
    return out


def _extracted_by_visitor(func: Node) -> bool:
    """Whether the regular visitor already extracts ``func`` as a named function.

    That covers ``const name = () => ...`` and ``name = function () {}``
    (identifier targets, not generators) when they are top-level
    statements.
    """
    parent = func.parent
    if parent is None or func.type not in _VISITOR_EXPR_TYPES:
        return False
    if parent.type == "variable_declarator":
        return (
            parent.child_by_field_name("value") == func
            and parent.child_by_field_name("name") is not None
            and _is_top_level(parent.parent)
        )
    if parent.type == "assignment_expression":
        left = parent.child_by_field_name("left")
        statement = parent.parent
        return (
            parent.child_by_field_name("right") == func
            and left is not None
            and left.type == "identifier"
            and statement is not None
            and statement.type == "expression_statement"
            and _is_top_level(statement)
        )
    return False


def _is_top_level(statement: Node | None) -> bool:
    container = statement.parent if statement is not None else None
    if container is None:
        return False
    if container.type == "program":
        return True
    grandparent = container.parent
    return (
        container.type == "export_statement"
        and grandparent is not None
        and grandparent.type == "program"
    )


def _is_object_method(node: Node) -> bool:
    """``{ get(id) { ... } }``: a method in an object literal.

    Class bodies are skipped before the walk gets here.
    """
    parent = node.parent
    return node.type == "method_definition" and parent is not None and parent.type == "object"


def _factory_assignments(root: Node) -> dict[str, str]:
    """Map ``const app = express()`` to ``app -> express`` (top level only)."""
    out: dict[str, str] = {}
    for child in root.named_children:
        if child.type in _VARIABLE_DECL_TYPES:
            _record_factories(child, out)
        elif child.type == "export_statement":
            for inner in child.named_children:
                if inner.type in _VARIABLE_DECL_TYPES:
                    _record_factories(inner, out)
    return out


def _record_factories(decl: Node, out: dict[str, str]) -> None:
    for declarator in decl.named_children:
        if declarator.type != "variable_declarator":
            continue
        name = declarator.child_by_field_name("name")
        value = declarator.child_by_field_name("value")
        if value is not None and value.type == "await_expression":
            value = value.named_children[0] if value.named_children else None
        if (
            name is not None
            and name.type == "identifier"
            and value is not None
            and value.type == "call_expression"
        ):
            callee = _factory_name(value)
            if callee:
                out[node_text(name)] = callee


def _factory_name(call: Node) -> str:
    """Name of the factory ``call`` invokes, normalizing CommonJS.

    ``require('express')()`` becomes ``express`` and
    ``require('express').Router()`` becomes ``express.Router``.
    """
    func = call.child_by_field_name("function")
    if func is not None and func.type == "call_expression":
        required = _required_module(func)
        if required is not None:
            return required
    callee = _WHITESPACE.sub("", node_text(func)) if func is not None else ""
    match = _REQUIRE_MEMBER.match(callee)
    if match is not None:
        return f"{match.group(2)}.{match.group(3)}"
    return extract_call_name(call)


def _required_module(call: Node) -> str | None:
    """``require('x')`` becomes ``x``; anything else ``None``."""
    func = call.child_by_field_name("function")
    if func is None or node_text(func) != "require":
        return None
    arguments = call.child_by_field_name("arguments")
    args = arguments.named_children if arguments is not None else []
    if len(args) == 1 and args[0].type in _STRING_TYPES:
        return _strip_quotes(node_text(args[0]))
    return None


def _add_callback(
    func: Node,
    module_id: str,
    graph: CodeGraph,
    factories: dict[str, str],
    add_callback: AddCallback,
) -> None:
    line = func.start_point.row + 1
    attributes: list[Attribute] = [("anonymous", True)]
    label = "<anonymous>"
    parent = func.parent
    grandparent = parent.parent if parent is not None else None
    called_by = (
        grandparent if grandparent is not None and grandparent.type == "call_expression" else None
    )
    if parent is None:
        pass
    elif parent.type == "arguments" and called_by is not None:
        callee = _compact_field(called_by, "function")
        label = callee or label
        attributes.append(("callback_of", callee))
        args = parent.named_children
        first = args[0] if args else None
        if first is not None and first != func and first.type in _STRING_TYPES:
            attributes.append(("route", _strip_quotes(node_text(first))))
        if "." in callee:
            factory = factories.get(callee.split(".", 1)[0])
            if factory is not None:
                attributes.append(("registered_on_factory", factory))
    elif func.type == "method_definition":
        label = _key_label(func.child_by_field_name("name")) or label
    elif parent.type == "pair":
        label = _key_label(parent.child_by_field_name("key")) or label
    elif parent.type == "variable_declarator":
        label = _compact_field(parent, "name") or label
    elif parent.type == "assignment_expression":
        label = _compact_field(parent, "left") or label
    elif parent.type == "parenthesized_expression" and called_by is not None:
        label = "<iife>"
    if parent is not None and parent.type == "export_statement":
        # ``export default <function>`` is named ``default``, matching how
        # named default exports are referred to (and what the Next.js
        # Pages API detector expects).
        base = f"{module_id}:default"
    else:
        base = f"{module_id}:{label}@{line}"
    func_id = base
    n = 2
    while func_id in graph.nodes:
        func_id = f"{base}#{n}"
        n += 1
    add_callback(func, func_id, func_id[len(module_id) + 1 :], tuple(attributes))


def _key_label(key: Node | None) -> str:
    """Object key label: string keys unquoted, other keys compacted."""
    if key is None:
        return ""
    if key.type in _STRING_TYPES:
        return _strip_quotes(node_text(key))
    return _WHITESPACE.sub("", node_text(key))


def _compact_field(node: Node, field: str) -> str:
    """Source text of ``node``'s ``field`` child with all whitespace removed."""
    child = node.child_by_field_name(field)
    return _WHITESPACE.sub("", node_text(child)) if child is not None else ""


def _strip_quotes(text: str) -> str:
    return text[1:-1] if len(text) >= 2 else text
