"""Receiver type inference for Java ``method_invocation`` nodes.

A call such as ``repo.getProduct()`` is recorded with the raw target
``repo.getProduct``, which never matches a node ID. This module finds the
declared type of the receiver (``ProductRepository``) so that
``trailmark.analysis.receivers.resolve_receiver_types`` can link the call
to ``ProductRepository.getProduct`` once all files are merged.

Inference is purely syntactic and scope-based: locals, parameters, fields
(including those of enclosing classes), ``this``/``super``,
``new X().m()`` and static ``X.m()`` / ``Outer.Inner.m()``. Inferred types
are simple names: generic arguments and package qualifiers are dropped,
but nested type qualifiers are kept (``Comment.Id``).
"""

from __future__ import annotations

import re
from enum import Enum

from tree_sitter import Node

from trailmark.parsers._common import node_text

_CLASS_LIKE = frozenset(
    {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
    }
)
_METHOD_LIKE = frozenset(
    {
        "method_declaration",
        "constructor_declaration",
        "compact_constructor_declaration",
    }
)
_BLOCK_LIKE = frozenset(
    {
        "block",
        "constructor_body",
        "switch_block_statement_group",
        "switch_rule",
    }
)

_CAPITALIZED = re.compile(r"[A-Z]")
_NESTED_TYPE_PATH = re.compile(r"[A-Z]\w*(?:\.[A-Z]\w*)+", re.ASCII)
_WHITESPACE = re.compile(r"\s+")


class _Lookup(Enum):
    """Marker for a name with no declaration in scope."""

    NOT_DECLARED = "not_declared"


_NOT_DECLARED = _Lookup.NOT_DECLARED

# A declared type name, ``None`` for a declaration whose type is unknown or
# not a class type, or ``_NOT_DECLARED`` when no declaration was found.
type _LookupResult = str | None | _Lookup


def java_receiver_type(call: Node) -> str | None:
    """Return the declared type name of a call's receiver, if it can be inferred."""
    obj = call.child_by_field_name("object")
    if obj is None:
        return None
    if obj.type == "this":
        return _enclosing_class_name(call)
    if obj.type == "super":
        return _enclosing_superclass(call)
    if obj.type == "identifier":
        return _identifier_type(node_text(obj), call)
    if obj.type == "field_access":
        return _field_access_type(obj, call)
    if obj.type == "object_creation_expression":
        return simple_type_name(obj.child_by_field_name("type"))
    if obj.type == "parenthesized_expression" and obj.named_children:
        inner = obj.named_children[0]
        if inner.type == "object_creation_expression":
            return simple_type_name(inner.child_by_field_name("type"))
    return None


def simple_type_name(type_node: Node | None) -> str | None:
    """Reduce a type node to the simple name used to look up its declaration.

    ``List<Foo>`` becomes ``List`` and ``a.b.Foo`` becomes ``Foo``, while
    trailing capitalized segments are kept because they name nested types
    (``Outer.Inner``). Primitives, arrays and ``var`` yield ``None``.
    """
    node = type_node
    while node is not None:
        if node.type == "type_identifier":
            text = node_text(node)
            return None if text == "var" else text
        if node.type == "generic_type":
            node = next((c for c in node.named_children if c.type != "type_arguments"), None)
        elif node.type == "annotated_type":
            node = node.named_children[-1] if node.named_children else None
        elif node.type == "scoped_type_identifier":
            parts = _WHITESPACE.sub("", node_text(node)).split(".")
            start = len(parts) - 1
            while start > 0 and _CAPITALIZED.match(parts[start - 1]):
                start -= 1
            return ".".join(parts[start:]) or None
        else:
            return None
    return None


def _identifier_type(name: str, call: Node) -> str | None:
    """Type of a bare identifier receiver: a variable, else a capitalized class name."""
    found = _lookup_variable(name, call)
    if found is not _NOT_DECLARED:
        return found
    # Not a variable in scope: by Java convention a capitalized identifier
    # names a class, making this a static call.
    return name if _CAPITALIZED.match(name) else None


def _field_access_type(obj: Node, call: Node) -> str | None:
    """Type of a ``this.field`` receiver or a static ``Outer.Inner`` path."""
    target = obj.child_by_field_name("object")
    field = obj.child_by_field_name("field")
    if target is not None and target.type == "this" and field is not None:
        found = _field_type(node_text(field), call)
        return None if found is _NOT_DECLARED else found
    # ``Outer.Inner.m()`` is a static call on a nested type, unless
    # ``Outer`` is a variable.
    path = _WHITESPACE.sub("", node_text(obj))
    if _NESTED_TYPE_PATH.fullmatch(path) and (
        _lookup_variable(path.split(".")[0], call) is _NOT_DECLARED
    ):
        return path
    return None


def _declared_type(type_node: Node | None, declarator: Node | None) -> str | None:
    """Type of a declared variable, following ``var x = new T()`` to ``T``."""
    is_identifier = type_node is not None and type_node.type == "type_identifier"
    if is_identifier and node_text(type_node) == "var":
        value = declarator.child_by_field_name("value") if declarator is not None else None
        if value is not None and value.type == "object_creation_expression":
            return simple_type_name(value.child_by_field_name("type"))
        return None
    return simple_type_name(type_node)


def _declarator_named(declaration: Node, name: str) -> Node | None:
    """Return the last declarator of ``declaration`` that declares ``name``."""
    found: Node | None = None
    for declarator in declaration.children_by_field_name("declarator"):
        name_node = declarator.child_by_field_name("name")
        if name_node is not None and node_text(name_node) == name:
            found = declarator
    return found


def _has_name(node: Node, name: str) -> bool:
    """Return whether ``node``'s ``name`` field is ``name``."""
    name_node = node.child_by_field_name("name")
    return name_node is not None and node_text(name_node) == name


def _lookup_variable(name: str, start: Node) -> _LookupResult:
    """Find the type of variable ``name`` as seen from ``start``.

    Walks outwards through the enclosing scopes: block locals declared
    before the use, loop, catch, resource and lambda variables, method
    parameters, then the fields of each enclosing class.
    """
    child = start
    scope = start.parent
    while scope is not None:
        found = _lookup_in_scope(name, scope, child)
        if found is not _NOT_DECLARED:
            return found
        child = scope
        scope = scope.parent
    return _NOT_DECLARED


def _lookup_in_scope(name: str, scope: Node, child: Node) -> _LookupResult:
    """Look up ``name`` among the declarations ``scope`` makes visible to ``child``."""
    if scope.type in _BLOCK_LIKE:
        return _block_local_type(name, scope, child)
    if scope.type == "for_statement":
        init = scope.child_by_field_name("init")
        if init is not None and init.type == "local_variable_declaration":
            declarator = _declarator_named(init, name)
            if declarator is not None:
                return _declared_type(init.child_by_field_name("type"), declarator)
        return _NOT_DECLARED
    if scope.type == "enhanced_for_statement":
        if _has_name(scope, name):
            return _declared_type(scope.child_by_field_name("type"), None)
        return _NOT_DECLARED
    if scope.type == "catch_clause":
        return _catch_parameter_type(name, scope)
    if scope.type == "try_with_resources_statement":
        return _resource_type(name, scope)
    if scope.type == "lambda_expression":
        return _lambda_parameter_type(name, scope)
    if scope.type in _METHOD_LIKE:
        return _method_parameter_type(name, scope)
    if scope.type in _CLASS_LIKE:
        return _class_field_type(scope, name)
    return _NOT_DECLARED


def _block_local_type(name: str, block: Node, child: Node) -> _LookupResult:
    """Type of a local declared in ``block`` before the statement holding ``child``."""
    found: _LookupResult = _NOT_DECLARED
    for statement in block.named_children:
        if statement.start_byte >= child.start_byte:
            break
        if statement.type != "local_variable_declaration":
            continue
        declarator = _declarator_named(statement, name)
        if declarator is not None:
            found = _declared_type(statement.child_by_field_name("type"), declarator)
    return found


def _catch_parameter_type(name: str, clause: Node) -> _LookupResult:
    """Type of a catch parameter; multi-catch unions are unknown."""
    param = next((c for c in clause.named_children if c.type == "catch_formal_parameter"), None)
    if param is None or not _has_name(param, name):
        return _NOT_DECLARED
    catch_type = next((c for c in param.named_children if c.type == "catch_type"), None)
    types = catch_type.named_children if catch_type is not None else []
    return simple_type_name(types[0]) if len(types) == 1 else None


def _resource_type(name: str, statement: Node) -> _LookupResult:
    """Type of a try-with-resources variable."""
    spec = statement.child_by_field_name("resources")
    if spec is None:
        return _NOT_DECLARED
    for resource in spec.named_children:
        if resource.type == "resource" and _has_name(resource, name):
            return _declared_type(resource.child_by_field_name("type"), resource)
    return _NOT_DECLARED


def _lambda_parameter_type(name: str, lambda_node: Node) -> _LookupResult:
    """Type of an explicitly typed lambda parameter; inferred ones are unknown."""
    params = lambda_node.child_by_field_name("parameters")
    if params is None:
        return _NOT_DECLARED
    if params.type == "identifier":
        return None if node_text(params) == name else _NOT_DECLARED
    for param in params.named_children:
        if param.type == "identifier" and node_text(param) == name:
            return None
        if param.type == "formal_parameter" and _has_name(param, name):
            return simple_type_name(param.child_by_field_name("type"))
    return _NOT_DECLARED


def _method_parameter_type(name: str, method: Node) -> _LookupResult:
    """Type of a method or constructor parameter; varargs are arrays."""
    params = method.child_by_field_name("parameters")
    if params is None:
        return _NOT_DECLARED
    for param in params.named_children:
        if param.type == "formal_parameter" and _has_name(param, name):
            return simple_type_name(param.child_by_field_name("type"))
        if param.type == "spread_parameter":
            declarator = next(
                (c for c in param.named_children if c.type == "variable_declarator"),
                None,
            )
            if declarator is not None and _has_name(declarator, name):
                return None
    return _NOT_DECLARED


def _class_field_type(cls: Node, name: str) -> _LookupResult:
    """Type of a field (or record component) declared directly in ``cls``."""
    if cls.type == "record_declaration":
        components = next((c for c in cls.named_children if c.type == "formal_parameters"), None)
        for component in components.named_children if components is not None else []:
            if component.type == "formal_parameter" and _has_name(component, name):
                return simple_type_name(component.child_by_field_name("type"))
    body = cls.child_by_field_name("body")
    for member in _body_members(body):
        if member.type not in ("field_declaration", "constant_declaration"):
            continue
        declarator = _declarator_named(member, name)
        if declarator is not None:
            return _declared_type(member.child_by_field_name("type"), declarator)
    return _NOT_DECLARED


def _body_members(body: Node | None) -> list[Node]:
    """Named members of a type body, including those of an enum's body declarations."""
    members: list[Node] = []
    for member in body.named_children if body is not None else []:
        if member.type == "enum_body_declarations":
            members.extend(member.named_children)
        else:
            members.append(member)
    return members


def _field_type(name: str, start: Node) -> _LookupResult:
    """Type of ``this.name``: a field of the innermost enclosing class."""
    scope = start.parent
    while scope is not None:
        if scope.type in _CLASS_LIKE:
            return _class_field_type(scope, name)
        scope = scope.parent
    return _NOT_DECLARED


def _enclosing_class(start: Node) -> Node | None:
    """Return the innermost named type declaration around ``start``."""
    scope = start.parent
    while scope is not None:
        if scope.type in _CLASS_LIKE:
            return scope
        # Anonymous class bodies change what ``this`` means; give up
        # rather than guess.
        if (
            scope.type == "class_body"
            and scope.parent is not None
            and scope.parent.type == "object_creation_expression"
        ):
            return None
        scope = scope.parent
    return None


def _enclosing_class_name(start: Node) -> str | None:
    """Name of the enclosing type, qualified by its enclosing types (``Comment.Id``)."""
    names: list[str] = []
    cls = _enclosing_class(start)
    while cls is not None:
        name_node = cls.child_by_field_name("name")
        if name_node is None:
            return None
        names.append(node_text(name_node))
        cls = _enclosing_class(cls)
    return ".".join(reversed(names)) or None


def _enclosing_superclass(start: Node) -> str | None:
    """Simple name of the enclosing class's superclass, for ``super.m()``."""
    cls = _enclosing_class(start)
    superclass = cls.child_by_field_name("superclass") if cls is not None else None
    if superclass is None or not superclass.named_children:
        return None
    return simple_type_name(superclass.named_children[0])
