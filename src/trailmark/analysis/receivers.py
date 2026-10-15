"""Resolve ``receiver.method()`` calls using receiver types recorded by parsers.

Parsers attach a ``receiver_type`` attribute (a simple type name, possibly
qualified by enclosing types such as ``Outer.Inner``) to call edges whose
receiver type they could infer. After all files are merged, this pass maps
the type name to a class-like node and rewrites the edge to the method on
that type, or on the nearest supertype that declares it.

Rewritten edges keep their confidence and gain ``raw_target`` (the original
call text, as on unresolved proxies) and ``resolved_by = receiver_type``.
Anything ambiguous is left untouched and later becomes an unresolved proxy.
"""

from __future__ import annotations

import dataclasses
from collections import deque

from trailmark.models.edges import CodeEdge, EdgeKind
from trailmark.models.graph import CodeGraph
from trailmark.models.nodes import NodeKind

RECEIVER_TYPE_ATTR = "receiver_type"
RESOLVED_BY_ATTR = "resolved_by"

_TYPE_KINDS = frozenset(
    {
        NodeKind.CLASS,
        NodeKind.INTERFACE,
        NodeKind.ENUM,
        NodeKind.STRUCT,
        NodeKind.TRAIT,
        NodeKind.CONTRACT,
        NodeKind.LIBRARY,
    }
)
_CALLABLE_KINDS = frozenset({NodeKind.METHOD, NodeKind.FUNCTION})


def resolve_receiver_types(graph: CodeGraph) -> None:
    """Link dangling call edges that carry a ``receiver_type`` to methods.

    The type name is resolved to a unique type of that name, else the one
    in the caller's module, else the one in the caller's package; any
    remaining ambiguity leaves the edge unresolved rather than guessing.
    """
    if not any(_receiver_type(edge) is not None for edge in graph.edges):
        return
    resolver = _ReceiverResolver(graph)
    graph.edges = [resolver.resolve(edge) for edge in graph.edges]


class _ReceiverResolver:
    """Type and supertype indexes over one graph."""

    def __init__(self, graph: CodeGraph) -> None:
        self._nodes = graph.nodes
        # Index by simple name and by every trailing part of the type path
        # within the module, so ``Inner.Deep`` finds ``Outer.Inner.Deep``.
        self._types_by_name: dict[str, list[str]] = {}
        for node_id, unit in graph.nodes.items():
            if unit.kind not in _TYPE_KINDS:
                continue
            self._add_type(unit.name, node_id)
            parts = node_id[node_id.find(":") + 1 :].split(".")
            for start in range(len(parts) - 1):
                self._add_type(".".join(parts[start:]), node_id)
        self._supertypes: dict[str, list[str]] = {}
        for edge in graph.edges:
            if edge.kind in (EdgeKind.INHERITS, EdgeKind.IMPLEMENTS):
                self._supertypes.setdefault(edge.source_id, []).append(edge.target_id)

    def _add_type(self, key: str, node_id: str) -> None:
        ids = self._types_by_name.setdefault(key, [])
        if node_id not in ids:
            ids.append(node_id)

    def resolve(self, edge: CodeEdge) -> CodeEdge:
        """Return ``edge`` linked through its receiver type, or unchanged."""
        if edge.kind != EdgeKind.CALLS or edge.target_id in self._nodes:
            return edge
        type_name = _receiver_type(edge)
        if type_name is None:
            return edge
        type_id = self._resolve_type(type_name, edge.source_id)
        if type_id is None:
            return edge
        target = self._find_method(type_id, edge.target_id.rsplit(".", 1)[-1])
        if target is None:
            return edge
        return dataclasses.replace(
            edge,
            target_id=target,
            attributes=edge.attributes
            + (("raw_target", edge.target_id), (RESOLVED_BY_ATTR, "receiver_type")),
        )

    def _resolve_type(self, name: str, from_id: str) -> str | None:
        """Pick the type ``name`` as seen from ``from_id``."""
        candidates = self._types_by_name.get(name)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        module = _module_of(from_id)
        same_module = [c for c in candidates if _module_of(c) == module]
        if len(same_module) == 1:
            return same_module[0]
        package = _package_of(module)
        same_package = [c for c in candidates if _package_of(_module_of(c)) == package]
        return same_package[0] if len(same_package) == 1 else None

    def _find_method(self, type_id: str, method: str) -> str | None:
        """Search the type, then its supertypes breadth-first, for ``method``."""
        queue = deque([type_id])
        seen = {type_id}
        while queue:
            current = queue.popleft()
            candidate = f"{current}.{method}"
            unit = self._nodes.get(candidate)
            if unit is not None and unit.kind in _CALLABLE_KINDS:
                return candidate
            for supertype in self._supertypes.get(current, []):
                # Parsers point supertypes at ``<own module>:<Name>``; follow
                # them across files by name.
                if supertype in self._nodes:
                    super_id: str | None = supertype
                else:
                    super_name = supertype.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
                    super_id = self._resolve_type(super_name, current)
                if super_id is not None and super_id not in seen:
                    seen.add(super_id)
                    queue.append(super_id)
        return None


def _receiver_type(edge: CodeEdge) -> str | None:
    """Return the ``receiver_type`` attribute of a call edge, if any."""
    if edge.kind != EdgeKind.CALLS:
        return None
    for key, value in edge.attributes:
        if key == RECEIVER_TYPE_ATTR:
            return value if isinstance(value, str) else None
    return None


def _module_of(node_id: str) -> str:
    """Return the module part of a node ID."""
    return node_id.split(":", 1)[0]


def _package_of(module_id: str) -> str:
    """Return the package of a dotted module ID (``""`` at the top level)."""
    return module_id.rsplit(".", 1)[0] if "." in module_id else ""
