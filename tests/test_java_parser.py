"""Tests for the Java language parser."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from trailmark.models.edges import EdgeConfidence, EdgeKind
from trailmark.models.graph import CodeGraph
from trailmark.models.nodes import NodeKind, NodeOrigin
from trailmark.parsers.java.parser import JavaParser

SAMPLE_CODE = """\
import java.util.List;
import java.io.IOException;

/** Defines animal behavior. */
interface Animal {
    String speak();
}

/** A dog that extends Pet and implements Animal. */
class Dog extends Pet implements Animal {

    private String name;

    public Dog(String name) {
        this.name = name;
    }

    public String speak() {
        return "woof";
    }

    public boolean fetch(String item, int count) {
        if (item == null || item.isEmpty()) {
            throw new IllegalArgumentException("empty");
        }
        for (int i = 0; i < count; i++) {
            System.out.println(item);
        }
        while (count > 10) {
            count--;
        }
        return true;
    }
}

class Puppy extends Dog {

    public Puppy(String name) {
        super(name);
    }

    public String play() {
        if (name != null) {
            return "playing";
        }
        return "sleeping";
    }
}

enum AnimalType {
    DOG,
    CAT,
    BIRD
}
"""


def _parse_sample() -> tuple[JavaParser, CodeGraph]:
    parser = JavaParser()
    with tempfile.NamedTemporaryFile(
        suffix=".java",
        mode="w",
        delete=False,
    ) as f:
        f.write(SAMPLE_CODE)
        f.flush()
        graph = parser.parse_file(f.name)
    os.unlink(f.name)
    return parser, graph


class TestJavaParserNodes:
    def test_finds_module(self) -> None:
        _, graph = _parse_sample()
        modules = [n for n in graph.nodes.values() if n.kind == NodeKind.MODULE]
        assert len(modules) == 1

    def test_finds_interface(self) -> None:
        _, graph = _parse_sample()
        interfaces = [n for n in graph.nodes.values() if n.kind == NodeKind.INTERFACE]
        names = {i.name for i in interfaces}
        assert "Animal" in names

    def test_finds_classes(self) -> None:
        _, graph = _parse_sample()
        classes = [n for n in graph.nodes.values() if n.kind == NodeKind.CLASS]
        names = {c.name for c in classes}
        assert "Dog" in names
        assert "Puppy" in names

    def test_finds_enum(self) -> None:
        _, graph = _parse_sample()
        enums = [n for n in graph.nodes.values() if n.kind == NodeKind.ENUM]
        names = {e.name for e in enums}
        assert "AnimalType" in names

    def test_finds_methods(self) -> None:
        _, graph = _parse_sample()
        methods = [n for n in graph.nodes.values() if n.kind == NodeKind.METHOD]
        names = {m.name for m in methods}
        assert "speak" in names
        assert "fetch" in names
        assert "play" in names


class TestJavaParserParameters:
    def test_typed_parameters(self) -> None:
        _, graph = _parse_sample()
        fetch = next(n for n in graph.nodes.values() if n.name == "fetch")
        assert len(fetch.parameters) == 2
        param_names = {p.name for p in fetch.parameters}
        assert "item" in param_names
        assert "count" in param_names

    def test_parameter_types(self) -> None:
        _, graph = _parse_sample()
        fetch = next(n for n in graph.nodes.values() if n.name == "fetch")
        params = {p.name: p for p in fetch.parameters}
        assert params["item"].type_ref is not None
        assert params["item"].type_ref.name == "String"
        assert params["count"].type_ref is not None
        assert params["count"].type_ref.name == "int"

    def test_return_type(self) -> None:
        _, graph = _parse_sample()
        fetch = next(n for n in graph.nodes.values() if n.name == "fetch")
        assert fetch.return_type is not None
        assert fetch.return_type.name == "boolean"


class TestJavaParserComplexity:
    def test_simple_method_complexity(self) -> None:
        _, graph = _parse_sample()
        speak_methods = [
            n for n in graph.nodes.values() if n.name == "speak" and n.kind == NodeKind.METHOD
        ]
        speak = speak_methods[0]
        assert speak.cyclomatic_complexity == 1

    def test_branching_method_complexity(self) -> None:
        _, graph = _parse_sample()
        fetch = next(n for n in graph.nodes.values() if n.name == "fetch")
        assert fetch.cyclomatic_complexity is not None
        assert fetch.cyclomatic_complexity >= 4

    def test_branches_tracked(self) -> None:
        _, graph = _parse_sample()
        fetch = next(n for n in graph.nodes.values() if n.name == "fetch")
        assert len(fetch.branches) > 0


class TestJavaParserEdges:
    def test_contains_edges(self) -> None:
        _, graph = _parse_sample()
        contains = [e for e in graph.edges if e.kind == EdgeKind.CONTAINS]
        assert len(contains) > 0

    def test_inherits_edge(self) -> None:
        _, graph = _parse_sample()
        inherits = [e for e in graph.edges if e.kind == EdgeKind.INHERITS]
        assert len(inherits) >= 1
        targets = {e.target_id for e in inherits}
        pet_targets = [t for t in targets if t.endswith(":Pet")]
        assert len(pet_targets) >= 1

    def test_implements_edge(self) -> None:
        _, graph = _parse_sample()
        implements = [e for e in graph.edges if e.kind == EdgeKind.IMPLEMENTS]
        assert len(implements) >= 1

    def test_call_edges(self) -> None:
        _, graph = _parse_sample()
        calls = [e for e in graph.edges if e.kind == EdgeKind.CALLS]
        assert len(calls) > 0

    def test_exception_type(self) -> None:
        _, graph = _parse_sample()
        fetch = next(n for n in graph.nodes.values() if n.name == "fetch")
        assert len(fetch.exception_types) >= 1

    def test_edge_confidence(self) -> None:
        _, graph = _parse_sample()
        calls = [e for e in graph.edges if e.kind == EdgeKind.CALLS]
        certain = [e for e in calls if e.confidence == EdgeConfidence.CERTAIN]
        inferred = [e for e in calls if e.confidence == EdgeConfidence.INFERRED]
        assert len(certain) > 0 or len(inferred) > 0


class TestJavaParserDependencies:
    def test_imports_tracked(self) -> None:
        _, graph = _parse_sample()
        assert "java" in graph.dependencies


GENERIC_JAVA_CODE = """\
import java.util.List;
import java.util.Map;

class Container {
    public List<String> getItems(Map<String, Integer> lookup) {
        return null;
    }
}
"""


def _parse_generic() -> CodeGraph:
    parser = JavaParser()
    with tempfile.NamedTemporaryFile(
        suffix=".java",
        mode="w",
        delete=False,
    ) as f:
        f.write(GENERIC_JAVA_CODE)
        f.flush()
        graph = parser.parse_file(f.name)
    os.unlink(f.name)
    return graph


class TestJavaGenericTypes:
    def test_generic_parameter_type(self) -> None:
        graph = _parse_generic()
        get_items = next(n for n in graph.nodes.values() if n.name == "getItems")
        assert len(get_items.parameters) == 1
        param = get_items.parameters[0]
        assert param.type_ref is not None
        assert param.type_ref.name == "Map"
        assert len(param.type_ref.generic_args) == 2

    def test_generic_return_type(self) -> None:
        graph = _parse_generic()
        get_items = next(n for n in graph.nodes.values() if n.name == "getItems")
        assert get_items.return_type is not None
        assert get_items.return_type.name == "List"
        assert len(get_items.return_type.generic_args) == 1


class TestJavaParseDirectory:
    def test_parses_multiple_files(self) -> None:
        parser = JavaParser()
        code_a = "class FromA {}\n"
        code_b = "class FromB {}\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, code in [("a.java", code_a), ("b.java", code_b)]:
                path = os.path.join(tmpdir, name)
                with open(path, "w") as f:
                    f.write(code)
            graph = parser.parse_directory(tmpdir)
        assert graph.language == "java"
        assert graph.root_path == tmpdir
        names = {n.name for n in graph.nodes.values()}
        assert "FromA" in names
        assert "FromB" in names

    def test_ignores_wrong_extensions(self) -> None:
        parser = JavaParser()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "skip.txt")
            with open(path, "w") as f:
                f.write("not source code")
            graph = parser.parse_directory(tmpdir)
        assert len(graph.nodes) == 0


RECORD_JAVA_CODE = """\
public record Comment(Id id, Body body) implements Serializable {
    public Comment {
        if (id == null) throw new IllegalArgumentException("id");
    }
    public Body body() { return body; }
    public static record Id(String value) {
        public static Id generate() { return new Id("x"); }
    }
    public record Body(String text) {}
    class Inner { void m() { this.n(); } void n() {} }
}
enum Color {
    RED, GREEN;
    int code() { return Util.f(); }
}
class Util { static int f() { return 0; } }
"""


@pytest.fixture
def record_graph(tmp_path: Path) -> CodeGraph:
    path = tmp_path / "Comment.java"
    path.write_text(RECORD_JAVA_CODE)
    return JavaParser().parse_file(str(path))


class TestJavaRecords:
    def test_record_is_class_tagged_as_record(self, record_graph: CodeGraph) -> None:
        record = record_graph.nodes["Comment:Comment"]
        assert record.kind == NodeKind.CLASS
        assert ("java_type_kind", "record") in record.attributes

    def test_record_implements_edge(self, record_graph: CodeGraph) -> None:
        implements = [
            e
            for e in record_graph.edges
            if e.kind == EdgeKind.IMPLEMENTS and e.source_id == "Comment:Comment"
        ]
        assert len(implements) == 1

    def test_synthetic_accessor_per_component(self, record_graph: CodeGraph) -> None:
        accessor = record_graph.nodes["Comment:Comment.id"]
        assert accessor.kind == NodeKind.METHOD
        assert accessor.origin == NodeOrigin.SYNTHETIC
        assert accessor.attributes == (("synthetic_reason", "record_accessor"),)
        assert accessor.return_type is not None
        assert accessor.return_type.name == "Id"
        assert record_graph.nodes["Comment:Comment.Id.value"].origin == NodeOrigin.SYNTHETIC

    def test_declared_accessor_wins_over_synthetic(self, record_graph: CodeGraph) -> None:
        accessor = record_graph.nodes["Comment:Comment.body"]
        assert accessor.origin == NodeOrigin.SOURCE
        assert accessor.attributes == ()

    def test_accessors_are_contained_by_record(self, record_graph: CodeGraph) -> None:
        contains = {
            (e.source_id, e.target_id) for e in record_graph.edges if e.kind == EdgeKind.CONTAINS
        }
        assert ("Comment:Comment", "Comment:Comment.id") in contains
        assert ("Comment:Comment", "Comment:Comment.body") in contains

    def test_compact_constructor(self, record_graph: CodeGraph) -> None:
        ctor = record_graph.nodes["Comment:Comment.Comment"]
        assert ctor.kind == NodeKind.METHOD
        assert ctor.parameters == ()
        assert [t.name for t in ctor.exception_types] == ["IllegalArgumentException"]
        assert ctor.cyclomatic_complexity == 2


class TestJavaNestedTypes:
    def test_nested_types_use_outer_qualified_ids(self, record_graph: CodeGraph) -> None:
        for node_id in ("Comment:Comment.Id", "Comment:Comment.Body", "Comment:Comment.Inner"):
            assert node_id in record_graph.nodes
        assert record_graph.nodes["Comment:Comment.Id"].name == "Id"
        assert "Comment:Comment.Id.generate" in record_graph.nodes
        assert "Comment:Comment.Inner.n" in record_graph.nodes

    def test_nested_types_are_contained_by_outer_type(self, record_graph: CodeGraph) -> None:
        contains = {
            (e.source_id, e.target_id) for e in record_graph.edges if e.kind == EdgeKind.CONTAINS
        }
        assert ("Comment:Comment", "Comment:Comment.Id") in contains
        assert ("Comment:Comment.Id", "Comment:Comment.Id.generate") in contains
        assert ("Comment", "Comment:Comment.Id") not in contains

    def test_unqualified_call_in_nested_type_targets_nested_type(
        self, record_graph: CodeGraph
    ) -> None:
        callers = {e.source_id for e in record_graph.edges if e.kind == EdgeKind.CALLS}
        assert "Comment:Comment.Inner.m" in callers

    def test_deeply_nested_types_do_not_recurse(self, tmp_path: Path) -> None:
        depth = 1500
        source = "class C { " * depth + "void m() {}" + " }" * depth
        path = tmp_path / "Deep.java"
        path.write_text(source)
        graph = JavaParser().parse_file(str(path))
        innermost = "Deep:" + ".".join(["C"] * depth)
        assert f"{innermost}.m" in graph.nodes


class TestJavaEnumBodyMethods:
    def test_enum_body_methods_are_extracted(self, record_graph: CodeGraph) -> None:
        code = record_graph.nodes["Comment:Color.code"]
        assert code.kind == NodeKind.METHOD
        contains = {
            (e.source_id, e.target_id) for e in record_graph.edges if e.kind == EdgeKind.CONTAINS
        }
        assert ("Comment:Color", "Comment:Color.code") in contains

    def test_enum_body_method_calls_are_extracted(self, record_graph: CodeGraph) -> None:
        targets = {
            e.target_id
            for e in record_graph.edges
            if e.kind == EdgeKind.CALLS and e.source_id == "Comment:Color.code"
        }
        assert targets == {"Util.f"}
