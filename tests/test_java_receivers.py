"""Tests for Java receiver-type inference and receiver-type call resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from trailmark.analysis.receivers import resolve_receiver_types
from trailmark.models.edges import CodeEdge, EdgeConfidence, EdgeKind
from trailmark.models.graph import CodeGraph
from trailmark.models.nodes import AttributeValue, CodeUnit, NodeKind, SourceLocation
from trailmark.parse import parse_directory, parse_file
from trailmark.parsers.java.parser import JavaParser
from trailmark.query.api import QueryEngine

REPO_JAVA_CODE = """\
class Repo {
  Item find(Id id) { return null; }
  static Repo create() { return new Repo(); }
  void close() {}
}
class Base { void audit() {} }
class Svc extends Base {
  private final Repo repo;
  private Repo other, third;
  Svc(Repo repo) { this.repo = repo; }
  void run(Repo param, java.util.List<Repo> repos) {
    repo.find(null);
    this.other.find(null);
    param.find(null);
    Repo local = Repo.create();
    local.close();
    var made = new Repo();
    made.close();
    new Repo().close();
    for (Repo r : repos) r.close();
    try (Repo res = Repo.create()) { res.close(); }
    catch (IllegalStateException e) { e.printStackTrace(); }
    repos.forEach(x -> x.close());
    this.helper();
    super.audit();
    audit();
  }
  void shadow() {
    String repo = "";
    repo.find(null);
  }
  void helper() {}
}
"""

RECORD_JAVA_CODE = """\
public record Comment(Id id, Body body) implements java.io.Serializable {
  public Comment {
    if (id == null) throw new IllegalArgumentException("id");
  }
  public Body body() { return body; }
  public static record Id(String value) {
    public static Id generate() { return new Id("x"); }
  }
  public record Body(String text) {}
  class Inner { void m() { Comment.Id.generate(); this.n(); } void n() {} }
}
enum Color { RED, GREEN; int code() { return Util.f(); } }
class Util { static int f() { return 0; } }
class Use {
  void go(Comment c, Comment.Id id) { c.id(); c.body(); id.value(); Comment.Id.generate(); }
}
"""


def _attr(edge: CodeEdge, key: str) -> AttributeValue:
    return dict(edge.attributes).get(key)


def _calls_from(graph: CodeGraph, source_suffix: str) -> dict[str, str]:
    """Map raw call text to the resolved target for calls from ``source_suffix``."""
    calls: dict[str, str] = {}
    for edge in graph.edges:
        if edge.kind != EdgeKind.CALLS or not edge.source_id.endswith(source_suffix):
            continue
        raw = _attr(edge, "raw_target")
        calls[str(raw) if raw is not None else edge.target_id] = edge.target_id
    return calls


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture
def svc_graph(tmp_path: Path) -> CodeGraph:
    path = tmp_path / "Svc.java"
    path.write_text(REPO_JAVA_CODE)
    return parse_file(str(path))


@pytest.fixture
def comment_graph(tmp_path: Path) -> CodeGraph:
    path = tmp_path / "Comment.java"
    path.write_text(RECORD_JAVA_CODE)
    return parse_file(str(path))


class TestReceiverKinds:
    @pytest.mark.parametrize(
        ("raw", "target"),
        [
            ("repo.find", "Svc:Repo.find"),
            ("this.other.find", "Svc:Repo.find"),
            ("param.find", "Svc:Repo.find"),
            ("Repo.create", "Svc:Repo.create"),
            ("local.close", "Svc:Repo.close"),
            ("made.close", "Svc:Repo.close"),
            ("new Repo().close", "Svc:Repo.close"),
            ("r.close", "Svc:Repo.close"),
            ("res.close", "Svc:Repo.close"),
            ("this.helper", "Svc:Svc.helper"),
            ("super.audit", "Svc:Base.audit"),
        ],
    )
    def test_receiver_resolves(self, svc_graph: CodeGraph, raw: str, target: str) -> None:
        assert _calls_from(svc_graph, "Svc.run")[raw] == target

    @pytest.mark.parametrize("raw", ["x.close", "e.printStackTrace"])
    def test_unknown_receiver_stays_unresolved_proxy(self, svc_graph: CodeGraph, raw: str) -> None:
        assert _calls_from(svc_graph, "Svc.run")[raw] == f"proxy.unresolved:{raw}"

    def test_resolved_edge_is_tagged_and_keeps_confidence(self, svc_graph: CodeGraph) -> None:
        edge = next(
            e
            for e in svc_graph.edges
            if e.source_id == "Svc:Svc.run" and e.target_id == "Svc:Repo.find"
        )
        assert _attr(edge, "receiver_type") == "Repo"
        assert _attr(edge, "resolved_by") == "receiver_type"
        assert _attr(edge, "raw_target") == "repo.find"
        assert edge.confidence == EdgeConfidence.INFERRED

    def test_parser_records_receiver_type_without_resolving(self, tmp_path: Path) -> None:
        path = tmp_path / "Svc.java"
        path.write_text(REPO_JAVA_CODE)
        graph = JavaParser().parse_file(str(path))
        edge = next(
            e for e in graph.edges if e.kind == EdgeKind.CALLS and e.target_id == "param.find"
        )
        assert edge.attributes == (("receiver_type", "Repo"),)


class TestShadowing:
    def test_local_declared_in_method_wins_over_field(self, svc_graph: CodeGraph) -> None:
        assert _calls_from(svc_graph, "Svc.shadow")["repo.find"] == "proxy.unresolved:repo.find"

    def test_local_declared_after_use_is_not_in_scope(self, tmp_path: Path) -> None:
        path = tmp_path / "Late.java"
        path.write_text(
            "class Late {\n"
            "  Late f;\n"
            "  void m() {\n"
            "    f.m();\n"
            "    String f = null;\n"
            "    { Late inner = this; inner.m(); }\n"
            "    inner.m();\n"
            "  }\n"
            "}\n"
        )
        graph = parse_file(str(path))
        edges = [
            e for e in graph.edges if e.kind == EdgeKind.CALLS and e.source_id == "Late:Late.m"
        ]
        targets = [e.target_id for e in edges]
        assert targets == [
            "Late:Late.m",
            "Late:Late.m",
            "proxy.unresolved:inner.m",
        ]


class TestCrossFileResolution:
    def test_links_across_files_and_through_inherited_methods(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "repo/BaseRepository.java": (
                    "package repo; public class BaseRepository { public void save(Object o) {} }"
                ),
                "repo/ProductRepository.java": (
                    "package repo; public class ProductRepository extends BaseRepository {"
                    " public Object getProduct(String id) { return null; } }"
                ),
                "web/ProductController.java": (
                    "package web;\n"
                    "public class ProductController {\n"
                    "  private ProductRepository repo;\n"
                    "  public Object byId(String id) {"
                    " var p = repo.getProduct(id); repo.save(p); return p; }\n"
                    "}\n"
                ),
            },
        )
        engine = QueryEngine.from_directory(str(tmp_path), language="java")
        caller = "web.ProductController:ProductController.byId"
        getproduct_callers = engine.callers_of(
            "repo.ProductRepository:ProductRepository.getProduct"
        )
        save_callers = engine.callers_of("repo.BaseRepository:BaseRepository.save")
        assert [u["id"] for u in getproduct_callers] == [caller]
        assert [u["id"] for u in save_callers] == [caller]

    def test_parser_parse_directory_resolves(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "a/Repo.java": "package a; class Repo { void go() {} }",
                "b/Use.java": "package b; class Use { Repo r; void m() { r.go(); } }",
            },
        )
        graph = JavaParser().parse_directory(str(tmp_path))
        assert _calls_from(graph, "Use.m")["r.go"] == "a.Repo:Repo.go"


class TestAmbiguity:
    def test_does_not_guess_between_unrelated_packages(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "a/Repo.java": "package a; class Repo { void go() {} }",
                "b/Repo.java": "package b; class Repo { void go() {} }",
                "c/Use.java": "package c; class Use { Repo r; void m() { r.go(); } }",
            },
        )
        graph = parse_directory(str(tmp_path), language="java")
        assert _calls_from(graph, "Use.m")["r.go"] == "proxy.unresolved:r.go"

    def test_prefers_type_in_same_module(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "a/Repo.java": "package a; class Repo { void go() {} }",
                "b/Repo.java": (
                    "package b; class Repo { void go() {} }\n"
                    "class Use { Repo r; void m() { r.go(); } }"
                ),
            },
        )
        graph = parse_directory(str(tmp_path), language="java")
        assert _calls_from(graph, "Use.m")["r.go"] == "b.Repo:Repo.go"

    def test_prefers_type_in_same_package(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            {
                "a/Repo.java": "package a; class Repo { void go() {} }",
                "b/Repo.java": "package b; class Repo { void go() {} }",
                "b/Use.java": "package b; class Use { Repo r; void m() { r.go(); } }",
            },
        )
        graph = parse_directory(str(tmp_path), language="java")
        assert _calls_from(graph, "Use.m")["r.go"] == "b.Repo:Repo.go"


class TestRecordsAndNestedTypes:
    @pytest.mark.parametrize(
        ("raw", "target"),
        [
            ("c.id", "Comment:Comment.id"),
            ("c.body", "Comment:Comment.body"),
            ("id.value", "Comment:Comment.Id.value"),
            ("Comment.Id.generate", "Comment:Comment.Id.generate"),
        ],
    )
    def test_resolves_records_accessors_and_qualified_nested_types(
        self, comment_graph: CodeGraph, raw: str, target: str
    ) -> None:
        assert _calls_from(comment_graph, "Use.go")[raw] == target

    def test_resolves_calls_from_nested_types(self, comment_graph: CodeGraph) -> None:
        calls = _calls_from(comment_graph, "Inner.m")
        assert calls["Comment.Id.generate"] == "Comment:Comment.Id.generate"
        assert calls["this.n"] == "Comment:Comment.Inner.n"

    def test_resolves_static_call_from_enum_body_method(self, comment_graph: CodeGraph) -> None:
        assert _calls_from(comment_graph, "Color.code")["Util.f"] == "Comment:Util.f"


class TestReceiverEdgeCases:
    def test_finds_fields_declared_in_enum_bodies(self, tmp_path: Path) -> None:
        path = tmp_path / "Mode.java"
        path.write_text(
            "class Helper { void help() {} }\n"
            "enum Mode { A, B; static Helper H = new Helper(); void run() { H.help(); } }\n"
        )
        graph = parse_file(str(path))
        assert _calls_from(graph, "Mode.run")["H.help"] == "Mode:Helper.help"

    def test_resolves_partially_qualified_nested_type_paths(self, tmp_path: Path) -> None:
        path = tmp_path / "Outer.java"
        path.write_text(
            "class Outer {\n"
            "  static class Inner { static class Deep { static void stat() {} } }\n"
            "  void m() { Inner.Deep.stat(); Outer.Inner.Deep.stat(); }\n"
            "}\n"
        )
        graph = parse_file(str(path))
        calls = _calls_from(graph, "Outer.m")
        assert calls["Inner.Deep.stat"] == "Outer:Outer.Inner.Deep.stat"
        assert calls["Outer.Inner.Deep.stat"] == "Outer:Outer.Inner.Deep.stat"


def _unit(node_id: str, name: str, kind: NodeKind) -> CodeUnit:
    location = SourceLocation(file_path="x.java", start_line=1, end_line=1)
    return CodeUnit(id=node_id, name=name, kind=kind, location=location)


def _call(target: str, receiver_type: str | None) -> CodeEdge:
    attributes = () if receiver_type is None else (("receiver_type", receiver_type),)
    return CodeEdge(
        source_id="app:Main.run",
        target_id=target,
        kind=EdgeKind.CALLS,
        confidence=EdgeConfidence.INFERRED,
        attributes=attributes,
    )


class TestResolveReceiverTypesPass:
    @pytest.fixture
    def graph(self) -> CodeGraph:
        graph = CodeGraph(language="java", root_path=".")
        for node_id, name, kind in [
            ("app:Main", "Main", NodeKind.CLASS),
            ("app:Main.run", "run", NodeKind.METHOD),
            ("lib:Child", "Child", NodeKind.CLASS),
            ("lib:Parent", "Parent", NodeKind.CLASS),
            ("lib:Parent.save", "save", NodeKind.METHOD),
            ("lib:Parent.Box", "Box", NodeKind.CLASS),
            ("lib:Parent.Box.open", "open", NodeKind.METHOD),
            ("lib:Child.field", "field", NodeKind.CLASS),
        ]:
            graph.nodes[node_id] = _unit(node_id, name, kind)
        graph.edges.append(
            CodeEdge(source_id="lib:Child", target_id="lib:Parent", kind=EdgeKind.INHERITS)
        )
        return graph

    def test_follows_supertypes(self, graph: CodeGraph) -> None:
        graph.edges.append(_call("c.save", "Child"))
        resolve_receiver_types(graph)
        assert graph.edges[-1].target_id == "lib:Parent.save"

    def test_indexes_nested_types_by_path(self, graph: CodeGraph) -> None:
        graph.edges.append(_call("b.open", "Parent.Box"))
        resolve_receiver_types(graph)
        assert graph.edges[-1].target_id == "lib:Parent.Box.open"

    def test_only_links_to_methods_and_functions(self, graph: CodeGraph) -> None:
        graph.edges.append(_call("c.field", "Child"))
        resolve_receiver_types(graph)
        assert graph.edges[-1].target_id == "c.field"

    def test_ignores_edges_without_receiver_type(self, graph: CodeGraph) -> None:
        graph.edges.append(_call("c.save", None))
        resolve_receiver_types(graph)
        assert graph.edges[-1].target_id == "c.save"

    def test_leaves_unknown_types_and_methods_alone(self, graph: CodeGraph) -> None:
        graph.edges.extend([_call("x.save", "Missing"), _call("c.missing", "Child")])
        resolve_receiver_types(graph)
        assert [e.target_id for e in graph.edges[-2:]] == ["x.save", "c.missing"]

    def test_keeps_edges_whose_target_exists(self, graph: CodeGraph) -> None:
        edge = _call("lib:Parent.save", "Child")
        graph.edges.append(edge)
        resolve_receiver_types(graph)
        assert graph.edges[-1] is edge
