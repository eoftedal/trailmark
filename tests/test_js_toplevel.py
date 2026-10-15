"""Tests for anonymous callbacks and top-level calls in JS/TS modules.

Both the JavaScript and TypeScript parsers share the top-level pass in
``trailmark.parsers._js_toplevel``, so every behavior is checked for
both languages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trailmark.models.edges import EdgeKind
from trailmark.models.graph import CodeGraph
from trailmark.models.nodes import NodeKind
from trailmark.parsers.javascript.parser import JavaScriptParser
from trailmark.parsers.typescript.parser import TypeScriptParser

SERVER_TS = """\
import express, { Request, Response } from 'express'
import { getProductById } from './db'

const app = express()
const router = express.Router()
const cache = makeCache()

app.get('/product/:id', (req: Request, res: Response) => {
  const row = getProductById(req.params.id)
  return res.send(row)
})
router.post('/login', async function (req, res) { audit(req) })
app.use((req, res, next) => next())
cache.get('/not-a-route', (v) => v.trim())
app.listen(3000)
"""

SERVER_JS = SERVER_TS.replace("(req: Request, res: Response)", "(req, res)")

DB = """\
export function getProductById(id) { return query(id) }
function query(id) { return id }
"""

NAMING = """\

module.exports.handler = async (event) => { work(event) }
export default () => boot()
;(function () { init() })()
const routes = { list: () => fetchAll() }
if (flag) { const nested = () => go() }
[1, 2].map((x) => x * 2); [3].map((x) => x)
"""

LANGUAGES = ("javascript", "typescript")


def _parser(language: str) -> JavaScriptParser | TypeScriptParser:
    return JavaScriptParser() if language == "javascript" else TypeScriptParser()


def _ext(language: str) -> str:
    return ".js" if language == "javascript" else ".ts"


def _parse_project(tmp_path: Path, language: str) -> CodeGraph:
    ext = _ext(language)
    server = SERVER_JS if language == "javascript" else SERVER_TS
    (tmp_path / f"server{ext}").write_text(server)
    (tmp_path / f"db{ext}").write_text(DB)
    return _parser(language).parse_directory(str(tmp_path))


def _parse_snippet(tmp_path: Path, language: str, code: str) -> CodeGraph:
    path = tmp_path / f"m{_ext(language)}"
    path.write_text(code)
    return _parser(language).parse_file(str(path))


def _attrs(graph: CodeGraph, node_id: str) -> dict[str, object]:
    return dict(graph.nodes[node_id].attributes)


def _call_targets(graph: CodeGraph, source_id: str) -> list[str]:
    """Call targets from ``source_id``, undoing proxy rewriting."""
    targets: list[str] = []
    for edge in graph.edges:
        if edge.kind == EdgeKind.CALLS and edge.source_id == source_id:
            raw = dict(edge.attributes).get("raw_target")
            targets.append(raw if isinstance(raw, str) else edge.target_id)
    return targets


@pytest.mark.parametrize("language", LANGUAGES)
class TestAnonymousCallbacks:
    def test_route_callback_attributes(self, tmp_path: Path, language: str) -> None:
        graph = _parse_project(tmp_path, language)
        node = graph.nodes["server:app.get@8"]
        assert node.kind == NodeKind.FUNCTION
        assert node.name == "app.get@8"
        assert _attrs(graph, "server:app.get@8") == {
            "anonymous": True,
            "callback_of": "app.get",
            "route": "/product/:id",
            "registered_on_factory": "express",
        }

    def test_callback_parameters_and_location(self, tmp_path: Path, language: str) -> None:
        graph = _parse_project(tmp_path, language)
        node = graph.nodes["server:app.get@8"]
        assert [p.name for p in node.parameters] == ["req", "res"]
        assert node.location.start_line == 8
        assert node.location.end_line == 11

    def test_router_factory_recorded(self, tmp_path: Path, language: str) -> None:
        graph = _parse_project(tmp_path, language)
        attrs = _attrs(graph, "server:router.post@12")
        assert attrs["registered_on_factory"] == "express.Router"
        assert attrs["route"] == "/login"

    def test_middleware_without_route(self, tmp_path: Path, language: str) -> None:
        graph = _parse_project(tmp_path, language)
        attrs = _attrs(graph, "server:app.use@13")
        assert attrs["callback_of"] == "app.use"
        assert "route" not in attrs

    def test_receiver_from_any_factory_call_is_recorded(
        self, tmp_path: Path, language: str
    ) -> None:
        graph = _parse_project(tmp_path, language)
        attrs = _attrs(graph, "server:cache.get@14")
        assert attrs["registered_on_factory"] == "makeCache"

    def test_callback_contained_by_module(self, tmp_path: Path, language: str) -> None:
        graph = _parse_project(tmp_path, language)
        assert any(
            e.kind == EdgeKind.CONTAINS
            and e.source_id == "server"
            and e.target_id == "server:app.get@8"
            for e in graph.edges
        )

    def test_calls_inside_callback_belong_to_callback(self, tmp_path: Path, language: str) -> None:
        graph = _parse_project(tmp_path, language)
        targets = _call_targets(graph, "server:app.get@8")
        assert "db:getProductById" in targets
        assert "res.send" in targets
        assert _call_targets(graph, "server:router.post@12") == ["server:audit"]


@pytest.mark.parametrize("language", LANGUAGES)
class TestModuleLevelCalls:
    def test_top_level_calls_come_from_module(self, tmp_path: Path, language: str) -> None:
        graph = _parse_project(tmp_path, language)
        targets = _call_targets(graph, "server")
        for expected in (
            "server:express",
            "express.Router",
            "server:makeCache",
            "app.get",
            "router.post",
            "app.use",
            "cache.get",
            "app.listen",
        ):
            assert expected in targets

    def test_callback_calls_not_attributed_to_module(self, tmp_path: Path, language: str) -> None:
        graph = _parse_project(tmp_path, language)
        targets = _call_targets(graph, "server")
        assert "server:getProductById" not in targets
        assert "db:getProductById" not in targets
        assert "res.send" not in targets
        assert "next" not in targets and "server:next" not in targets

    def test_no_module_calls_without_top_level_code(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(tmp_path, language, "function f() { g() }\n")
        assert _call_targets(graph, "m") == []


@pytest.mark.parametrize("language", LANGUAGES)
class TestCallbackNaming:
    def test_names_from_context(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(tmp_path, language, NAMING)
        for node_id in (
            "m:module.exports.handler@2",
            "m:default",
            "m:<iife>@4",
            "m:list@5",
            "m:nested@6",
            "m:[1,2].map@7",
            "m:[3].map@7",
        ):
            assert node_id in graph.nodes, node_id
            assert graph.nodes[node_id].kind == NodeKind.FUNCTION

    def test_non_call_contexts_have_no_callback_attributes(
        self, tmp_path: Path, language: str
    ) -> None:
        graph = _parse_snippet(tmp_path, language, NAMING)
        assert _attrs(graph, "m:list@5") == {"anonymous": True}
        assert _attrs(graph, "m:default") == {"anonymous": True}

    def test_expression_bodied_arrow_records_call(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(tmp_path, language, NAMING)
        assert _call_targets(graph, "m:list@5") == ["m:fetchAll"]
        assert _call_targets(graph, "m:default") == ["m:boot"]

    def test_same_label_on_one_line_is_deduplicated(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(tmp_path, language, "on('a', () => a()); on('b', () => b())\n")
        assert _attrs(graph, "m:on@1")["route"] == "a"
        assert _attrs(graph, "m:on@1#2")["route"] == "b"
        assert graph.nodes["m:on@1#2"].name == "on@1#2"

    def test_nested_callbacks_belong_to_outer(self, tmp_path: Path, language: str) -> None:
        code = "describe('x', () => {\n  it('y', () => check())\n})\n"
        graph = _parse_snippet(tmp_path, language, code)
        functions = [n.id for n in graph.nodes.values() if n.kind == NodeKind.FUNCTION]
        assert functions == ["m:describe@1"]
        assert _call_targets(graph, "m:describe@1") == ["m:it", "m:check"]
        assert _call_targets(graph, "m") == ["m:describe"]


@pytest.mark.parametrize("language", LANGUAGES)
class TestObjectLiteralsAndDefaults:
    def test_object_literal_methods_extracted(self, tmp_path: Path, language: str) -> None:
        code = (
            "const repo = {\n"
            "  get(id) { return fetchOne(id) },\n"
            "  'quoted key': () => other(),\n"
            "  [computed + 1]: () => c(),\n"
            "}\n"
        )
        graph = _parse_snippet(tmp_path, language, code)
        assert _call_targets(graph, "m:get@2") == ["m:fetchOne"]
        assert [p.name for p in graph.nodes["m:get@2"].parameters] == ["id"]
        assert _attrs(graph, "m:get@2") == {"anonymous": True}
        assert _call_targets(graph, "m") == []

    def test_object_keys_labelled(self, tmp_path: Path, language: str) -> None:
        code = "const o = {\n  'quoted key': () => q(),\n  [computed + 1]: () => c(),\n}\n"
        graph = _parse_snippet(tmp_path, language, code)
        assert _call_targets(graph, "m:quoted key@2") == ["m:q"]
        assert _call_targets(graph, "m:[computed+1]@3") == ["m:c"]

    @pytest.mark.parametrize(
        "code",
        [
            "export default async (req, res) => { handle(req) }\n",
            "export default function () { handle() }\n",
            "export default async function* () { handle() }\n",
        ],
    )
    def test_anonymous_default_export_named_default(
        self, tmp_path: Path, language: str, code: str
    ) -> None:
        graph = _parse_snippet(tmp_path, language, code)
        assert graph.nodes["m:default"].name == "default"
        assert _call_targets(graph, "m:default") == ["m:handle"]

    def test_repeated_default_export_deduplicated(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(
            tmp_path,
            language,
            "export default () => a()\nexport default () => b()\n",
        )
        assert _call_targets(graph, "m:default") == ["m:a"]
        assert _call_targets(graph, "m:default#2") == ["m:b"]


@pytest.mark.parametrize("language", LANGUAGES)
class TestCommonJsFactories:
    def test_require_factories_normalized(self, tmp_path: Path, language: str) -> None:
        code = (
            "const app = require('express')()\n"
            "const router = require('express').Router()\n"
            "const fastify = require('fastify')({ logger: true })\n"
            "app.get('/a', (req, res) => res.end())\n"
            "router.post('/b', (req, res) => res.end())\n"
            "fastify.put('/c', async (request) => request.body)\n"
            "const other = require('x' + y)()\n"
            "other.get('/d', () => 1)\n"
        )
        graph = _parse_snippet(tmp_path, language, code)
        assert _attrs(graph, "m:app.get@4")["registered_on_factory"] == "express"
        assert _attrs(graph, "m:router.post@5")["registered_on_factory"] == "express.Router"
        assert _attrs(graph, "m:fastify.put@6")["registered_on_factory"] == "fastify"
        assert _attrs(graph, "m:other.get@8")["registered_on_factory"] == "require"


@pytest.mark.parametrize("language", LANGUAGES)
class TestFunctionsAlreadyExtracted:
    def test_named_functions_untouched(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(
            tmp_path,
            language,
            "export const handler = (e) => run(e)\nfunction named() {}\nlater = function () {}\n",
        )
        assert list(graph.nodes) == ["m", "m:handler", "m:named", "m:later"]
        assert graph.nodes["m:handler"].attributes == ()
        assert _call_targets(graph, "m") == []

    def test_nested_declarator_named_by_line(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(tmp_path, language, "{ const inner = () => go() }\n")
        assert "m:inner" not in graph.nodes
        assert _attrs(graph, "m:inner@1") == {"anonymous": True}
        assert _call_targets(graph, "m:inner@1") == ["m:go"]

    def test_top_level_generator_expressions_extracted(self, tmp_path: Path, language: str) -> None:
        # The regular visitor does not extract generator expressions, so
        # the top-level pass must not treat them as already extracted.
        graph = _parse_snippet(
            tmp_path,
            language,
            "const gen = function* () { yield step() }\nlater = function* () { yield l() }\n",
        )
        assert "m:gen" not in graph.nodes
        assert _call_targets(graph, "m:gen@1") == ["m:step"]
        assert _call_targets(graph, "m:later@2") == ["m:l"]

    def test_class_bodies_skipped(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(
            tmp_path,
            language,
            "class A { m() { return [1].map((x) => f(x)) } }\n",
        )
        assert list(graph.nodes) == ["m", "m:A", "m:A.m"]
        assert _call_targets(graph, "m") == []

    def test_class_expressions_skipped(self, tmp_path: Path, language: str) -> None:
        graph = _parse_snippet(
            tmp_path,
            language,
            "const B = class { m() { return [1].map((x) => f(x)) } }\n",
        )
        assert list(graph.nodes) == ["m"]
        assert _call_targets(graph, "m") == []
