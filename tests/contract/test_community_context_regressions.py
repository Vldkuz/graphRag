"""Регрессии: статистика и замены не должны теряться на границе сообществ."""

from collections import Counter

import networkx as nx

from assistant.graphrag.communities import summarize_community
from assistant.graphrag.community_context import community_context_edges
from assistant.infra.schemas import Community
from assistant.infra.testing import FakeLLM


def add_fact(graph, source, relation, target, weight=1, document="manual.md"):
    for node in (source, target):
        kind, name = node.split(":", 1)
        graph.add_node(node, type=kind, name=name, sources=[f"{document}#0"])
    graph.add_edge(source, target, type=relation, weight=weight, sources=[f"{document}#0"])


def test_boundary_failure_changes_model_ranking_and_is_owned_once():
    graph = nx.DiGraph()
    for model in ("Model:BZ-110", "Model:BZ-120"):
        add_fact(graph, model, "BELONGS_TO", "ProductLine:Бриз")
        add_fact(graph, model, "FAILED", "Component:вентилятор испарителя", 2)
        add_fact(graph, model, "FAILED", "Component:нагреватель оттайки", 1)
    add_fact(graph, "Model:BZ-120", "FAILED", "Component:компрессор", 1, "report_extra.md")
    owners = Community(0, ["Model:BZ-110", "Model:BZ-120", "Component:нагреватель оттайки"])
    neighbours = Community(1, [node for node in graph if node not in owners.nodes])
    llm = FakeLLM(["Компоненты связаны с моделями."])

    summary = summarize_community(owners, graph, llm)

    assert "Модель BZ-110 (Бриз): FAILED всего=3" in summary
    assert "Модель BZ-120 (Бриз): FAILED всего=4" in summary
    assert "Итого FAILED Бриз только по моделям BZ-110,BZ-120: вентилятор испарителя=4; компрессор=1" in summary
    prompt = llm.requests[0]["messages"][1]["content"]
    assert "Модель BZ-120 (Бриз): FAILED всего=4" in prompt
    # Граница не выкидывает событие и не создаёт вторую копию в сообществе компонента.
    selected = Counter(
        edge
        for community in (owners, neighbours)
        for edge in community_context_edges(community, graph)
        if graph.edges[edge]["type"] == "FAILED"
    )
    expected = {edge for edge in graph.edges if graph.edges[edge]["type"] == "FAILED"}
    assert set(selected) == expected
    assert all(count == 1 for count in selected.values())
    assert ("Model:BZ-120", "Component:компрессор") in community_context_edges(owners, graph)


def test_installed_replacements_keep_model_and_line_across_communities():
    graph = nx.DiGraph()
    add_fact(graph, "Model:BZ-120", "BELONGS_TO", "ProductLine:Бриз")
    add_fact(graph, "Model:BZ-120", "HAS_PART", "Part:P-2054")
    add_fact(graph, "Part:P-2054", "REPLACED_BY", "Part:P-2054B", document="bulletin_B.md")
    add_fact(graph, "Part:P-2054B", "REPLACED_BY", "Part:P-2054C", document="bulletin_C.md")
    add_fact(graph, "Model:BZ-210", "BELONGS_TO", "ProductLine:Бриз")
    add_fact(graph, "Model:BZ-210", "HAS_PART", "Part:P-2110")
    add_fact(graph, "Part:P-2110", "REPLACED_BY", "Part:P-2110B", document="bulletin_compressor.md")
    # Эта модель использует ту же деталь, но принадлежит другому сообществу.
    add_fact(graph, "Model:AR-210", "HAS_PART", "Part:P-2054")
    community = Community(0, ["Model:BZ-120", "Model:BZ-210"])

    summary = summarize_community(community, graph, FakeLLM(["Замены указаны в бюллетенях."]))
    context = community_context_edges(community, graph)

    assert (
        "Снята с производства: P-2054, модели BZ-120 (Бриз); замены (снятая→новая): P-2054→P-2054B; P-2054B→P-2054C."
    ) in summary
    assert ("Снята с производства: P-2110, модели BZ-210 (Бриз); замены (снятая→новая): P-2110→P-2110B.") in summary
    assert ("Part:P-2054B", "Part:P-2054C") in context
    assert graph.edges["Part:P-2054B", "Part:P-2054C"]["sources"] == ["bulletin_C.md#0"]
    assert all("Model:AR-210" not in edge for edge in context)
    assert "AR-210" not in summary


def test_statistics_and_replacements_survive_high_weight_fact_limit_and_llm_omission():
    graph = nx.DiGraph()
    add_fact(graph, "Model:BZ-120", "BELONGS_TO", "ProductLine:Бриз")
    add_fact(graph, "Model:BZ-120", "FAILED", "Component:компрессор", 1)
    add_fact(graph, "Model:BZ-120", "HAS_PART", "Part:P-2110")
    add_fact(graph, "Part:P-2110", "REPLACED_BY", "Part:P-2110B")
    for index in range(45):
        add_fact(graph, f"Part:P-{3000 + index}", "IS_A", "Component:компрессор", 100)
    community = Community(0, list(graph))
    llm = FakeLLM([" ".join(["прочее"] * 400)])

    summary = summarize_community(community, graph, llm)

    assert "Модель BZ-120 (Бриз): FAILED всего=1; компрессор=1." in summary
    assert ("Снята с производства: P-2110, модели BZ-120 (Бриз); замены (снятая→новая): P-2110→P-2110B.") in summary
    assert len(summary.split()) <= 150
    assert len(llm.requests) == 1
    assert "100" not in summary  # популярность IS_A не становится статистикой отказов
