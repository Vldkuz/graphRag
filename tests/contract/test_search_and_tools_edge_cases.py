"""Дополнительные проверки поиска и инструментов без настоящей модели и ERP."""

import json

import networkx as nx
import pytest

from assistant.agent.tools import build_tools
from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.schemas import Community, KnowledgeBase
from assistant.infra.testing import FakeLLM


def test_replacement_chain_beyond_path_limit_and_cycle(kb):
    graph = kb.graph
    for name in ("P-1042C", "P-1042D"):
        graph.add_node(f"Part:{name}", type="Part", name=name, sources=[])
    for old, new in (("P-1042B", "P-1042C"), ("P-1042C", "P-1042D"), ("P-1042D", "P-1042B")):
        graph.add_edge(f"Part:{old}", f"Part:{new}", type="REPLACED_BY", sources=["replacement.md#0"])
    result = local_search("E-17 на AR-210", kb, FakeLLM(["Обнаружена цепочка замен."]))
    assert any("P-1042C -REPLACED_BY-> P-1042D" in fact for fact in result.facts)
    assert "replacement.md" in result.sources


@pytest.fixture
def kb_with_supplier_chain(kb):
    graph = kb.graph
    for node_type, name in (
        ("Part", "P-1042C"),
        ("Supplier", "НовыйПоставщик"),
        ("Part", "P-9998"),
        ("Model", "AR-999"),
    ):
        graph.add_node(f"{node_type}:{name}", type=node_type, name=name, sources=[])
    graph.add_edge("Part:P-1042B", "Part:P-1042C", type="REPLACED_BY", sources=["replacement_C.md#0"])
    graph.add_edge(
        "Part:P-1042C", "Supplier:НовыйПоставщик", type="SUPPLIED_BY", sources=["supplier_C.md#0"]
    )
    # Общий поставщик не делает чужую деталь и её модель подходящими к вопросу.
    graph.add_edge(
        "Part:P-9998", "Supplier:НовыйПоставщик", type="SUPPLIED_BY", sources=["unrelated.md#0"]
    )
    graph.add_edge("Model:AR-999", "Part:P-9998", type="HAS_PART", sources=["unrelated.md#0"])
    return kb


@pytest.mark.parametrize(
    "question",
    [
        "Кто поставляет пусковое реле для AR-210 с учётом замен?",
        "У какого поставщика заказать деталь при E-17 на AR-210?",
        "Кто поставляет замену для детали P-1042?",
    ],
)
def test_supplier_context_follows_replacements_without_crossing_shared_supplier(kb_with_supplier_chain, question):
    llm = FakeLLM(["Найден поставщик актуальной детали."])

    result = local_search(question, kb_with_supplier_chain, llm)

    prompt = llm.requests[0]["messages"][1]["content"]
    for fact in (
        "P-1042B -REPLACED_BY-> P-1042C",
        "P-1042B -SUPPLIED_BY-> ХолодКомплект",
        "P-1042C -SUPPLIED_BY-> НовыйПоставщик",
    ):
        assert fact in result.facts
        assert fact in prompt
    assert {"replacement_C.md", "supplier_C.md"} <= set(result.sources)
    assert "unrelated.md" not in result.sources
    for unrelated in ("P-9998", "AR-999"):
        assert all(unrelated not in fact for fact in result.facts)
        assert unrelated not in prompt


@pytest.mark.parametrize("action", ["заказать", "поставить"])
def test_non_supplier_question_does_not_add_suppliers_outside_paths(kb_with_supplier_chain, action):
    llm = FakeLLM(["Для замены нужна P-1042C."])

    result = local_search(f"Какую деталь {action} при E-17 на AR-210?", kb_with_supplier_chain, llm)

    prompt = llm.requests[0]["messages"][1]["content"]
    assert "P-1042B -REPLACED_BY-> P-1042C" in result.facts
    assert "P-1042B -REPLACED_BY-> P-1042C" in prompt
    assert all("SUPPLIED_BY" not in fact for fact in result.facts)
    assert "SUPPLIED_BY" not in prompt
    assert "replacement_C.md" in result.sources
    assert "supplier_C.md" not in result.sources
    assert "unrelated.md" not in result.sources


def test_disconnected_anchors_refuse_without_model_call(kb):
    kb.graph.add_node("Model:GS-999", type="Model", name="GS-999", sources=[])
    llm = FakeLLM([])
    assert local_search("E-17 на GS-999", kb, llm).refused
    assert not llm.requests


def test_context_limit_preserves_whole_facts_and_only_their_sources():
    graph = nx.DiGraph()
    graph.add_node("Part:P-1042", type="Part", name="P-1042", sources=[])
    for index in range(200):
        name = f"Поставщик {index:03d} " + "длинное название " * 8
        node = f"Supplier:{name}"
        graph.add_node(node, type="Supplier", name=name, sources=[])
        graph.add_edge("Part:P-1042", node, type="SUPPLIED_BY", sources=[f"supplier_{index:03d}.md#0"])
    kb = KnowledgeBase(graph, [], {})
    llm = FakeLLM(["Сведения о поставщиках."])
    result = local_search("P-1042", kb, llm)
    context = llm.requests[0]["messages"][1]["content"].split("Факты:\n", 1)[1]
    assert 0 < len(context) <= 8000
    assert len(result.facts) < 200
    assert context == "\n".join(f"{i}. {fact}" for i, fact in enumerate(result.facts, 1))
    assert result.sources == [f"supplier_{i:03d}.md" for i in range(len(result.facts))]


@pytest.mark.parametrize(
    "partial",
    [
        "[]",
        "null",
        '{"answer": 1, "score": 5}',
        '{"answer": "да", "score": true}',
        '{"answer": "да", "score": 11}',
        '{"answer": "да", "score": "5"}',
        '{"answer": " ", "score": 5}',
    ],
)
def test_map_results_with_invalid_schema_are_skipped(partial):
    graph = nx.DiGraph()
    graph.add_node("Part:P-1000", sources=["invalid.md#0"])
    graph.add_node("Part:P-2000", sources=["useful.md#0"])
    kb = KnowledgeBase(
        graph,
        [Community(0, ["Part:P-1000"], "Первая сводка"), Community(1, ["Part:P-2000"], "Вторая сводка")],
        {},
    )
    useful_answer = "Чаще всего отказывает пусковое реле."
    final_answer = "По отчётам чаще всего отказывает пусковое реле."
    llm = FakeLLM([partial, json.dumps({"answer": useful_answer, "score": 8}), final_answer])

    result = global_search("Какой узел чаще всего отказывает?", kb, llm)

    assert not result.refused
    assert result.text == final_answer
    assert result.sources == ["useful.md"]
    assert len(llm.requests) == 3
    merge_context = llm.requests[2]["messages"][1]["content"].split("Частичные ответы:\n", 1)[1]
    assert useful_answer in merge_context
    assert "Вторая сводка" in merge_context
    assert "Первая сводка" not in merge_context


def test_global_sources_only_come_from_useful_communities():
    graph = nx.DiGraph()
    graph.add_node("Part:P-1000", sources=["useful.md#1", "useful.md#2"])
    graph.add_node("Part:P-2000", sources=["irrelevant.md#0"])
    kb = KnowledgeBase(graph, [Community(0, ["Part:P-1000"], "Первая"), Community(1, ["Part:P-2000"], "Вторая")], {})
    llm = FakeLLM([json.dumps({"answer": "Факт", "score": 5}), json.dumps({"answer": "Мусор", "score": 0}), "Факт"])
    assert global_search("Вопрос", kb, llm).sources == ["useful.md"]


@pytest.mark.parametrize(
    "args", [{}, {"part_id": None}, {"part_id": []}, {"part_id": "P-1042B", "extra": 1}]
)
def test_invalid_price_arguments_never_reach_erp(kb, erp, args):
    assert "error" in build_tools(kb, FakeLLM([]), erp)["get_price"](**args)
    assert not erp.calls


@pytest.mark.parametrize("qty", [True, 1.5, "2", None])
def test_wrong_quantity_type_is_rejected_before_erp_call(kb, erp, qty):
    result = build_tools(kb, FakeLLM([]), erp)["create_service_order"](
        part_id="P-1042B", warehouse="Казань", qty=qty, model="AR-210"
    )
    assert "error" in result and not erp.calls


# Инструмент делает одну попытку и передаёт ErpUnavailable агенту, который отвечает за повторы.
def test_unavailable_erp_propagates_without_retry(kb, erp, monkeypatch):
    calls = []

    def unavailable(part_id):
        calls.append(part_id)
        raise ErpUnavailable("нет связи")

    monkeypatch.setattr(erp, "get_price", unavailable)
    with pytest.raises(ErpUnavailable):
        build_tools(kb, FakeLLM([]), erp)["get_price"](part_id="P-1042B")
    assert calls == ["P-1042B"]


def test_summarize_topic_returns_global_answer(kb, erp):
    llm = FakeLLM(['{"answer": "Реле", "score": 8}', '{"answer": "нет", "score": 0}', "Реле"])
    result = build_tools(kb, llm, erp)["summarize_topic"](question="Что ломается?")
    assert result["answer"] == "Реле" and not result["refused"] and result["sources"]
    assert not erp.calls
