"""Проверки выдуманных кодов и потери области ответа при MAP/REDUCE."""

import json

import pytest

from assistant.graphrag.search import REFUSAL, global_search, local_search
from assistant.infra.schemas import Community
from assistant.infra.testing import FakeLLM


@pytest.mark.parametrize("unsupported", ["P-9998", "P-1142"])
def test_local_corrects_a_code_absent_from_selected_facts(kb, unsupported):
    # P-1142 есть в графе, но не относится к найденной цепочке AR-210.
    llm = FakeLLM([f"Закажите {unsupported}.", "Закажите P-1042B."])

    result = local_search("Что заказать при E-17 на AR-210?", kb, llm)

    assert result.text == "Закажите P-1042B." and not result.refused
    assert len(llm.requests) == 2
    assert unsupported in llm.requests[1]["messages"][-1]["content"]
    assert "P-1042 -REPLACED_BY-> P-1042B" in result.facts
    assert "bulletin_SB-2025-03.md" in result.sources


def test_local_refuses_after_one_unsuccessful_correction(kb):
    llm = FakeLLM(["Закажите P-9998.", "Тогда закажите P-9997."])

    result = local_search("Что заказать при E-17 на AR-210?", kb, llm)

    assert result.text == REFUSAL and result.refused
    assert len(llm.requests) == 2
    assert result.facts and result.sources


def test_single_part_does_not_expand_another_parts_chain_via_supplier(kb):
    graph = kb.graph
    for kind, name in (("Part", "P-9998"), ("Part", "P-9998B"), ("Supplier", "ДругойПоставщик")):
        graph.add_node(f"{kind}:{name}", type=kind, name=name, sources=[])
    graph.add_edge("Part:P-9998", "Supplier:ХолодКомплект", type="SUPPLIED_BY", sources=["shared.md#0"])
    graph.add_edge("Part:P-9998", "Part:P-9998B", type="REPLACED_BY", sources=["foreign.md#0"])
    graph.add_edge("Part:P-9998B", "Supplier:ДругойПоставщик", type="SUPPLIED_BY", sources=["foreign.md#0"])

    result = local_search("Кто поставляет P-1042B?", kb, FakeLLM(["ХолодКомплект."]))

    # Исходная окрестность глубины 2 остаётся; чужая цепочка за ней не раскрывается.
    assert "P-9998 -SUPPLIED_BY-> ХолодКомплект" in result.facts
    assert all("P-9998B" not in fact for fact in result.facts)
    assert "foreign.md" not in result.sources


def test_global_does_not_recommend_parts_for_an_unknown_model(kb):
    llm = FakeLLM([])

    result = global_search("Какую деталь заказать при E-17 на AR-999?", kb, llm)

    assert result.refused and result.text == REFUSAL
    assert not result.sources
    assert not llm.requests


def test_global_ignores_high_scoring_answer_from_another_product_line(kb):
    graph = kb.graph
    for kind, name in (("Model", "BZ-110"), ("ProductLine", "Бриз"), ("Component", "компрессор")):
        graph.add_node(f"{kind}:{name}", type=kind, name=name, sources=["breeze.md#0"])
    graph.add_edge("Model:BZ-110", "ProductLine:Бриз", type="BELONGS_TO", sources=["breeze.md#0"])
    graph.add_edge("Model:BZ-110", "Component:компрессор", type="FAILED", weight=99, sources=["breeze.md#0"])
    kb.communities = [
        Community(0, ["Model:AR-210"], "Арктика: пусковое реле, FAILED=37."),
        Community(1, ["Model:BZ-110"], "Бриз: компрессор, FAILED=99."),
    ]
    llm = FakeLLM(
        [
            json.dumps({"answer": "Арктика: пусковое реле, 37 отказов.", "score": 8}),
            json.dumps({"answer": "Компрессор, 99 отказов.", "score": 10}),
            "В Арктике чаще отказывает пусковое реле.",
        ]
    )

    result = global_search("Какой узел чаще отказывает в линейке Арктика?", kb, llm)

    assert not result.refused
    assert len(llm.requests) == 3  # MAP вызывается для каждой сводки по контракту.
    reduce_context = llm.requests[2]["messages"][1]["content"]
    assert "Арктика: пусковое реле, FAILED=37." in reduce_context
    assert "Компрессор, 99" not in reduce_context
    assert "Бриз" not in reduce_context
    assert "breeze.md" not in result.sources
    # Эти сведения находятся за границей сообщества модели, но использованы в сводке.
    assert "report_R-2025-001.md" in result.sources
    assert "bulletin_SB-2025-03.md" in result.sources
