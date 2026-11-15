"""Регрессии: исходное намерение поиска и ответы агента без полученных сведений."""

import pytest

from assistant.agent.agent import run_agent
from assistant.agent.tools import build_tools
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM


def call(name, **arguments):
    return LLMResponse("", [ToolCall(f"call-{name}", name, arguments)])


def test_shortened_search_keeps_supplier_intent_in_actual_context(kb, erp):
    question = "Кто поставляет пусковое реле для AR-210 с учётом замены?"
    search_llm = FakeLLM(["Поставщик актуальной детали — ХолодКомплект."])
    agent_llm = FakeLLM(
        [
            call("search_knowledge", question="пусковое реле AR-210"),
            "Поставщик актуальной детали — ХолодКомплект.",
        ]
    )

    result = run_agent(question, agent_llm, build_tools(kb, search_llm, erp))

    executed_question = result.steps[0].args["question"]
    assert question in executed_question and "пусковое реле AR-210" in executed_question
    assert "P-1042B -SUPPLIED_BY-> ХолодКомплект" in search_llm.requests[0]["messages"][1]["content"]
    assert "bulletin_SB-2025-03.md" in result.sources
    assert not erp.calls


@pytest.mark.parametrize(
    ("tool_name", "question", "first_query", "followup"),
    [
        (
            "search_knowledge",
            "Кто поставляет замену реле для AR-210?",
            "Реле AR-210",
            "Кто поставляет P-1042B?",
        ),
        (
            "summarize_topic",
            "Какие детали чаще ломаются в линейке Арктика?",
            "Частые отказы",
            "Как часто ломается P-1042B?",
        ),
    ],
)
def test_followup_search_keeps_new_code_and_original_question(tool_name, question, first_query, followup):
    queries = []

    def search(question):
        queries.append(question)
        return {"answer": "Сведения о детали P-1042B.", "sources": ["replacement.md"], "refused": False}

    llm = FakeLLM(
        [
            call(tool_name, question=first_query),
            call(tool_name, question=followup),
            "Сведения о детали P-1042B.",
        ]
    )

    result = run_agent(question, llm, {tool_name: search})

    assert len(queries) == 2
    assert question in queries[0] and first_query in queries[0]
    assert question in queries[1] and followup in queries[1]
    assert result.steps[1].args["question"] == queries[1]
    assert not result.refused


@pytest.mark.parametrize(
    ("question", "tool_name"),
    [
        ("Кто поставляет P-1042B?", "search_knowledge"),
        ("Какая деталь стоит в модели AR-210?", "search_knowledge"),
        ("Чем заменена деталь P-1042 из линейки Арктика?", "search_knowledge"),
        ("Какие узлы чаще всего ломаются?", "summarize_topic"),
        ("Какие детали из руководств линейки Бриз сняты с производства?", "summarize_topic"),
        ("Перечисли заменённые детали линейки Арктика.", "summarize_topic"),
    ],
)
def test_answer_without_search_triggers_one_grounded_lookup(question, tool_name):
    calls = []

    def search(question):
        calls.append(question)
        return {"answer": "Подтверждённые сведения.", "sources": ["manual.md"], "refused": False}

    llm = FakeLLM(["Выдуманные сведения без инструментов.", "Подтверждённые сведения."])

    result = run_agent(question, llm, {tool_name: search})

    assert calls == [question]
    assert len(result.steps) == 1 and result.steps[0].tool == tool_name
    assert result.sources == ["manual.md"] and not result.refused
    assert result.answer == "Подтверждённые сведения."
    assert len(llm.requests) == 2
    assert llm.requests[1]["messages"][-1]["role"] == "tool"
    assert "Подтверждённые сведения" in llm.requests[1]["messages"][-1]["content"]


def test_missing_search_tool_refuses_instead_of_returning_unsupported_prose():
    llm = FakeLLM(["Поставщик — выдуманная компания."])

    result = run_agent("Кто поставляет P-1042B?", llm, {})

    assert result.refused and not result.steps
    assert "выдуманная компания" not in result.answer
    assert len(llm.requests) == 1


def test_failed_fallback_does_not_loop_or_return_unsupported_prose():
    calls = []

    def broken_search(question):
        calls.append(question)
        return {"error": "Индекс недоступен"}

    llm = FakeLLM(["Поставщик — выдуманная компания.", "Поставщик — выдуманная компания."])

    result = run_agent("Кто поставляет P-1042B?", llm, {"search_knowledge": broken_search})

    assert len(calls) == 1 and len(result.steps) == 1
    assert result.steps[0].error == "Индекс недоступен"
    assert result.refused and "выдуманная компания" not in result.answer
    assert len(llm.requests) == 2


def test_fallback_counts_toward_tool_limit():
    llm = FakeLLM(["Поставщик мне известен."])

    result = run_agent(
        "Кто поставляет P-1042B?",
        llm,
        {"search_knowledge": lambda question: {"answer": "Поставщик — ХолодКомплект.", "sources": ["supplier.md"]}},
        max_steps=1,
    )

    assert len(result.steps) == 1 and len(llm.requests) == 1
    assert result.sources == ["supplier.md"]
    assert "лимит" in result.answer and "ХолодКомплект" in result.answer


@pytest.mark.parametrize("failed_tool", ["unknown_search", "get_price"])
def test_unrelated_tool_error_does_not_bypass_required_knowledge_search(failed_tool):
    queries = []

    def search_knowledge(question):
        queries.append(question)
        return {"answer": "Поставщик — ХолодКомплект.", "sources": ["supplier.md"], "refused": False}

    llm = FakeLLM(
        [
            call(failed_tool, part_id="P-1042B"),
            "Поставщик — ВыдуманнаяКомпания.",
            "Поставщик — ХолодКомплект.",
        ]
    )
    tools = {
        "search_knowledge": search_knowledge,
        "get_price": lambda part_id: {"error": "Данные ERP получить не удалось"},
    }

    result = run_agent("Кто поставляет P-1042B?", llm, tools)

    assert queries == ["Кто поставляет P-1042B?"]
    assert [step.tool for step in result.steps] == [failed_tool, "search_knowledge"]
    assert result.steps[0].error and not result.steps[1].error
    assert result.answer == "Поставщик — ХолодКомплект." and not result.refused
    assert result.sources == ["supplier.md"]
    assert len(llm.requests) == 3
    assert "ХолодКомплект" in llm.requests[2]["messages"][-1]["content"]


@pytest.mark.parametrize("outcome", ["missing", "zero", "unavailable"])
def test_agent_distinguishes_missing_erp_data_from_zero_and_unavailable(kb, erp, monkeypatch, outcome):
    part_id = "P-9999" if outcome == "missing" else "P-1042B"
    if outcome == "unavailable":

        def unavailable(*args):
            raise ErpUnavailable("ERP недоступна")

        monkeypatch.setattr(erp, "get_stock", unavailable)
    texts = {
        "missing": "Данные ERP получить не удалось.",
        "zero": "Остаток на складе — 0 шт.",
        "unavailable": "Данные ERP получить не удалось.",
    }
    llm = FakeLLM([call("check_stock", part_id=part_id, warehouse="Москва"), texts[outcome]])
    tools = build_tools(kb, FakeLLM([]), erp)

    result = run_agent(f"Есть ли {part_id} на складе Москва?", llm, tools)

    assert result.refused is (outcome == "missing")
    if outcome == "missing":
        assert result.answer == "Данных в ERP нет: деталь P-9999 не найдена"
    else:
        assert result.answer == texts[outcome]
    assert len(result.steps) == 1 and result.steps[0].tool == "check_stock"
    assert not result.sources


def test_order_text_without_tool_is_corrected_into_pending_action(kb, erp):
    question = "Оформи заказ: 2 штуки для устранения ошибки E-17 на AR-210, склад Казань."
    order = {"part_id": "P-1042B", "warehouse": "Казань", "qty": 2, "model": "AR-210"}
    llm = FakeLLM(
        [
            call("search_knowledge", question="Деталь для E-17 на AR-210"),
            call("check_stock", part_id="P-1042B", warehouse="Казань"),
            call("get_price", part_id="P-1042B"),
            "Заказываю 2 штуки P-1042B. Подтвердите заказ.",
            call("create_service_order", **order),
            "Подтвердите заказ 2 штук P-1042B для AR-210 на склад Казань.",
        ]
    )
    tools = build_tools(kb, FakeLLM(["Нужна актуальная деталь P-1042B."]), erp)

    result = run_agent(question, llm, tools)

    assert result.needs_confirmation and result.pending_action == {"tool": "create_service_order", "args": order}
    assert result.steps[-1].tool == "create_service_order"
    assert erp.calls == [("get_stock", "P-1042B", "Казань"), ("get_price", "P-1042B")]
    assert not result.refused
    assert len(llm.requests) == 6
    assert "create_service_order" in llm.requests[4]["messages"][-1]["content"]


def test_order_that_ignores_tool_requirement_is_refused(kb, erp):
    llm = FakeLLM(["Заказ подготовлен. Подтвердите.", "Заказ создан."])

    result = run_agent(
        "Оформи заказ: 2 штуки P-1042B для AR-210, склад Казань.",
        llm,
        build_tools(kb, FakeLLM([]), erp),
    )

    assert result.refused and not result.needs_confirmation and result.pending_action is None
    assert "Заказ не создан" in result.answer
    assert not result.steps and not erp.calls
    assert len(llm.requests) == 2


@pytest.mark.parametrize(
    "question",
    [
        "Оформи заказ P-1042B для AR-210, склад Казань.",
        "Оформи заказ: 2 штуки P-1042B, склад Казань.",
        "Оформи заказ: 2 штуки P-1042B для AR-210.",
        "Оформи заказ: 2 штуки для AR-210, склад Казань.",
    ],
)
def test_incomplete_order_does_not_force_tool_call(kb, erp, question):
    llm = FakeLLM(["Уточните параметры заказа."])

    result = run_agent(question, llm, build_tools(kb, FakeLLM([]), erp))

    assert result.answer == "Уточните параметры заказа."
    assert not result.needs_confirmation and result.pending_action is None
    assert not result.steps and not erp.calls
    assert len(llm.requests) == 1
