"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable
from functools import wraps
from inspect import signature

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART = {"type": "string", "pattern": r"^P-[0-9]{4}[A-Z]?$", "description": "Артикул детали, например P-1042B"}
WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
QUESTION = {"type": "string", "minLength": 1, "description": "Полный вопрос с известными кодами и названиями"}


def _schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge", "Поиск фактов о модели, ошибке, детали и её актуальной замене.", {"question": QUESTION}
    ),
    _schema(
        "summarize_topic", "Обзорный ответ по сводкам: типичные поломки и общие закономерности.", {"question": QUESTION}
    ),
    _schema(
        "check_stock",
        "Остаток детали на складе и признак снятия с производства.",
        {"part_id": PART, "warehouse": WAREHOUSE},
    ),
    _schema("get_price", "Цена детали в рублях и признак снятия с производства.", {"part_id": PART}),
    _schema(
        "create_service_order",
        "Создать заказ детали. Агент должен предварительно получить подтверждение человека.",
        {
            "part_id": PART,
            "warehouse": WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1, "description": "Количество, указанное пользователем"},
            "model": {"type": "string", "pattern": r"^(AR|BZ|GS)-[0-9]{3}$"},
        },
    ),
]


def _valid_argument(value: object, spec: dict) -> bool:
    if spec["type"] == "integer":
        return type(value) is int and value >= spec.get("minimum", 0)
    if not isinstance(value, str) or not value.strip():
        return False
    if "enum" in spec and value not in spec["enum"]:
        return False
    return "pattern" not in spec or re.fullmatch(spec["pattern"], value) is not None


def _validated(function: Callable[..., dict], properties: dict) -> Callable[..., dict]:
    parameters = signature(function)

    @wraps(function)
    def invoke(*args, **kwargs) -> dict:
        try:
            bound = parameters.bind(*args, **kwargs)
        except TypeError as exc:
            return {"error": f"Некорректные аргументы: {exc}"}
        for name, value in bound.arguments.items():
            if not _valid_argument(value, properties[name]):
                return {"error": f"Некорректный аргумент {name}: {value!r}"}
        try:
            return function(*args, **kwargs)
        except ErpNotFound as exc:
            return {"error": f"Данных в ERP нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}
        # ErpUnavailable не перехватываем: повторными попытками управляет агент.

    return invoke


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Создаёт пять инструментов с проверкой аргументов до обращения к поиску или ERP."""

    def search_knowledge(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        return erp.get_stock(part_id, warehouse)

    def get_price(part_id: str) -> dict:
        return erp.get_price(part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        return erp.create_order(part_id, warehouse, qty, model)

    functions = {
        function.__name__: function
        for function in (search_knowledge, summarize_topic, check_stock, get_price, create_service_order)
    }
    return {
        schema["function"]["name"]: _validated(
            functions[schema["function"]["name"]], schema["function"]["parameters"]["properties"]
        )
        for schema in TOOL_SCHEMAS
    }
