"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation

SYSTEM_PROMPT = """Ты извлекаешь граф знаний из технической документации по холодильному оборудованию.
Верни JSON с полями "entities" и "relations". Извлекай только то, что прямо сказано в тексте.

Типы сущностей и запись имени:
- ProductLine – линейка оборудования (Арктика)
- Model – код модели (AR-210)
- Component – вид узла строчными буквами, в именительном падеже (пусковое реле)
- Part – номер детали (P-1042, P-1042B)
- ErrorCode – код ошибки (E-17)
- Procedure – номер регламента ремонта (SP-104)
- Supplier – поставщик без кавычек и ООО (ХолодКомплект)

Типы связей, направление строго subject -> object:
- BELONGS_TO: Model -> ProductLine – модель входит в линейку
- HAS_PART: Model -> Part – в модели установлена деталь
- IS_A: Part -> Component – деталь является узлом этого вида
- INDICATES: ErrorCode -> Component – ошибка указывает на отказ узла
- FIXED_BY: ErrorCode -> Procedure – ошибка устраняется по регламенту
- REPLACED_BY: Part -> Part – старая деталь снята с производства и заменена новой
- SUPPLIED_BY: Part -> Supplier – деталь поставляет поставщик
- FAILED: Model -> Component – в отчёте о ремонте зафиксирован отказ узла
Другие типы и другие направления запрещены."""

ENTITY_FIELDS = ("type", "name")
RELATION_FIELDS = ("subject_type", "subject", "type", "object_type", "object")

EXAMPLE_TEXT = (
    "# Сервисный бюллетень SB-2025-07\n\nДеталь P-2054 (компрессор) модели BZ-110 снята с производства, "
    "её заменяет P-2054B. Поставщик P-2054B – «КомпрессорМаш»."
)
EXAMPLE_ANSWER = {
    "entities": [
        {"type": "Model", "name": "BZ-110"},
        {"type": "Part", "name": "P-2054"},
        {"type": "Part", "name": "P-2054B"},
        {"type": "Component", "name": "компрессор"},
        {"type": "Supplier", "name": "КомпрессорМаш"},
    ],
    "relations": [
        dict(zip(RELATION_FIELDS, row, strict=True))
        for row in (
            ("Model", "BZ-110", "HAS_PART", "Part", "P-2054"),
            ("Part", "P-2054", "IS_A", "Component", "компрессор"),
            ("Part", "P-2054", "REPLACED_BY", "Part", "P-2054B"),
            ("Part", "P-2054B", "SUPPLIED_BY", "Supplier", "КомпрессорМаш"),
        )
    ],
}
RETRY_PROMPT = "Ответ не соответствует схеме. Верни только JSON-объект с массивами entities и relations."


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Всё, что не подтверждается текстом чанка или словарём, отбрасывается и считается в `dropped`.
    На неразборчивый ответ делается одна повторная попытка, затем возвращается пустой результат.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": EXAMPLE_TEXT},
        {"role": "assistant", "content": json.dumps(EXAMPLE_ANSWER, ensure_ascii=False)},
        {"role": "user", "content": chunk.text},
    ]
    for _ in range(2):
        content = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=2048).content
        parsed = _parse(content)
        if parsed is not None:
            return _ground(chunk, *parsed)
        # Повтор отличается от первого запроса, иначе кеш LLMClient вернёт тот же ответ.
        messages = [*messages, {"role": "assistant", "content": content}, {"role": "user", "content": RETRY_PROMPT}]
    return ExtractionResult(chunk.chunk_id, chunk.doc_id)


def _parse(content: str) -> tuple[list[dict], list[dict]] | None:
    """Разбирает ответ модели. None, если это не JSON нужной структуры."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    entities, relations = data.get("entities"), data.get("relations")
    if not (_items_ok(entities, ENTITY_FIELDS) and _items_ok(relations, RELATION_FIELDS)):
        return None
    return entities, relations


def _items_ok(items, fields: tuple[str, ...]) -> bool:
    """Список словарей, где каждое обязательное поле – строка."""
    return isinstance(items, list) and all(
        isinstance(item, dict) and all(isinstance(item.get(key), str) for key in fields) for item in items
    )


def _ground(chunk: Chunk, raw_entities: list[dict], raw_relations: list[dict]) -> ExtractionResult:
    """Оставляет только сущности и связи, подтверждённые текстом чанка и словарём."""
    codes = set(find_codes(chunk.text))
    names = vocabulary()

    def resolve(entity_type: str, raw_name: str) -> Entity | None:
        name = canonical_name(entity_type, raw_name, names)
        if name is None or (entity_type in CODE_TYPES and name not in codes):
            return None
        return Entity(entity_type, name)

    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    entities: dict[str, Entity] = {}
    for raw in raw_entities:
        entity = resolve(raw["type"], raw["name"])
        if entity is None:
            result.dropped += 1
        else:
            entities.setdefault(entity.node_id, entity)

    relations: dict[tuple[str, str, str], Relation] = {}
    for raw in raw_relations:
        subject = resolve(raw["subject_type"], raw["subject"])
        obj = resolve(raw["object_type"], raw["object"])
        relation = Relation(subject, raw["type"], obj) if subject and obj else None
        if relation is None or not relation.matches_schema():
            result.dropped += 1
            continue
        relations.setdefault((subject.node_id, relation.type, obj.node_id), relation)
        entities.setdefault(subject.node_id, subject)
        entities.setdefault(obj.node_id, obj)

    result.entities = list(entities.values())
    result.relations = list(relations.values())
    return result
