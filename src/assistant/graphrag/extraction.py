"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes, find_names
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
Другие типы и другие направления запрещены.

Правила:
- В руководстве модели для каждой детали из состава узлов дай HAS_PART от этой модели, IS_A и SUPPLIED_BY.
- REPLACED_BY соединяет два разных кода: subject – снятая с производства деталь, object – деталь, которая её заменяет.
- FIXED_BY всегда идёт от ошибки к регламенту, даже если текст начинается с регламента.
- INDICATES всегда идёт от ошибки к узлу.
Коды в примерах выдуманы, бери коды только из текста."""

ENTITY_FIELDS = ("type", "name")
RELATION_FIELDS = ("subject_type", "subject", "type", "object_type", "object")


def _answer(entities: list[tuple[str, str]], relations: list[tuple[str, str, str, str, str]]) -> str:
    return json.dumps(
        {
            "entities": [dict(zip(ENTITY_FIELDS, row, strict=True)) for row in entities],
            "relations": [dict(zip(RELATION_FIELDS, row, strict=True)) for row in relations],
        },
        ensure_ascii=False,
    )



EXAMPLES = [
    (
        "# Руководство по эксплуатации: шкаф AR-900\n\nШкаф AR-900 входит в линейку «Арктика».\n\n"
        "## Состав узлов\n\nВ AR-900 установлены следующие узлы.\n\n"
        "- компрессор – деталь P-9010, поставщик «КомпрессорМаш»\n"
        "- контроллер – деталь P-9014, поставщик «АвтоматикаСевер»\n\n"
        "## Коды ошибок\n\n- E-90 – компрессор не запускается, отказ узла «компрессор»",
        _answer(
            [
                ("Model", "AR-900"),
                ("ProductLine", "Арктика"),
                ("Part", "P-9010"),
                ("Part", "P-9014"),
                ("Component", "компрессор"),
                ("Component", "контроллер"),
                ("Supplier", "КомпрессорМаш"),
                ("Supplier", "АвтоматикаСевер"),
                ("ErrorCode", "E-90"),
            ],
            [
                ("Model", "AR-900", "BELONGS_TO", "ProductLine", "Арктика"),
                ("Model", "AR-900", "HAS_PART", "Part", "P-9010"),
                ("Part", "P-9010", "IS_A", "Component", "компрессор"),
                ("Part", "P-9010", "SUPPLIED_BY", "Supplier", "КомпрессорМаш"),
                ("Model", "AR-900", "HAS_PART", "Part", "P-9014"),
                ("Part", "P-9014", "IS_A", "Component", "контроллер"),
                ("Part", "P-9014", "SUPPLIED_BY", "Supplier", "АвтоматикаСевер"),
                ("ErrorCode", "E-90", "INDICATES", "Component", "компрессор"),
            ],
        ),
    ),
    (
        "# Сервисный бюллетень SB-2025-90\n\nДеталь P-9010 (компрессор) снята с производства. "
        "Вместо неё устанавливается деталь P-9010B.\n\nПоставщик детали P-9010B – «КомпрессорМаш».\n\n"
        "# Регламент SP-190: Замена компрессора\n\nРегламент SP-190 применяется при ошибке E-90.",
        _answer(
            [
                ("Part", "P-9010"),
                ("Part", "P-9010B"),
                ("Component", "компрессор"),
                ("Supplier", "КомпрессорМаш"),
                ("Procedure", "SP-190"),
                ("ErrorCode", "E-90"),
            ],
            [
                ("Part", "P-9010", "IS_A", "Component", "компрессор"),
                ("Part", "P-9010", "REPLACED_BY", "Part", "P-9010B"),
                ("Part", "P-9010B", "SUPPLIED_BY", "Supplier", "КомпрессорМаш"),
                ("ErrorCode", "E-90", "FIXED_BY", "Procedure", "SP-190"),
            ],
        ),
    ),
]
RETRY_PROMPT = "Ответ не соответствует схеме. Верни только JSON-объект с массивами entities и relations."



def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Всё, что не подтверждается текстом чанка или словарём, отбрасывается и считается в `dropped`.
    На неразборчивый ответ делается одна повторная попытка, затем возвращается пустой результат.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for text, answer in EXAMPLES:
        messages += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": chunk.text})
    for _ in range(2):
        content = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=4096).content
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
    mentioned_names = set(find_names(chunk.text, names))

    def resolve(entity_type: str, raw_name: str) -> Entity | None:
        name = canonical_name(entity_type, raw_name, names)
        if name is None or (entity_type in CODE_TYPES and name not in codes):
            return None
        # Название линейки нельзя угадывать по коду модели.
        if entity_type == "ProductLine" and (entity_type, name) not in mentioned_names:
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
        relation = _oriented(subject, raw["type"], obj) if subject and obj else None
        if relation is None:
            result.dropped += 1
            continue
        relations.setdefault((relation.subject.node_id, relation.type, relation.object.node_id), relation)
        entities.setdefault(subject.node_id, subject)
        entities.setdefault(obj.node_id, obj)

    result.entities = list(entities.values())
    result.relations = list(relations.values())
    return result


def _oriented(subject: Entity, kind: str, obj: Entity) -> Relation | None:
    """Связь по схеме. Перепутанное моделью направление разворачивается, петли отбрасываются."""
    if subject.node_id == obj.node_id:
        return None
    for relation in (Relation(subject, kind, obj), Relation(obj, kind, subject)):
        if relation.matches_schema():
            return relation
    return None
