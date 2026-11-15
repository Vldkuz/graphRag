"""Линейка должна подтверждаться исходным текстом, а не только словарём."""

import json

from assistant.graphrag.extraction import extract_chunk
from assistant.infra.schemas import Chunk
from assistant.infra.testing import FakeLLM


def _answer(product_line: str) -> str:
    return json.dumps(
        {
            "entities": [
                {"type": "Model", "name": "BZ-120"},
                {"type": "ProductLine", "name": product_line},
            ],
            "relations": [
                {
                    "subject_type": "Model",
                    "subject": "BZ-120",
                    "type": "BELONGS_TO",
                    "object_type": "ProductLine",
                    "object": product_line,
                }
            ],
        },
        ensure_ascii=False,
    )


def test_product_line_from_vocabulary_is_dropped_when_absent_from_report():
    chunk = Chunk(
        "report.md#0",
        "report.md",
        "Оборудование: моноблок BZ-120. Диагностика показала отказ узла: вентилятор испарителя.",
    )

    result = extract_chunk(chunk, FakeLLM([_answer("Арктика")]))

    assert {entity.node_id for entity in result.entities} == {"Model:BZ-120"}
    assert result.relations == []
    assert result.dropped == 2  # Неподтверждённая линейка и связь модели с ней.


def test_product_line_in_chunk_keeps_canonical_name_despite_case_and_quotes():
    chunk = Chunk("manual.md#0", "manual.md", "Моноблок BZ-120 входит в линейку «бРИЗ».")

    result = extract_chunk(chunk, FakeLLM([_answer("линейка «БРИЗ»")]))

    assert {entity.node_id for entity in result.entities} == {"Model:BZ-120", "ProductLine:Бриз"}
    assert [
        (relation.subject.node_id, relation.type, relation.object.node_id)
        for relation in result.relations
    ] == [("Model:BZ-120", "BELONGS_TO", "ProductLine:Бриз")]
    assert result.dropped == 0
