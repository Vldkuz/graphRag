"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

from collections import defaultdict

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    nodes: dict[str, Entity] = {}
    node_sources: dict[str, set[str]] = defaultdict(set)
    edge_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    def add_node(entity: Entity, chunk_id: str) -> None:
        nodes.setdefault(entity.node_id, entity)
        node_sources[entity.node_id].add(chunk_id)

    for result in results:
        for entity in result.entities:
            add_node(entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            add_node(relation.subject, result.chunk_id)
            add_node(relation.object, result.chunk_id)
            edge_sources[(relation.subject.node_id, relation.type, relation.object.node_id)].add(result.chunk_id)

    conflicts = 0
    for source, kind, target in list(edge_sources):
        backward = (target, kind, source)
        if kind != "REPLACED_BY" or backward not in edge_sources or source >= target:
            continue  # каждую встречную пару разбираем один раз
        conflicts += 1
        forward_weight, backward_weight = len(edge_sources[(source, kind, target)]), len(edge_sources[backward])
        if forward_weight <= backward_weight:
            del edge_sources[(source, kind, target)]
        if backward_weight <= forward_weight:
            del edge_sources[backward]

    graph = nx.DiGraph(conflicts=conflicts)
    for node_id, entity in nodes.items():
        graph.add_node(node_id, type=entity.type, name=entity.name, sources=sorted(node_sources[node_id]))
    for (source, kind, target), chunks in edge_sources.items():
        graph.add_edge(source, target, type=kind, weight=len(chunks), sources=sorted(chunks))
    return graph
