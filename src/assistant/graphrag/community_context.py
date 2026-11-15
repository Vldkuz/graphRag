"""Факты сообщества с контекстом моделей, включая связи через границы Louvain."""

import networkx as nx

from assistant.infra.schemas import Community


def community_models(community: Community, graph: nx.DiGraph) -> list[str]:
    """Модели-владельцы статистики; модель входит ровно в одно сообщество."""
    return sorted(node for node in community.nodes if graph.nodes[node].get("type") == "Model")


def model_product_lines(model: str, graph: nx.DiGraph) -> list[str]:
    """Линейки модели из BELONGS_TO, даже если они в других сообществах."""
    return sorted(target for _, target, data in graph.out_edges(model, data=True) if data["type"] == "BELONGS_TO")


def community_context_edges(community: Community, graph: nx.DiGraph) -> list[tuple[str, str]]:
    """Возвращает внутренние факты и контекст моделей-владельцев сообщества.

    Каждая модель принадлежит одному сообществу. Только оно описывает её отказы:
    компонент может быть снаружи, но входящие FAILED чужих моделей не добавляются.
    Для установленных деталей сохраняются полные цепочки замен, без перехода к чужим моделям.
    """
    members = set(community.nodes)
    edges = set(graph.subgraph(members).edges)
    parts: list[str] = []
    for model in community_models(community, graph):
        for _, target, data in graph.out_edges(model, data=True):
            if data["type"] in {"FAILED", "BELONGS_TO", "HAS_PART"}:
                edges.add((model, target))
            if data["type"] == "HAS_PART":
                parts.append(target)

    visited: set[str] = set()
    while parts:
        part = parts.pop()
        if part in visited:
            continue
        visited.add(part)
        for _, target, data in graph.out_edges(part, data=True):
            if data["type"] == "REPLACED_BY":
                edges.add((part, target))
                parts.append(target)
    return sorted(edges)
