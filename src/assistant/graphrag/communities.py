"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community

MAX_SUMMARY_WORDS = 150
MAX_FACTS = 40

SUMMARY_PROMPT = (
    "Ты пишешь сводку сообщества графа знаний сервисной службы. "
    "Опиши, какие сущности входят в сообщество, как они связаны и что встречается чаще всего. "
    "Используй только переданные факты и числа, ничего не пересчитывай и не придумывай. "
    f"Не больше {MAX_SUMMARY_WORDS} слов, по-русски, без списков."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    parts = nx.community.louvain_communities(graph.to_undirected(), weight="weight", seed=SEED)
    groups = sorted((sorted(part) for part in parts), key=lambda nodes: (-len(nodes), nodes[0]))
    return [Community(community_id, nodes) for community_id, nodes in enumerate(groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    members = set(community.nodes)
    edges = sorted(
        ((a, b, data) for a, b, data in graph.edges(data=True) if a in members and b in members),
        key=lambda edge: (-edge[2].get("weight", 1), edge[0], edge[1]),
    )

    by_type = Counter(graph.nodes[node].get("type", "?") for node in community.nodes)
    relation_totals: Counter[str] = Counter()
    target_totals: Counter[tuple[str, str]] = Counter()
    for _, target, data in edges:
        weight = data.get("weight", 1)
        relation_totals[data["type"]] += weight
        target_totals[(data["type"], target)] += weight

    lines = [f"Сущностей в сообществе: {len(community.nodes)}."]
    lines.append("По типам: " + ", ".join(f"{kind} – {count}" for kind, count in sorted(by_type.items())) + ".")
    lines.append("Сущности: " + ", ".join(_name(graph, node) for node in community.nodes) + ".")
    if relation_totals:
        lines.append("Сумма весов связей по типам (сколько раз связь найдена в документах):")
        lines += [f"- {kind}: {total}" for kind, total in relation_totals.most_common()]
        lines.append("Чаще всего встречаются (тип связи → сущность, суммарный вес):")
        lines += [f"- {kind} → {_name(graph, node)}: {total}" for (kind, node), total in target_totals.most_common(10)]
        lines.append("Связи (с весом):")
        lines += [
            f"- {_name(graph, a)} —{data['type']}→ {_name(graph, b)}: {data.get('weight', 1)}"
            for a, b, data in edges[:MAX_FACTS]
        ]

    response = llm.chat(
        [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": "\n".join(lines)}],
        max_tokens=400,
    )
    return " ".join(response.content.split()[:MAX_SUMMARY_WORDS])


def _name(graph: nx.DiGraph, node: str) -> str:
    """Читаемое имя узла с типом: «пусковое реле (Component)»."""
    data = graph.nodes[node]
    return f"{data.get('name', node)} ({data.get('type', '?')})"
