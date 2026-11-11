"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter, defaultdict

import networkx as nx

from assistant.graphrag.community_context import community_context_edges, community_models, model_product_lines
from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community

MAX_SUMMARY_WORDS = 150
MAX_FACTS = 40

SUMMARY_PROMPT = (
    "Ты пишешь краткое пояснение к сводке сообщества графа знаний сервисной службы. "
    "Обязательный блок статистики и замен будет сохранён кодом перед твоим пояснением. "
    "Не повторяй его, не пересчитывай числа и не добавляй новые. "
    "Опиши прочие существенные связи только по переданным фактам, по-русски, до 40 слов. "
    "Частота отказов определяется только FAILED, а не популярностью INDICATES, IS_A или SUPPLIED_BY. "
    "Статистика закреплена за моделями-владельцами этого сообщества: не приписывай её чужим моделям "
    "или всей линейке, если здесь представлены не все её модели. Модель и линейка — разные типы сущностей."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы алгоритмом Лувена с фиксированным seed."""
    parts = nx.community.louvain_communities(graph.to_undirected(), weight="weight", seed=SEED)
    groups = sorted((sorted(part) for part in parts), key=lambda nodes: (-len(nodes), nodes[0]))
    return [Community(community_id, nodes) for community_id, nodes in enumerate(groups)]


def _model_statistics(community: Community, graph: nx.DiGraph) -> list[str]:
    """Считаем веса FAILED один раз у модели-владельца, включая внешние компоненты."""
    lines: list[str] = []
    by_line: dict[str, Counter[str]] = defaultdict(Counter)
    owners: dict[str, list[str]] = defaultdict(list)
    for model in community_models(community, graph):
        product_lines = model_product_lines(model, graph)
        line_names = ",".join(graph.nodes[node]["name"] for node in product_lines) or "линейка неизвестна"
        counts = Counter(
            {
                target: data.get("weight", 1)
                for _, target, data in graph.out_edges(model, data=True)
                if data["type"] == "FAILED"
            }
        )
        if not counts:
            lines.append(f"Модель {graph.nodes[model]['name']} ({line_names}): данных FAILED нет.")
            continue
        components = "; ".join(f"{graph.nodes[node]['name']}={count}" for node, count in sorted(counts.items()))
        lines.append(
            f"Модель {graph.nodes[model]['name']} ({line_names}): FAILED всего={sum(counts.values())}; {components}."
        )
        for product_line in product_lines:
            by_line[product_line].update(counts)
            owners[product_line].append(graph.nodes[model]["name"])
    for product_line, counts in sorted(by_line.items()):
        components = "; ".join(f"{graph.nodes[node]['name']}={count}" for node, count in sorted(counts.items()))
        models = ",".join(owners[product_line])
        lines.append(f"Итого FAILED {graph.nodes[product_line]['name']} только по моделям {models}: {components}.")
    return lines


def _replacement_lines(community: Community, graph: nx.DiGraph) -> list[str]:
    """Связываем снятые детали с моделями и линейками, а не только с сообществом детали."""
    installed: dict[str, list[str]] = defaultdict(list)
    for model in community_models(community, graph):
        product_lines = model_product_lines(model, graph)
        names = ",".join(graph.nodes[node]["name"] for node in product_lines) or "линейка неизвестна"
        for _, part, data in graph.out_edges(model, data=True):
            if data["type"] == "HAS_PART":
                installed[part].append(f"{graph.nodes[model]['name']} ({names})")
    lines: list[str] = []
    for part, models in sorted(installed.items()):
        pending = [part]
        visited: set[str] = set()
        replacements: set[tuple[str, str]] = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            for _, target, data in graph.out_edges(node, data=True):
                if data["type"] == "REPLACED_BY":
                    replacements.add((node, target))
                    pending.append(target)
        if replacements:
            chain = "; ".join(f"{graph.nodes[a]['name']}→{graph.nodes[b]['name']}" for a, b in sorted(replacements))
            lines.append(
                f"Снята с производства: {graph.nodes[part]['name']}, модели {','.join(models)}; "
                f"замены (снятая→новая): {chain}."
            )
    return lines


def _fit_lines(lines: list[str], budget: int) -> tuple[list[str], bool]:
    """Не разрываем запись статистики или цепочку замен границей в 150 слов."""
    selected: list[str] = []
    used = 0
    for line in lines:
        count = len(line.split())
        if used + count > budget:
            return selected, True
        selected.append(line)
        used += count
    return selected, False


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка до 150 слов: вычисленные показатели и одно пояснение модели."""
    required = _model_statistics(community, graph) + _replacement_lines(community, graph)
    # Числа уже посчитаны: модель не должна терять их при пересказе или складывать дважды.
    if required:
        required.insert(
            0, "FAILED — веса упоминаний, не число уникальных аварий. Строки моделей и итоги не складываются."
        )
    important, incomplete = _fit_lines(required, MAX_SUMMARY_WORDS - 3)
    if incomplete:
        important.append("Сводка неполная.")

    edges = sorted(
        community_context_edges(community, graph),
        key=lambda edge: (-graph.edges[edge].get("weight", 1), edge),
    )
    facts = [f"{_name(graph, a)} —{graph.edges[a, b]['type']}→ {_name(graph, b)}" for a, b in edges[:MAX_FACTS]]
    owners = ", ".join(graph.nodes[node]["name"] for node in community_models(community, graph)) or "нет"
    prompt = (
        f"Модели-владельцы статистики: {owners}.\n"
        "Обязательный блок (числа посчитаны кодом, только FAILED):\n"
        + "\n".join(important)
        + "\nПрочие факты (не использовать частоту упоминания как частоту отказов):\n"
        + "\n".join(facts)
        + "\nСущности: "
        + ", ".join(_name(graph, node) for node in community.nodes)
    )
    response = llm.chat(
        [{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": prompt}],
        max_tokens=400,
    )
    remaining = MAX_SUMMARY_WORDS - len(" ".join(important).split())
    explanation = " ".join(response.content.split()[:remaining])
    return " ".join([*important, explanation]).strip()


def _name(graph: nx.DiGraph, node: str) -> str:
    """Читаемое имя узла с типом: «пусковое реле (Component)»."""
    data = graph.nodes[node]
    return f"{data.get('name', node)} ({data.get('type', '?')})"
