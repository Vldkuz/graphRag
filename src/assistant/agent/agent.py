"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

SYSTEM_PROMPT = "..."

def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # TODO: цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps, sources = [], []
    known = set(find_codes(question))
    cache = {}
    while len(steps) < max_steps:
        resp=llm.chat(messages, tools=TOOL_SCHEMAS)
        if not resp.tool_calls :
            return AgentResult(answer=resp.content,steps=steps, sources=sources)
        messages.append(resp.as_message())
        for call in resp.tool_calls:
            if len(steps) >= max_steps:
                break
            if call.name not in tools:
                observation = {"error": f"неизвестный инструмент {call.name}, доступны: {', '.join(tools)}"}
                step = ToolStep(tool=call.name, args=call.arguments, error=observation["error"])
            elif "_raw" in call.arguments:
                observation = {"error": "аргументы не разобраны: передайте их как JSON-объект"}
                step = ToolStep(tool=call.name, args=call.arguments, error=observation["error"])
            elif (part := call.arguments.get("part_id")) is not None and normalize_code(str(part)) not in known:
                observation = {
                    "error": f"номер детали {part} не встречался в вопросе и результатах инструментов, "
                    "сначала найдите его через search_knowledge"
                }
                step = ToolStep(tool=call.name, args=call.arguments, error=observation["error"])
            else:
                key = (call.name, json.dumps(call.arguments, sort_keys=True))
                if key not in cache:
                    cache[key] = tools[call.name](**call.arguments)
                result = cache[key]
                known.update(find_codes(json.dumps(result, ensure_ascii=False)))
                observation = result
                step = ToolStep(tool=call.name, args=call.arguments, result=result)
                for source in result.get("sources", []):
                    if source not in sources:
                        sources.append(source)
            steps.append(step)
            messages.append(tool_message(call, observation))
    lines = [f"Работа прервана: исчерпан лимит в {max_steps} вызовов инструментов."]
    found = [step for step in steps if step.result is not None]
    if found:
        lines.append("Что удалось узнать:")
        lines += [f"- {step.tool}({step.args}): {step.result}" for step in found]
    return AgentResult(answer="\n".join(lines), steps=steps, sources=sources)
