"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
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
    for _ in range(max_steps):
        resp=llm.chat(messages, tools=TOOL_SCHEMAS)
        if not resp.tool_calls :
            return AgentResult(answer=resp.content,steps=steps, sources=sources)
        messages.append(resp.as_message())
        for call in resp.tool_calls:
            if call.name not in tools:
                observation = {"error": f"неизвестный инструмент {call.name}, доступны: {', '.join(tools)}"}
                step = ToolStep(tool=call.name, args=call.arguments, error=observation["error"])
            elif "_raw" in call.arguments:
                observation = {"error": "аргументы не разобраны: передайте их как JSON-объект"}
                step = ToolStep(tool=call.name, args=call.arguments, error=observation["error"])
            else:
                result = tools[call.name](**call.arguments)
                observation = result
                step = ToolStep(tool=call.name, args=call.arguments, result=result)
                for source in result.get("sources", []):
                    if source not in sources:
                        sources.append(source)
            steps.append(step)
            messages.append(tool_message(call, observation))
    return AgentResult(answer="Работа прервана", steps=steps, sources=sources)
