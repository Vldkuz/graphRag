"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

SYSTEM_PROMPT = """Ты – ассистент сервисного инженера компании «ПолярТех» по холодильному оборудованию.
Отвечай по-русски, коротко и по делу.

Инструменты:
- search_knowledge – факты о модели, коде ошибки, детали и её актуальной замене. Вызывай первым,
  если в вопросе нет точного номера детали (вида P-1042 или P-1042B).
- summarize_topic – обзорные вопросы: типичные поломки, общие закономерности.
- check_stock – остаток детали на складе: Москва, Санкт-Петербург, Казань, Новосибирск.
- get_price – цена детали в рублях.
- create_service_order – заказ детали для модели оборудования.

Правила:
1. Коды моделей, деталей и ошибок, цены и остатки бери только из вопроса и результатов инструментов.
   Ничего не придумывай и не угадывай номера деталей.
2. Если деталь снята с производства (discontinued), найди её замену через search_knowledge
   и работай с заменой.
3. Если инструмент вернул error с «не найдено», данных нет. Это не то же самое, что остаток 0:
   остаток 0 значит, что деталь есть в ERP, но на этом складе закончилась.
4. Если ERP недоступна, так и напиши: «данные ERP получить не удалось». Не подставляй цифры.
5. Заказ оформляй только когда пользователь назвал деталь, модель, склад и количество.
   Если чего-то не хватает, спроси. Количество никогда не выбирай сам.
6. Если заказ ждёт подтверждения, перескажи пользователю его параметры и попроси подтвердить.
   Если заказ создан, назови его номер.
7. Если ни инструменты, ни вопрос не дают ответа, напиши: «В базе знаний не найдено».
"""

_CODE_LIKE = re.compile(r"[A-Za-zА-Яа-яЁё]+-?\d+[A-Za-zА-Яа-яЁё]*")


_THOUSANDS = re.compile(r"(?<=\d)[ \u00a0](?=\d{3}(?!\d))")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")
REFUSAL = "Не могу дать ответ: модель называет данные, которых нет в вопросе и результатах инструментов."


def _numbers_named(text: str) -> set[str]:
    text = _LIST_MARKER.sub(" ", _THOUSANDS.sub("", text))
    return set(re.findall(r"\d+", _CODE_LIKE.sub(" ", text)))


def _ungrounded(answer: str, known: set[str], seen: set[str]) -> list[str]:
    codes = [code for code in find_codes(answer) if code not in known]
    numbers = sorted(_numbers_named(answer) - seen)
    return codes + numbers


def _normalized(arguments: dict) -> dict:
    args = dict(arguments)
    for name in ("part_id", "model"):
        if isinstance(args.get(name), str) and (code := normalize_code(args[name])):
            args[name] = code
    qty = args.get("qty")
    if isinstance(qty, str) and qty.strip().isdigit():
        args["qty"] = int(qty)
    elif isinstance(qty, float) and qty.is_integer():
        args["qty"] = int(qty)
    return args


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
    numbers = _numbers_named(question)
    seen = numbers | set(re.findall(r"\d+", question))
    pending_action = None
    corrected = False
    while len(steps) < max_steps:
        resp = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not resp.tool_calls:
            answer, refused = resp.content, "не найдено" in resp.content.lower()
            invented = _ungrounded(answer, known, seen)
            if invented and not corrected:
                corrected = True
                messages.append(resp.as_message())
                messages.append(
                    {
                        "role": "user",
                        "content": f"В ответе есть {', '.join(invented)}: этого нет ни в вопросе, "
                        "ни в результатах инструментов. Исправьте ответ, используя только эти данные.",
                    }
                )
                continue
            if invented:
                answer, refused = REFUSAL, True
            return AgentResult(
                answer=answer,
                steps=steps,
                sources=sources,
                refused=refused,
                needs_confirmation=pending_action is not None,
                pending_action=pending_action,
            )
        messages.append(resp.as_message())
        for call in resp.tool_calls:
            if len(steps) >= max_steps:
                break
            args = _normalized(call.arguments)
            if call.name not in tools:
                observation = {"error": f"неизвестный инструмент {call.name}, доступны: {', '.join(tools)}"}
                step = ToolStep(tool=call.name, args=args, error=observation["error"])
            elif "_raw" in args:
                observation = {"error": "аргументы не разобраны: передайте их как JSON-объект"}
                step = ToolStep(tool=call.name, args=args, error=observation["error"])
            elif unknown := [
                str(value)
                for name in ("part_id", "model")
                if (value := args.get(name)) is not None and value not in known
            ]:
                observation = {
                    "error": f"код {unknown[0]} не встречался в вопросе и результатах инструментов, "
                    "сначала найдите его через search_knowledge или спросите пользователя"
                }
                step = ToolStep(tool=call.name, args=args, error=observation["error"])
            elif call.name == "create_service_order" and str(args.get("qty")) not in numbers:
                observation = {
                    "error": "количество не названо пользователем, заказ не создан: спросите, сколько штук заказать"
                }
                step = ToolStep(tool=call.name, args=args, error=observation["error"])
            elif call.name == "create_service_order" and not confirm:
                pending_action = {"tool": call.name, "args": args}
                observation = {
                    "error": "заказ не создан: нужно подтверждение пользователя, перескажите ему параметры заказа"
                }
                step = ToolStep(tool=call.name, args=args, error=observation["error"])
            else:
                key = (call.name, json.dumps(args, sort_keys=True))
                result, error = cache.get(key), None
                if result is None:
                    for _ in range(3):
                        try:
                            result = tools[call.name](**args)
                            break
                        except ErpUnavailable as exc:
                            error = str(exc)
                    if result is not None:
                        cache[key] = result
                if result is None:
                    observation = {"error": f"ERP недоступна после 3 попыток: {error}"}
                    step = ToolStep(tool=call.name, args=args, error=observation["error"])
                else:
                    dumped = json.dumps(result, ensure_ascii=False)
                    known.update(find_codes(dumped))
                    seen.update(re.findall(r"\d+", dumped))
                    observation = result
                    step = ToolStep(tool=call.name, args=args, result=result, error=result.get("error"))
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
    return AgentResult(
        answer="\n".join(lines),
        steps=steps,
        sources=sources,
        needs_confirmation=pending_action is not None,
        pending_action=pending_action,
    )
