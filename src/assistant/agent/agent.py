"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS, WAREHOUSES
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, LLMResponse, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

SYSTEM_PROMPT = """Ты – ассистент сервисного инженера компании «ПолярТех» по холодильному оборудованию.
Отвечай по-русски, коротко и по делу.

Инструменты:
- search_knowledge – факты о модели, коде ошибки, детали, её актуальной замене и поставщике.
  Вызывай перед ответом о них, даже если известен точный номер детали.
  Перед обращением в ERP вызывай первым, если точный номер детали ещё неизвестен.
- summarize_topic – обзорные вопросы: типичные поломки, общие закономерности,
  замены и снятые с производства детали по всей линейке оборудования.
- check_stock – остаток детали на складе: Москва, Санкт-Петербург, Казань, Новосибирск.
- get_price – цена детали в рублях.
- create_service_order – заказ детали для модели оборудования.

Правила:
1. Коды моделей, деталей и ошибок, цены и остатки бери только из вопроса и результатов инструментов.
   Ничего не придумывай и не угадывай номера деталей.
2. Если деталь снята с производства (discontinued), найди её замену через search_knowledge
   и работай с заменой.
   search_knowledge отвечает только на техническую часть вопроса. Если он нашёл деталь,
   но сообщил об отсутствии цены или остатка в базе знаний, запроси их через get_price/check_stock.
   Отсутствие этих сведений в документах не означает их отсутствия в ERP.
3. Если инструмент вернул error с «не найдено», данных нет. Это не то же самое, что остаток 0:
   остаток 0 значит, что деталь есть в ERP, но на этом складе закончилась.
4. Если ERP недоступна, так и напиши: «данные ERP получить не удалось». Не подставляй цифры.
5. Заказ оформляй только когда пользователь назвал деталь, модель, склад и количество.
   Если чего-то не хватает, спроси. Количество никогда не выбирай сам.
6. Если заказ ждёт подтверждения, перескажи пользователю его параметры и попроси подтвердить.
   Когда все параметры заказа известны, вызови create_service_order: код сам остановит создание
   до подтверждения и сохранит ожидающее действие. Одного текстового вопроса о подтверждении недостаточно.
   Если заказ создан, назови его номер.
7. Если ни инструменты, ни вопрос не дают ответа, напиши: «В базе знаний не найдено».
"""

_CODE_LIKE = re.compile(r"[A-Za-zА-Яа-яЁё]+-?\d+[A-Za-zА-Яа-яЁё]*")


_THOUSANDS = re.compile(r"(?<=\d)[ \u00a0](?=\d{3}(?!\d))")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")
REFUSAL = "Не могу дать ответ: модель называет данные, которых нет в вопросе и результатах инструментов."
_KNOWLEDGE_TOOLS = {"search_knowledge", "summarize_topic"}
_KNOWLEDGE_QUERY = re.compile(
    r"поставщик|поставля|замен|ошиб|полом|лома|отказ|неисправ|совмест|подойд|подход|установ|"
    r"причин|процедур|ремонт|обслуж|эксплуатац|как(?:ая|ие|ую) детал|какое? реле",
    re.IGNORECASE,
)
_ERP_QUERY = re.compile(r"\b(?:цен|стоим|стоит|стоят|остат|склад|налич|есть ли|закаж|заказ|оформ)", re.IGNORECASE)
_GLOBAL_QUERY = re.compile(
    r"чаще|типич|закономер|обзор|в целом|тенденц|сравн|статист|больше всего"
    # Обзор замен требует и группы оборудования, и запроса списка; одиночная замена остаётся local.
    r"|(?=.*(?:линейк|семейств|серии))(?=.*(?:снят|замен|устар))"
    r"(?=.*(?:\bкакие\b|\bвсе\b|\bсписок\b|\bперечисл))",
    re.IGNORECASE | re.DOTALL,
)
_ORDER_INTENT = re.compile(r"\b(?:оформи(?:те)?\s+заказ|закажи(?:те)?|создай(?:те)?\s+заказ)\b", re.IGNORECASE)
_ORDER_QUANTITY = re.compile(
    r"\b[1-9]\d*\s*(?:шт(?:ук[аиу]?)?|единиц[уы]?|детал[ьи])\b"
    r"|\bколичество\s*[:=]?\s*[1-9]\d*\b|\bзакажи(?:те)?\s+[1-9]\d*\b",
    re.IGNORECASE,
)


def _needs_knowledge(question: str) -> bool:
    """Цена, остаток и оформление заказа достаточны из ERP; технические сведения — из поиска."""
    return bool(_KNOWLEDGE_QUERY.search(question)) or not _ERP_QUERY.search(question)


def _order_is_ready(question: str, known: set[str]) -> bool:
    """Проверяем наличие параметров, но сами не выбираем деталь и не составляем заказ."""
    return bool(
        _ORDER_INTENT.search(question)
        and _ORDER_QUANTITY.search(question)
        and any(warehouse.casefold() in question.casefold() for warehouse in WAREHOUSES)
        and any(code.startswith(("AR-", "BZ-", "GS-")) for code in find_codes(question))
        and any(code.startswith("P-") for code in known)
    )


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
    order_requested = False
    needs_knowledge = _needs_knowledge(question)
    while len(steps) < max_steps:
        resp = llm.chat(messages, tools=TOOL_SCHEMAS)
        knowledge_steps = [step for step in steps if step.tool in _KNOWLEDGE_TOOLS]
        if not resp.tool_calls and needs_knowledge and not knowledge_steps and pending_action is None:
            # Модель пропустила поиск: один read-only вызов проходит тот же учёт шагов и ошибок.
            # Ошибка другого инструмента не подтверждает технический ответ и не заменяет поиск.
            tool_name = "summarize_topic" if _GLOBAL_QUERY.search(question) else "search_knowledge"
            if tool_name not in tools:
                return AgentResult(
                    answer="Поиск по базе знаний недоступен; ответ не подтверждён источниками.",
                    steps=steps,
                    sources=sources,
                    refused=True,
                )
            resp = LLMResponse("", [ToolCall("required-knowledge-search", tool_name, {"question": question})])
        if not resp.tool_calls:
            searched = any(step.result is not None and not step.error for step in knowledge_steps)
            if needs_knowledge and knowledge_steps and not searched and pending_action is None:
                return AgentResult(
                    answer="Не удалось получить сведения из базы знаний; ответ не подтверждён источниками.",
                    steps=steps,
                    sources=sources,
                    refused=True,
                )
            last_result = steps[-1].result if steps else None
            if last_result and last_result.get("refused") and last_result.get("error"):
                return AgentResult(answer=last_result["error"], steps=steps, sources=sources, refused=True)
            if (
                pending_action is None
                and not any(step.tool == "create_service_order" for step in steps)
                and _order_is_ready(question, known)
            ):
                if not order_requested and "create_service_order" in tools:
                    order_requested = True
                    messages.append(resp.as_message())
                    messages.append(
                        {
                            "role": "user",
                            "content": "Пользователь попросил оформить заказ, и его параметры уже известны. "
                            "Вызовите create_service_order с этими параметрами, чтобы подготовить "
                            "структурированное подтверждение. Текстовая просьба подтвердить заказ "
                            "не создаёт ожидающего действия. Не придумывайте параметры.",
                        }
                    )
                    continue
                return AgentResult(
                    answer="Не удалось подготовить заказ к подтверждению. Заказ не создан.",
                    steps=steps,
                    sources=sources,
                    refused=True,
                )
            answer = resp.content
            refused = bool(last_result and last_result.get("refused")) or bool(
                re.search(r"не найден[аоы]?\b", answer.lower())
            )
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
            if call.name in _KNOWLEDGE_TOOLS and isinstance(args.get("question"), str):
                refinement = args["question"].strip()
                if refinement and refinement != question.strip():
                    args["question"] = f"Исходный вопрос пользователя: {question}\nУточнение поиска: {refinement}"
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
