import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

from ddgs import DDGS
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

SYSTEM_PROMPT = """Ты эксперт по разведению кошек.
Помогай с породами, вязкой, беременностью, уходом за котятами и генетикой окрасов.
Отвечай по-русски, ясно и практично.

Инструменты:
- `get_cat_breed_info` — сначала ищи породу в локальной базе.
- `search_breed_online` — если породы нет в базе или нужны свежие данные из интернета.
- `estimate_due_date` — расчёт ожидаемой даты окота по дате вязки.
- `check_breeding_readiness` — чеклист готовности животного к вязке.

Не выдумывай медицинские диагнозы: при серьёзных симптомах советуй обратиться к ветеринару."""

BREED_DB = {
    "мейн-кун": {
        "origin": "США (Мэн)",
        "size": "крупная",
        "weight_kg": "кот 6–12, кошка 4–8",
        "gestation_days": "63–67",
        "litter_size": "обычно 3–6 котят",
        "notes": "Поздние сроки половой зрелости; важно проверять HCM (гипертрофическая кардиомиопатия).",
    },
    "британская короткошёрстная": {
        "origin": "Великобритания",
        "size": "средняя / крупная",
        "weight_kg": "кот 5–8, кошка 3–6",
        "gestation_days": "63–65",
        "litter_size": "обычно 3–5 котят",
        "notes": "Склонность к лишнему весу; следить за почками у возрастных животных.",
    },
    "сиамская": {
        "origin": "Таиланд",
        "size": "средняя",
        "weight_kg": "кот 4–6, кошка 3–5",
        "gestation_days": "63–65",
        "litter_size": "обычно 3–6 котят",
        "notes": "Очень общительная; окрасы связаны с температурозависимым аллелем колорпойнт.",
    },
    "сфинкс": {
        "origin": "Канада",
        "size": "средняя",
        "weight_kg": "кот 3.5–7, кошка 3–5",
        "gestation_days": "63–65",
        "litter_size": "обычно 3–5 котят",
        "notes": "Нужен особый уход за кожей; чувствительны к холоду.",
    },
    "персидская": {
        "origin": "Иран / Европа (современный тип)",
        "size": "средняя",
        "weight_kg": "кот 4–7, кошка 3–5",
        "gestation_days": "63–65",
        "litter_size": "обычно 2–4 котёнка",
        "notes": "Частый уход за шерстью; у брахицефальных линий — риски дыхательных проблем.",
    },
    "бенгальская": {
        "origin": "США",
        "size": "средняя / крупная",
        "weight_kg": "кот 4.5–8, кошка 3.5–6",
        "gestation_days": "63–65",
        "litter_size": "обычно 3–5 котят",
        "notes": "Активная порода; для разведения важны F-поколения и статус гибридности.",
    },
}

BREED_ALIASES = {
    "мейн кун": "мейн-кун",
    "maine coon": "мейн-кун",
    "британка": "британская короткошёрстная",
    "british shorthair": "британская короткошёрстная",
    "siamese": "сиамская",
    "sphynx": "сфинкс",
    "persian": "персидская",
    "bengal": "бенгальская",
}


def _normalize_breed(breed: str) -> str:
    key = breed.strip().lower()
    return BREED_ALIASES.get(key, key)


@tool
def get_cat_breed_info(breed: str) -> str:
    """Справка по породе кошек из локальной базы: размер, сроки беременности, помёт, особенности.
    Примеры: мейн-кун, британская короткошёрстная, сиамская, сфинкс, персидская, бенгальская.
    Если породы нет в базе — используй search_breed_online.
    """
    key = _normalize_breed(breed)
    info = BREED_DB.get(key)
    if not info:
        known = ", ".join(BREED_DB)
        return (
            f"Порода «{breed}» не найдена в локальной базе. "
            f"Доступны: {known}. "
            "Вызови search_breed_online для поиска в интернете."
        )
    return (
        f"Порода: {key}\n"
        f"Происхождение: {info['origin']}\n"
        f"Размер: {info['size']}\n"
        f"Вес: {info['weight_kg']} кг\n"
        f"Беременность: {info['gestation_days']} дней\n"
        f"Размер помёта: {info['litter_size']}\n"
        f"Для заводчика: {info['notes']}"
    )


@tool
def search_breed_online(breed: str) -> str:
    """Поиск в интернете сведений о породе кошек (для неизвестных пород или уточнения данных)."""
    query = f"порода кошек {breed} разведение беременность характеристики"
    try:
        results = DDGS().text(query, region="ru-ru", max_results=5)
    except Exception as e:
        return f"Поиск не удался: {e}"
    if not results:
        return f"По запросу «{breed}» ничего не найдено."
    lines = [f"Результаты поиска по породе «{breed}»:"]
    for i, item in enumerate(results, 1):
        title = item.get("title", "")
        body = item.get("body", "")
        href = item.get("href", "")
        lines.append(f"{i}. {title}\n   {body}\n   Источник: {href}")
    return "\n".join(lines)


@tool
def estimate_due_date(mating_date: str, breed: str = "") -> str:
    """Рассчитать ожидаемую дату окота по дате вязки.
    mating_date: дата вязки в формате ГГГГ-ММ-ДД (например 2026-03-01).
    breed: опционально название породы — для уточнения срока беременности.
    """
    try:
        start = datetime.strptime(mating_date.strip(), "%Y-%m-%d").date()
    except ValueError:
        return "Неверный формат даты. Используй ГГГГ-ММ-ДД, например 2026-03-01."

    min_days, max_days = 63, 65
    breed_note = "стандартный срок для домашних кошек"
    if breed:
        key = _normalize_breed(breed)
        info = BREED_DB.get(key)
        if info:
            parts = info["gestation_days"].replace("–", "-").split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                min_days, max_days = int(parts[0]), int(parts[1])
                breed_note = f"по данным локальной базы для «{key}»"

    early = start + timedelta(days=min_days)
    late = start + timedelta(days=max_days)
    mid = start + timedelta(days=(min_days + max_days) // 2)
    return (
        f"Дата вязки: {start.isoformat()}\n"
        f"Ожидаемый окот: {early.isoformat()} – {late.isoformat()} "
        f"(ориентир ~{mid.isoformat()})\n"
        f"Срок: {min_days}–{max_days} дней ({breed_note}).\n"
        "Это ориентир, не медицинский прогноз — при отклонениях обратись к ветеринару."
    )


@tool
def check_breeding_readiness(
    sex: str,
    age_months: int,
    weight_kg: float,
    vaccinated: bool = False,
    health_checked: bool = False,
) -> str:
    """Чеклист готовности кошки/кота к вязке.
    sex: 'кошка' или 'кот'.
    age_months: возраст в месяцах.
    weight_kg: вес в кг.
    vaccinated: есть ли актуальные прививки.
    health_checked: пройден ли вет-осмотр / скрининг перед вязкой.
    """
    sex_norm = sex.strip().lower()
    checks: list[str] = []
    blockers: list[str] = []

    if sex_norm in {"кошка", "female", "f"}:
        if age_months < 12:
            blockers.append("Кошке обычно лучше не вязать раньше 12 месяцев (лучше после 1–2 течек у зрелого животного).")
        elif age_months > 84:
            checks.append("Возраст высокий (>7 лет): вязка только после консультации с ветеринаром.")
        else:
            checks.append(f"Возраст {age_months} мес. — в допустимом диапазоне для многих пород.")
        if weight_kg < 3.0:
            blockers.append("Вес слишком низкий для вязки (<3 кг) — риск для здоровья.")
        else:
            checks.append(f"Вес {weight_kg} кг — ок для ориентира.")
    elif sex_norm in {"кот", "male", "m"}:
        if age_months < 12:
            blockers.append("Кота обычно не используют в разведении раньше 12 месяцев.")
        else:
            checks.append(f"Возраст {age_months} мес. — ок по минимальному порогу.")
        if weight_kg < 3.5:
            blockers.append("Вес низкий для племенного кота (<3.5 кг) — уточни норму для породы.")
        else:
            checks.append(f"Вес {weight_kg} кг — ок для ориентира.")
    else:
        return "Укажи sex как «кошка» или «кот»."

    if vaccinated:
        checks.append("Прививки отмечены как актуальные.")
    else:
        blockers.append("Нет отметки об актуальных прививках — обнови вакцинацию до вязки.")

    if health_checked:
        checks.append("Вет-осмотр / скрининг отмечен как пройденный.")
    else:
        blockers.append("Нет отметки о вет-осмотре — желателен осмотр и породные тесты (HCM и др.).")

    status = "НЕ ГОТОВ(А)" if blockers else "УСЛОВНО ГОТОВ(А)"
    lines = [f"Итог: {status}", "", "Плюсы:"]
    lines += [f"- {c}" for c in checks] or ["- нет"]
    lines += ["", "Что закрыть до вязки:"]
    lines += [f"- {b}" for b in blockers] or ["- критичных блокеров нет"]
    lines.append("\nЭто общий чеклист, не замена правилам клуба/питомника и совету ветеринара.")
    return "\n".join(lines)


model = init_chat_model(
    "openai:gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    max_tokens=1500,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[
        get_cat_breed_info,
        search_breed_online,
        estimate_due_date,
        check_breeding_readiness,
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)


def main() -> None:
    config = {"configurable": {"thread_id": "cat-breeder-1"}}
    print("Эксперт по разведению кошек. Пустая строка или 'exit' — выход.\n")
    while True:
        user_input = input("Вы: ").strip()
        if not user_input or user_input.lower() in {"exit", "quit", "выход"}:
            break
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
        )
        print(f"Агент: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
