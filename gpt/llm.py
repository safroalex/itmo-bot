import os
import re
import logging
import openai
from typing import List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("itmo-bot")

OPENAI_API_KEY_MAIN = os.getenv("OPENAI_API_KEY", "")

client = openai.Client(api_key=OPENAI_API_KEY_MAIN)

def extract_choice_number(text: str) -> int:
    """
    Ищем число 1..10 в тексте. Если не нашли, вернём -1.
    """
    match = re.search(r"\b([1-9]|10)\b", text)
    if match:
        return int(match.group(1))
    return -1

def finalize_answer_sync(facts_list: List[str], question: str) -> str:
    if not facts_list:
        return "Недостаточно данных."

    system_prompt = (
        "Ты помощник, который отвечает на вопрос, используя только предоставленные факты. "
        "Если в вопросе есть варианты (1..N), выбери правильный. Если нет данных, ответ -1."
    )
    user_content = (
        f"Вопрос:\n{question}\n\n"
        f"Факты:\n{' --- '.join(facts_list)}\n\n"
        "Ответ:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"finalize_answer_sync error: {e}")
        return "Ошибка при формировании ответа"

import re

def parse_variants_in_question(question: str) -> int:
    """
    Ищем варианты в тексте вопроса вида:
      "1. ...\n2. ...\n3. ..."
    Возвращаем кол-во найденных вариантов (максимальную цифру).
    
    Например, если есть "1. 2007\n2. 2009\n3. 2011", вернём 3.
    Если формата нет или не нашли, вернём 0 (значит нет вариантов).
    """
    # Поищем все упоминания 'X. ' в начале строки (где X — цифра)
    # или упоминания 'X. ' c некоторым текстом
    # Упростим, допустим, ищем по паттерну r"(?m)^(\d+)\.\s"
    pattern = re.compile(r"(?m)^(\d+)\.\s")
    matches = pattern.findall(question)
    if not matches:
        return 0
    # Преобразуем в int
    found = list(map(int, matches))
    # вернём максимальное число
    return max(found)

def generate_fallback_answer(question: str) -> str:
    """
    GPT "из головы", но:
    - Если есть варианты 1..N, просим его обязательно выбрать одну из них (и не писать -1).
    - Если вариантов нет, пусть пишет что угодно, но мы позже выставим answer = null.
    """
    num_variants = parse_variants_in_question(question)

    if num_variants > 0:
        # Есть варианты. GPT ДОЛЖЕН выбрать одну из цифр 1..num_variants, НЕ говорить -1.
        system_prompt = (
            "Ты — помощник, отвечающий на вопрос с помощью всей своей базы знаний. "
            f"В вопросе есть варианты ответа (1..{num_variants}). "
            "Не используй '-1'. "
            "Тебе надо однозначно выбрать одну цифру и кратко объяснить почему."
        )
        user_content = (
            f"Вопрос:\n{question}\n\n"
            "Ответ: укажи одну цифру и короткий комментарий."
        )
    else:
        # Нет вариантов. Можно дать любой ответ, но -1 не используем.
        system_prompt = (
            "Ты — помощник с базой знаний GPT. "
            "Отвечай на вопрос свободно, но никогда не пиши '-1'. "
            "Вопрос не содержит явных вариантов, так что просто дай разумный ответ."
        )
        user_content = f"Вопрос: {question}\n\nОтвет:"

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"generate_fallback_answer error: {e}")
        return "Ошибка при обращении к GPT."