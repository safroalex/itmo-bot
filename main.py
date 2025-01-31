import re
import os
import logging
from typing import List, Tuple

import requests
import openai
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# === Шаг 1: Базовые схемы данных (пример) ===
class PredictionRequest(BaseModel):
    id: int
    query: str

class PredictionResponse(BaseModel):
    id: int
    answer: int
    reasoning: str
    sources: List[HttpUrl]

# === Инициализация приложения FastAPI ===
app = FastAPI()

# Загрузка переменных окружения
load_dotenv()

OPENAI_API_KEY_MAIN = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY_SECONDARY = os.getenv("OPENAI_API_KEY_SECONDARY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# === Логгер ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("itmo-bot")

# === Утилиты: chunkify, scraping, web_search, prompts ===

def chunkify(text: str, chunk_size: int = 2000, overlap: int = 0) -> List[str]:
    """
    Разбивает текст на части (чанки) длины chunk_size.
    При необходимости можно делать overlap (перекрытие) в символах.
    """
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # если overlap > 0, часть текста будет повторяться
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks

def scrape_html_to_text(url: str) -> str:
    """
    Скачивает и парсит HTML-страницу, убирает скрипты и стили,
    возвращает «очищенный» текст. 
    """
    logger.info(f"Scraping URL: {url}")
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Удаляем скрипты/стили
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        # Если слишком большой текст, урезаем (примерно)
        max_len = 5000
        if len(text) > max_len:
            text = text[:max_len] + "..."
        return text
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return ""

def do_web_search(query: str, max_links: int = 3) -> List[str]:
    """
    Запрос к Google Custom Search (Programmable Search Engine).
    Возвращает список ссылок не более max_links.
    """
    logger.info(f"Searching in Google: {query}")
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("Missing Google API credentials!")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "lr": "lang_ru",
        "hl": "ru",
        "gl": "ru"
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        links = [it["link"] for it in items if "link" in it]
        return links[:max_links]
    except Exception as e:
        logger.warning(f"Google search error: {e}")
        return []

def simplify_query_with_openai(original_query: str) -> str:
    """
    Используем второй ключ для упрощения длинного запроса,
    например, удаляем «лишние» слова, не меняя язык.
    """
    logger.info(f"Simplifying query: {original_query[:80]}...")
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY_SECONDARY)

        # Короткий system-подход, без перевода:
        system_prompt = (
            "Ты упрощатель поисковых запросов. "
            "Извлеки из входной строки только ключевые фразы. "
            "Не переводи на другие языки."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Входной запрос: {original_query}\nВыдай ключевые слова/фразы:"}
            ],
            temperature=0.2,
            max_tokens=50
        )
        simplified = response.choices[0].message.content.strip()
        return simplified if simplified else original_query
    except Exception as e:
        logger.warning(f"Could not simplify query: {e}")
        return original_query

def extract_facts_from_chunk(chunk_text: str, question: str) -> str:
    """
    1-й вызов модели: извлекаем факты/данные из конкретного чанка, 
    относящиеся к вопросу. Возвращаем краткое описание. 
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY_MAIN)
    system_prompt = (
        "Ты помощник, который читает предоставленный текст и извлекает факты, "
        "относящиеся к заданному вопросу. Отвечай только фактами из текста. "
        "Если фактов нет, скажи: 'Нет информации'."
    )
    try:
        user_content = f"Вопрос:\n{question}\n\nТекст чанка:\n{chunk_text}\n\nИзвлеки факты, связанные с вопросом:"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,  # строгий режим, чтобы не выдумывал
            max_tokens=300
        )
        facts = response.choices[0].message.content.strip()
        return facts
    except Exception as e:
        logger.warning(f"extract_facts error: {e}")
        return ""

def combine_and_answer(all_facts: List[str], question: str) -> str:
    """
    2-й вызов модели: объединяем все извлечённые факты
    и просим выбрать правильный вариант (1..N) или -1, если нет данных.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY_MAIN)
    system_prompt = (
        "Ты помощник, который отвечает на вопрос, основываясь только на фактах, "
        "которые переданы ниже. Если в вопросе есть варианты (1..N), выбери правильный, "
        "если данных нет - ответ -1."
    )

    combined_facts = "\n".join(all_facts)
    user_content = (
        f"Вопрос:\n{question}\n\n"
        f"Факты из разных чанков:\n{combined_facts}\n\n"
        "Определи ответ."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            max_tokens=400
        )
        final_answer = response.choices[0].message.content.strip()
        return final_answer
    except Exception as e:
        logger.warning(f"combine_and_answer error: {e}")
        return "Error during final answer"

def extract_choice_number(text: str) -> int:
    """
    Ищем в итоговом тексте число (1..10). Если нет, возвращаем -1.
    """
    match = re.search(r"\b([1-9]|10)\b", text)
    if match:
        return int(match.group(1))
    return -1

# === Основной эндпоинт FastAPI ===
@app.post("/api/request", response_model=PredictionResponse)
def predict(request_data: PredictionRequest):
    """
    Итоговая логика:
    1) Упрощаем запрос (если нужно).
    2) Делаем Google-поиск.
    3) Парсим каждую ссылку в чанки.
    4) Для каждого чанка извлекаем факты.
    5) Объединяем факты, запрашиваем окончательный ответ.
    6) Находим номер ответа (или -1).
    """
    question = request_data.query
    logger.info(f"Processing request {request_data.id} with question: {question}")

    # 1) Упрощение
    simplified_query = simplify_query_with_openai(question)

    # 2) Поиск
    links = do_web_search(simplified_query, max_links=3)
    if not links:
        # Расширяем попытку
        links = do_web_search(simplified_query, max_links=5)

    # 3) Парсим и разбиваем на чанки
    chunked_texts = []
    for link in links:
        page_text = scrape_html_to_text(link)
        # разрезаем на чанки, чтобы не перегружать один запрос
        chunks = chunkify(page_text, chunk_size=2000)
        for chunk in chunks:
            chunked_texts.append((link, chunk))

    # 4) Извлекаем факты из каждого чанка
    all_facts_list = []
    for link, chunk in chunked_texts:
        facts = extract_facts_from_chunk(chunk, question)
        # Если факты не пусты, добавим к итоговому списку
        if facts and not re.search(r"нет информации|no info", facts, re.IGNORECASE):
            # Можно добавить ссылку, чтоб знать откуда факты
            all_facts_list.append(f"[{link}]: {facts}")

    # 5) Комбинируем факты и получаем финальный ответ
    final_response = combine_and_answer(all_facts_list, question)
    chosen_answer = extract_choice_number(final_response)

    # Собираем поля для ответа
    # Если не удалось найти соответствие в тексте, будет -1
    if chosen_answer is None:
        chosen_answer = -1

    # Валидация ссылок
    valid_sources = []
    for link in links:
        try:
            valid_sources.append(HttpUrl(link))
        except Exception:
            pass

    result = PredictionResponse(
        id=request_data.id,
        answer=chosen_answer,
        reasoning=final_response,
        sources=valid_sources
    )

    return result