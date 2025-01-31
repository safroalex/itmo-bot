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

# Загрузка переменных окружения (.env)
load_dotenv()

# Ключи для OpenAI (основной и вторичный)
OPENAI_API_KEY_MAIN = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY_SECONDARY = os.getenv("OPENAI_API_KEY_SECONDARY", "")

# Ключи для Google Search
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class PredictionRequest(BaseModel):
    id: int
    query: str

class PredictionResponse(BaseModel):
    id: int
    answer: int
    reasoning: str
    sources: List[HttpUrl]

@app.post("/api/request", response_model=PredictionResponse)
def predict(request_data: PredictionRequest):
    # Шаг 1. Упрощение вопроса (если нужно) — используем вторичный ключ
    simplified_query = simplify_query_with_openai(request_data.query)

    # Шаг 2. Поиск в Google
    found_links = do_web_search(simplified_query, max_links=3)
    if not found_links:
        found_links = do_web_search(simplified_query, max_links=5)

    # Шаг 3. Парсим текст с каждой ссылки
    scraped_texts = []
    for link in found_links:
        text = scrape_html_to_text(link)
        if text:
            scraped_texts.append((link, text))

    # Шаг 4. Формируем ответ с учётом реального содержимого страниц
    answer_text, chosen_answer = generate_answer_with_openai(
        request_data.query,
        scraped_texts
    )

    if chosen_answer is None:
        chosen_answer = -1

    # Валидация ссылок
    valid_sources = []
    for link in found_links:
        try:
            valid_sources.append(HttpUrl(link))
        except Exception:
            pass

    return PredictionResponse(
        id=request_data.id,
        answer=chosen_answer,
        reasoning=answer_text,
        sources=valid_sources
    )

def simplify_query_with_openai(original_query: str) -> str:
    """
    Используем вторичный ключ, чтобы упростить/укоротить запрос
    (не переводим язык, если он русский, а лишь удаляем «шум»).
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY_SECONDARY)
    system_prompt = (
        "Ты упрощатель поисковых запросов. Из длинного текста оставляй только ключевые слова. "
        "Не добавляй лишних слов, не переводи на другие языки."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Убери все лишние слова из запроса:\n{original_query}"
                }
            ],
            max_tokens=50,
            temperature=0.3
        )
        simplified = response.choices[0].message.content.strip()
        return simplified or original_query
    except Exception as e:
        logger.warning(f"Could not simplify query: {e}")
        return original_query

def do_web_search(query: str, max_links: int = 3) -> List[str]:
    """
    Отправляем запрос в Google Custom Search (CSE),
    возвращаем список ссылок.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("Missing Google keys!")
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
        links = [item["link"] for item in items if "link" in item]
        return links[:max_links]
    except Exception as e:
        logger.warning(f"Search error: {e}")
        return []

def scrape_html_to_text(url: str) -> str:
    """
    Скачивает HTML, парсит с помощью BeautifulSoup,
    возвращает текст (без стилей, скриптов и лишних пробелов).
    """
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        # Удаляем скрипты и стили
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n")
        # Чистим пробелы
        text = re.sub(r"\s+", " ", text)
        # Можно ограничить размер, если страница слишком большая
        if len(text) > 5000:
            text = text[:5000] + "..."
        return text.strip()
    except Exception as e:
        logger.info(f"Failed to scrape {url}: {e}")
        return ""

def generate_answer_with_openai(original_query: str, source_texts: List[Tuple[str, str]]):
    """
    Передаём модельке исходный вопрос + тексты (содержимое ссылок).
    Модель анализирует фактическую информацию и выдаёт ответ.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY_MAIN)
    system_prompt = (
        "Ты помощник, который отвечает только на основе предоставленных текстов. "
        "Если вопрос содержит варианты (1..N), выбери верный; иначе ответ -1. "
        "Ссылки не выдумывай, а цитируй реальные URL из списка."
    )

    # Составляем общий контекст
    combined_text = ""
    for link, text in source_texts:
        combined_text += f"[URL: {link}]\n{text}\n\n"

    # Если текст слишком большой, можно дополнительно «сжать» его через GPT,
    # но для примера передаём как есть (уследив за лимитом).
    user_message = f"Вопрос:\n{original_query}\n\nВот тексты:\n{combined_text}\n\nОтветь, ссылаясь на факты из текстов."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=800,
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI main error: {e}")
        return f"Error: {e}", None

    # Пытаемся извлечь из ответа вариант (число 1..10)
    chosen_answer = None
    match = re.search(r"\b([1-9]|10)\b", content)
    if match:
        chosen_answer = int(match.group(1))

    return content, chosen_answer