import time
import re
import os
import logging
from typing import List, Tuple

import openai
import requests
import urllib.parse
from fastapi import FastAPI, HTTPException, Request
from pydantic import HttpUrl
from dotenv import load_dotenv

from schemas.request import PredictionRequest, PredictionResponse

# Инициализация FastAPI
app = FastAPI()

# Загрузка переменных окружения (если нужно)
load_dotenv()

# Ключ основной модели (для генерации ответа)
OPENAI_API_KEY_MAIN = os.getenv("OPENAI_API_KEY", "")

# Ключ для "агента-упрощателя"
# Можно либо вынести в переменную окружения (например OPENAI_API_KEY_SECONDARY),
# либо напрямую прописать, как вы просили.
OPENAI_API_KEY_SECONDARY = os.getenv("OPENAI_API_KEY_SECONDARY", "")

# Ключи Google Custom Search
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()  # вывод логов в консоль
    ]
)
logger = logging.getLogger("itmo-bot")


@app.on_event("startup")
def startup_event():
    logger.info("Application started")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    body = await request.body()

    try:
        body_text = body.decode("utf-8")
    except UnicodeDecodeError:
        body_text = "<Could not decode request body>"

    logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body_text}"
    )

    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Duration: {duration:.3f}s"
    )
    return response


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    """
    Обрабатывает запрос, упрощает его через "агента-упрощателя",
    потом отправляет упрощённый запрос в Google Search, а затем использует
    основную модель OpenAI для выдачи ответа.
    """
    try:
        logger.info(f"Processing prediction request with id: {body.id}")

        # 1) Упрощаем запрос для Google:
        simplified_query = simplify_query_with_openai(body.query)

        logger.info(f"GOOGLE_API_KEY: {GOOGLE_API_KEY}, GOOGLE_CSE_ID: {GOOGLE_CSE_ID}")

        # 2) Получаем ссылки из Google на основе упрощённого запроса
        found_links = do_web_search(simplified_query, max_links=3)

        if not found_links:
            logger.warning(
                f"No search results found for simplified query: {simplified_query}. Trying again with max_links=5..."
            )
            found_links = do_web_search(simplified_query, max_links=5)

        # 3) Генерируем ответ, используя основную модель (старый ключ или ваш текущий OPENAI_API_KEY)
        answer_text, chosen_answer = generate_answer_with_openai(body.query, found_links)

        # Если нет выбранного варианта, ставим -1
        if chosen_answer is None:
            chosen_answer = -1

        # Валидируем ссылки
        valid_sources = []
        for link in found_links:
            try:
                valid_sources.append(HttpUrl(link))
            except Exception as e:
                logger.debug(f"Skipping invalid link: {link} | error: {e}")

        # Формируем ответ
        response_data = PredictionResponse(
            id=body.id,
            answer=chosen_answer,
            reasoning=answer_text,
            sources=valid_sources
        )

        logger.info(f"Successfully processed request {body.id}")
        logger.info(f"Response data: {response_data.json()}")

        return response_data

    except ValueError as e:
        logger.error(f"Validation error for request {body.id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


def simplify_query_with_openai(original_query: str) -> str:
    logger.info(f"Using secondary OpenAI key to simplify query: {original_query}")

    client = openai.OpenAI(api_key=OPENAI_API_KEY_SECONDARY)

    # Смотрим, есть ли в тексте кириллические символы
    # (простейшая эвристика, либо можно langdetect использовать)
    is_russian = re.search(r"[а-яА-ЯёЁ]", original_query)

    # Если распознали русский, просим GPT упрощать на русском
    if is_russian:
        system_prompt = (
            "Ты упрощатель поисковых запросов на русском языке. "
            "Получая длинный вопрос, оставляй только ключевые слова, "
            "не переводя на другие языки. "
            "Выдай итоговый запрос в 3-7 словах на русском языке без лишних пояснений."
        )
    else:
        system_prompt = (
            "You are a query simplifier. "
            "Output only a short search query, in English, without extra explanations."
        )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Original user query:\n{original_query}\n\n"
                        "Return a short query suitable for Google."
                    )
                }
            ],
            max_tokens=50,
            temperature=0.3
        )

        simplified_query = response.choices[0].message.content.strip()
        logger.info(f"Simplified query: {simplified_query}")
        return simplified_query
    except Exception as e:
        logger.error(f"Error simplifying query with secondary key: {str(e)}")
        return original_query


def do_web_search(query: str, max_links: int = 3) -> List[str]:
    logger.info(f"🔍 Searching Google for: {query}")

    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("❌ Missing Google API credentials!")
        return []

    # Добавляем параметры для русского языка (если нужно)
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        # Указываем русский язык, если мы предполагаем, что запрос на русском:
        "lr": "lang_ru",  # поиск страниц на русском
        "hl": "ru",       # язык интерфейса
        "gl": "ru"        # Геолокация - Россия
    }

    url = "https://www.googleapis.com/customsearch/v1"

    logger.info(f"🔍 Google API Request params: {params}")

    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        logger.info(f"✅ Google API Response: {data}")

        links = [item["link"] for item in data.get("items", []) if "link" in item]
        # Ограничиваемся max_links
        links = links[:max_links]

        logger.info(f"✅ Found links: {links}")
        return links
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Google API Error: {str(e)}")
        return []



def generate_answer_with_openai(original_query: str, sources: List[str]) -> Tuple[str, int]:
    """
    Использует основной ключ OpenAI, чтобы сформировать итоговый ответ
    на исходный вопрос, учитывая найденные ссылки.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY_MAIN)

    system_prompt = (
        "You are an assistant that answers questions strictly based on the provided sources. "
        "Do not generate links yourself. If there are no sources, respond with 'No relevant information found'."
    )


    sources_context = "\n".join([f"- {link}" for link in sources])

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{original_query}\n\n"
                        f"Sources:\n{sources_context}\n\n"
                        "Answer with the choice number if applicable, reasoning, and sources."
                    )
                }
            ],
            max_tokens=400,
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()
        logger.info(f"OpenAI main model response:\n{content}")

    except Exception as e:
        logger.error(f"OpenAI error (main key): {str(e)}")
        return f"Error: {str(e)}", -1

    chosen_answer = None
    match = re.search(r"\b([1-9]|10)\b", content)
    if match:
        chosen_answer = int(match.group(1))

    return content, chosen_answer if chosen_answer is not None else -1
