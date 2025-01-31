import os
import logging
import asyncio
from typing import List, Dict, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional


from search.google_cse import do_google_cse_async
from search.tavily import do_tavily_search
from extract.extractor import fetch_and_extract_all
from gpt.llm import (
    finalize_answer_sync,
    extract_choice_number,
    generate_fallback_answer,
    parse_variants_in_question
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("itmo-bot")

app = FastAPI()
load_dotenv()

# == Модель запроса / ответа ==
class PredictionRequest(BaseModel):
    id: int
    query: str

class PredictionResponse(BaseModel):
    id: int
    answer: Optional[int] 
    reasoning: str
    sources: List[str]

# == Читаем ENV-переменные ==
GOOGLE_CSE_ID_WIDE = os.getenv("GOOGLE_CSE_ID", "")
GOOGLE_CSE_ID_THIN = os.getenv("GOOGLE_CSE_ID_THIN", "")

ENABLE_THIN_CSE = os.getenv("ENABLE_THIN_CSE", "true").lower() == "true"
ENABLE_TAVILY   = os.getenv("ENABLE_TAVILY",   "true").lower() == "true"
ENABLE_WIDE_CSE = os.getenv("ENABLE_WIDE_CSE", "true").lower() == "true"

# Кэши
cse_thin_cache: Dict[str, List[str]] = {}
tavily_cache: Dict[str, List[str]] = {}
google_wide_cache: Dict[str, List[str]] = {}

CHUNK_SIZE = 3000

@app.on_event("startup")
async def on_startup():
    logger.info("App startup...")

@app.post("/api/request", response_model=PredictionResponse)
async def predict(request_data: PredictionRequest):
    """
    Запускаем 4 независимые асинхронные задачи:
      1) run_thin_cse  (приоритет 1)   [опционально]
      2) run_tavily    (приоритет 2)   [опционально]
      3) run_wide_cse  (приоритет 3)   [опционально]
      4) run_fallback_gpt (приоритет 4, никогда не даёт -1)

    Дожидаемся завершения ВСЕХ.  
    Выбираем ответ по приоритету:
      - если шаг 1 дал answer != -1 → return
      - иначе, если шаг 2 != -1 → return
      - иначе, если шаг 3 != -1 → return
      - иначе → берём шаг 4.
    """
    question = request_data.query
    logger.info(f"Processing request {request_data.id} with question: {question}")

    # ПАРАЛЛЕЛЬНО запускаем все четыре задачи
    t1 = asyncio.create_task(run_thin_cse(question))
    t2 = asyncio.create_task(run_tavily(question))
    t3 = asyncio.create_task(run_wide_cse(question))
    t4 = asyncio.create_task(run_fallback_gpt(question))  # не возвращает -1

    results = await asyncio.gather(t1, t2, t3, t4)

    thin_res = results[0]    # (ans, reasoning, src)
    tavily_res = results[1]
    wide_res  = results[2]
    fallback_res = results[3]

    # Приоритет 1
    if thin_res[0] != -1:
        return PredictionResponse(
            id=request_data.id,
            answer=thin_res[0],
            reasoning=thin_res[1],
            sources=thin_res[2]
        )
    # Приоритет 2
    if tavily_res[0] != -1:
        return PredictionResponse(
            id=request_data.id,
            answer=tavily_res[0],
            reasoning=tavily_res[1],
            sources=tavily_res[2]
        )
    # Приоритет 3
    if wide_res[0] != -1:
        return PredictionResponse(
            id=request_data.id,
            answer=wide_res[0],
            reasoning=wide_res[1],
            sources=wide_res[2]
        )

    # Приоритет 4 (fallback GPT)
    return PredictionResponse(
        id=request_data.id,
        answer=fallback_res[0],
        reasoning=fallback_res[1],
        sources=fallback_res[2]
    )

# ============== Шаг 1: Thin CSE =================
async def run_thin_cse(question: str) -> Tuple[int, str, List[str]]:
    """
    Если включён:
      - Делает поиск тонкой CSE, извлекает факты, GPT => (answer, reasoning, sources)
    Иначе возвращаем -1 сразу.
    """
    if not ENABLE_THIN_CSE:
        return -1, "Thin CSE disabled", []

    # Проверяем, есть ли вообще cse_id
    if not GOOGLE_CSE_ID_THIN:
        return -1, "Thin CSE not configured", []

    links = await do_google_cse_async(
        query=question,
        cse_id=GOOGLE_CSE_ID_THIN,
        cache_dict=cse_thin_cache,
        max_links=3
    )
    if not links:
        return -1, "Thin CSE: no links", []

    facts, sources = await fetch_and_extract_all(links, question, chunk_size=CHUNK_SIZE)
    if not facts:
        return -1, "Thin CSE: no relevant facts", sources

    final_ans = finalize_answer_sync(facts, question)
    reasoning = f"{final_ans} (Ответ сгенерирован с использованием GPT-4)"
    return extract_choice_number(final_ans), reasoning, sources

# ============== Шаг 2: Tavily =================
async def run_tavily(question: str) -> Tuple[int, str, List[str]]:
    """
    Если включён:
      - Вызываем Tavily (синхронно, но через executor), потом GPT => (answer, ...)
    Иначе -1.
    """
    if not ENABLE_TAVILY:
        return -1, "Tavily disabled", []

    loop = asyncio.get_running_loop()
    links = await loop.run_in_executor(None, lambda: do_tavily_search(question, tavily_cache))
    if not links:
        return -1, "Tavily: no links", []

    facts, sources = await fetch_and_extract_all(links, question, chunk_size=CHUNK_SIZE)
    if not facts:
        return -1, "Tavily: no relevant facts", sources

    final_ans = finalize_answer_sync(facts, question)
    chosen = extract_choice_number(final_ans)
    return chosen, final_ans, sources

# ============== Шаг 3: Wide CSE =================
async def run_wide_cse(question: str) -> Tuple[int, str, List[str]]:
    """
    Если включён:
      - Поиск в широкой CSE, GPT => ...
    Иначе -1.
    """
    if not ENABLE_WIDE_CSE:
        return -1, "Wide CSE disabled", []

    if not GOOGLE_CSE_ID_WIDE:
        return -1, "Wide CSE not configured", []

    links = await do_google_cse_async(
        query=question,
        cse_id=GOOGLE_CSE_ID_WIDE,
        cache_dict=google_wide_cache,
        max_links=3
    )
    if not links:
        return -1, "Wide CSE: no links", []

    facts, sources = await fetch_and_extract_all(links, question, chunk_size=CHUNK_SIZE)
    if not facts:
        return -1, "Wide CSE: no relevant facts", sources

    final_ans = finalize_answer_sync(facts, question)
    chosen = extract_choice_number(final_ans)
    return chosen, final_ans, sources

# ============== Шаг 4: Fallback GPT =================
from typing import Optional

async def run_fallback_gpt(question: str) -> Tuple[Optional[int], str, List[str]]:
    """
    1. GPT генерирует ответ.
    2. Проверяем, есть ли в вопросе варианты ответа (parse_variants_in_question).
    3. Если есть варианты → пытаемся извлечь номер варианта (1..N).
    4. Если вариантов нет → answer = None (по ТЗ).
    """
    text = await asyncio.get_running_loop().run_in_executor(None, generate_fallback_answer, question)
    num_variants = parse_variants_in_question(question)

    if num_variants > 0:
        chosen = extract_choice_number(text)
        if 1 <= chosen <= num_variants:
            reasoning = f"{text} (Ответ сгенерирован с использованием GPT-4)"
            return chosen, reasoning, []
        else:
            reasoning = f"{text} (Ответ сгенерирован с использованием GPT-4)"
            return None, reasoning, []  # GPT не выбрал вариант -> answer = None
    else:
        reasoning = f"{text} (Ответ сгенерирован с использованием GPT-4)"
        return None, reasoning, []  # Нет вариантов -> answer = None
