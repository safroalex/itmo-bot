import logging
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger("itmo-bot")

HTTP_TIMEOUT = 2.0

def extract_text_from_html(html: str) -> str:
    """Простая вытяжка текста из HTML."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

async def fetch_and_chunk_all(links: List[str], chunk_size: int) -> List[str]:
    """Асинхронно скачиваем страницы, берём текст, обрезаем до chunk_size."""
    texts = []
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for link in links:
            try:
                resp = await client.get(link)
                resp.raise_for_status()
                text = extract_text_from_html(resp.text)
                texts.append(text[:chunk_size])
            except Exception as e:
                logger.warning(f"Ошибка при загрузке {link}: {e}")
    return texts

def extract_facts_from_text(text: str, question: str) -> Optional[str]:
    """
    Простейший «поисковый» подход:
    Если в тексте упоминается запрос (lowercase match), вернём первые 300 символов.
    """
    if question.lower() in text.lower():
        return text[:300]
    return None

async def extract_facts_for_all_chunks_async(
    chunks: List[str], 
    question: str
) -> List[str]:
    facts = []
    for chunk in chunks:
        extracted = extract_facts_from_text(chunk, question)
        if extracted:
            facts.append(extracted)
    return facts

async def fetch_and_extract_all(
    links: List[str],
    question: str,
    chunk_size: int = 3000
) -> Tuple[List[str], List[str]]:
    """
    1) Скачиваем страницы
    2) Разбиваем/обрезаем
    3) Ищем «факты»
    Возвращаем (список_фактов, исходные_ссылки).
    """
    chunked_texts = await fetch_and_chunk_all(links, chunk_size)
    facts_list = await extract_facts_for_all_chunks_async(chunked_texts, question)
    return facts_list, links