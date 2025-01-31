import os
import logging
import httpx
from typing import List, Dict
from dotenv import load_dotenv


load_dotenv()


logger = logging.getLogger("itmo-bot")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HTTP_TIMEOUT = 2.0

async def do_google_cse_async(
    query: str,
    cse_id: str,
    cache_dict: Dict[str, List[str]],
    max_links: int = 3
) -> List[str]:
    """
    Асинхронная функция для Google Custom Search.
    Приоритетно используем «тонкую» (thin) CSE или «широкую» (wide) CSE — 
    в зависимости от cse_id.
    """
    # Удаляем переносы строк и т.д., чтобы не ломать запрос
    clean_query = " ".join(query.strip().splitlines())

    # Проверка кэша
    key = (clean_query.lower(), cse_id)
    if key in cache_dict:
        logger.info(f"[Cache] Found google cse links for: {key}")
        return cache_dict[key][:max_links]

    if not GOOGLE_API_KEY or not cse_id:
        logger.warning("Missing Google credentials or cse_id!")
        return []

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": cse_id,
        "q": clean_query,
        "lr": "lang_ru",
        "hl": "ru",
        "gl": "ru",
    }

    links = []
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            for it in items[:max_links]:
                if "link" in it:
                    links.append(it["link"])
            cache_dict[key] = links
        except Exception as e:
            logger.warning(f"do_google_cse_async error: {e}")

    return links