import requests
import logging
from typing import List, Dict

logger = logging.getLogger("itmo-bot")

TAVILY_API_KEY = "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"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

def do_tavily_search(
    query: str,
    tavily_cache: Dict[str, List[str]],
    max_links: int = 3
) -> List[str]:
    """
    Поиск через Tavily API (синхронно через requests).
    Возвращаем список ссылок или пустой список, если ошибка/нет данных.
    """

    # «Чистим» запрос
    clean_query = " ".join(query.strip().splitlines())

    key = clean_query.lower()
    if key in tavily_cache:
        return tavily_cache[key][:max_links]

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": clean_query
    }
    headers = {
        "Content-Type": "application/json"
    }

    results_links = []
    try:
        resp = requests.post(TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Предположим, что data имеет поле "results", где лежат ссылки
            # Нужно смотреть реальный формат ответа Tavily
            for item in data.get("results", []):
                if "url" in item:
                    results_links.append(item["url"])
            tavily_cache[key] = results_links
        else:
            logger.warning(f"Tavily error: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.warning(f"Tavily exception: {e}")

    return results_links[:max_links]