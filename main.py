import os
import base64
import asyncio
import logging
import re
from html import unescape
from urllib.parse import unquote, urljoin

import httpx
import psycopg
import uvicorn

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ============================================================
# CONFIG
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
WEBHOOK_SECRET = os.getenv(
    "WEBHOOK_SECRET",
    "oxalpha_webhook_secret",
)

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

FREE_SERP_URL = (
    "https://freeserp.ai/api.php"
)

DDG_URL = (
    "https://html.duckduckgo.com/html/"
)

OPENROUTER_TEXT_MODEL = (
    "nvidia/nemotron-3-ultra-550b-a55b:free"
)

OPENROUTER_MULTIMODAL_MODEL = (
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
)

MAX_HISTORY = 30
MAX_WEB_RESULTS = 8
MAX_PAGE_TEXT = 7000
MAX_WEB_CONTEXT_CHARS = 30000

WEBHOOK_URL = os.getenv(
    "WEBHOOK_URL",
    (
        "https://bot-ai-oxalpha.onrender.com/"
        "telegram/oxalpha_webhook_7f92kLm31Qx8"
    ),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("oxalpha")


# ============================================================
# ERRORS
# ============================================================

class WebSearchError(RuntimeError):
    pass


class OpenRouterError(RuntimeError):
    pass


# ============================================================
# DATABASE
# ============================================================

def db_connect():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured"
        )

    return psycopg.connect(DATABASE_URL)


def init_db():
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    persona TEXT DEFAULT '',
                    style TEXT DEFAULT 'Clean',
                    memory TEXT DEFAULT ''
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )

        conn.commit()

    logger.info("Database initialized")


def ensure_user(user_id: int):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id)
                VALUES (%s)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (user_id,),
            )

        conn.commit()


def get_user_settings(user_id: int):
    ensure_user(user_id)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT persona, style, memory
                FROM users
                WHERE user_id = %s
                """,
                (user_id,),
            )

            row = cur.fetchone()

    if not row:
        return "", "Clean", ""

    persona, style, memory = row

    return (
        persona or "",
        style or "Clean",
        memory or "",
    )


def set_persona(
    user_id: int,
    persona: str,
):
    ensure_user(user_id)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET persona = %s
                WHERE user_id = %s
                """,
                (
                    persona,
                    user_id,
                ),
            )

        conn.commit()


def set_style(
    user_id: int,
    style: str,
):
    ensure_user(user_id)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET style = %s
                WHERE user_id = %s
                """,
                (
                    style,
                    user_id,
                ),
            )

        conn.commit()


def set_memory(
    user_id: int,
    memory: str,
):
    ensure_user(user_id)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET memory = %s
                WHERE user_id = %s
                """,
                (
                    memory,
                    user_id,
                ),
            )

        conn.commit()


def add_message(
    user_id: int,
    role: str,
    content: str,
):
    ensure_user(user_id)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages
                    (user_id, role, content)
                VALUES
                    (%s, %s, %s)
                """,
                (
                    user_id,
                    role,
                    content,
                ),
            )

            cur.execute(
                """
                DELETE FROM messages
                WHERE user_id = %s
                  AND id NOT IN (
                      SELECT id
                      FROM messages
                      WHERE user_id = %s
                      ORDER BY id DESC
                      LIMIT %s
                  )
                """,
                (
                    user_id,
                    user_id,
                    MAX_HISTORY,
                ),
            )

        conn.commit()


def get_history(user_id: int):
    ensure_user(user_id)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM messages
                WHERE user_id = %s
                ORDER BY id ASC
                LIMIT %s
                """,
                (
                    user_id,
                    MAX_HISTORY,
                ),
            )

            rows = cur.fetchall()

    return [
        {
            "role": role,
            "content": content,
        }
        for role, content in rows
    ]


def clear_history(user_id: int):
    ensure_user(user_id)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM messages
                WHERE user_id = %s
                """,
                (user_id,),
            )

        conn.commit()


# ============================================================
# OPENROUTER
# ============================================================

def extract_openrouter_text(data):
    if not isinstance(data, dict):
        raise OpenRouterError(
            "OpenRouter returned non-object JSON"
        )

    choices = data.get("choices")

    if not isinstance(choices, list) or not choices:
        error = data.get("error")

        if isinstance(error, dict):
            message = (
                error.get("message")
                or error.get("code")
                or str(error)
            )

            raise OpenRouterError(
                f"OpenRouter error: {message}"
            )

        raise OpenRouterError(
            "OpenRouter response does not contain choices"
        )

    choice = choices[0]

    if not isinstance(choice, dict):
        raise OpenRouterError(
            "Invalid OpenRouter choice"
        )

    message = choice.get("message")

    if not isinstance(message, dict):
        raise OpenRouterError(
            "OpenRouter response does not contain message"
        )

    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if not isinstance(item, dict):
                continue

            value = item.get("text")

            if (
                isinstance(value, str)
                and value.strip()
            ):
                parts.append(value.strip())

        if parts:
            return "\n".join(parts).strip()

    output_text = message.get("output_text")

    if (
        isinstance(output_text, str)
        and output_text.strip()
    ):
        return output_text.strip()

    raise OpenRouterError(
        "OpenRouter returned no usable assistant text"
    )


async def openrouter_request(
    messages,
    model=None,
    temperature=0.7,
):
    if not OPENROUTER_API_KEY:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not configured"
        )

    payload = {
        "model": (
            model
            or OPENROUTER_TEXT_MODEL
        ),
        "messages": messages,
        "temperature": temperature,
    }

    headers = {
        "Authorization": (
            f"Bearer {OPENROUTER_API_KEY}"
        ),
        "Content-Type": "application/json",
        "HTTP-Referer": (
            "https://bot-ai-oxalpha.onrender.com"
        ),
        "X-Title": "Ox Alpha AI",
    }

    timeout = httpx.Timeout(
        120.0,
        connect=20.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
    ) as client:

        response = await client.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
        )

    if response.is_error:
        logger.error(
            "OpenRouter HTTP %s: %s",
            response.status_code,
            response.text[:3000],
        )

        raise OpenRouterError(
            f"OpenRouter HTTP {response.status_code}"
        )

    try:
        return response.json()

    except Exception as exc:
        raise OpenRouterError(
            "OpenRouter returned invalid JSON"
        ) from exc


# ============================================================
# PROMPTS
# ============================================================

STYLE_INSTRUCTIONS = {
    "Clean": (
        "Отвечай естественно, понятно и без "
        "лишнего форматирования."
    ),
    "Minimal": (
        "Отвечай очень коротко и по делу."
    ),
    "Terminal": (
        "Пиши в сухом техническом стиле, "
        "как консольный помощник."
    ),
    "Meme": (
        "Можно использовать лёгкий мемный стиль "
        "и уместные эмодзи, но сохраняй полезность."
    ),
    "Detailed": (
        "Отвечай подробно, структурированно "
        "и с объяснениями."
    ),
}


def build_system_prompt(
    persona,
    style,
    memory,
    web_mode=False,
):
    style_instruction = STYLE_INSTRUCTIONS.get(
        style,
        STYLE_INSTRUCTIONS["Clean"],
    )

    prompt = f"""
Ты — Ox Alpha AI, Telegram-бот пользователя.

Стиль ответа:
{style_instruction}

Персона:
{persona if persona else "Не задана."}

Долгосрочная память:
{memory if memory else "Пусто."}

Правила:
- Не выдумывай факты.
- Если не знаешь — честно скажи.
- Не утверждай, что проверил интернет,
  если веб-данные тебе не передали.
- Не называй устаревшую информацию текущей.
"""

    if web_mode:
        prompt += """
Сейчас используется веб-поиск.

Тебе переданы результаты поиска и,
если удалось, содержимое найденных страниц.

Используй эти материалы как основной источник
для актуальной информации.

Учитывай даты публикаций.

Если источники противоречат друг другу,
укажи это.

Не придумывай сведения, которых нет
в предоставленных источниках.

Если информации недостаточно,
честно скажи об этом.

Если уместно, указывай ссылки на источники.
"""

    return prompt.strip()


async def ask_text(
    user_id: int,
    text: str,
):
    persona, style, memory = await asyncio.to_thread(
        get_user_settings,
        user_id,
    )

    history = await asyncio.to_thread(
        get_history,
        user_id,
    )

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                persona,
                style,
                memory,
            ),
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": text,
        }
    )

    data = await openrouter_request(
        messages,
        model=OPENROUTER_TEXT_MODEL,
        temperature=0.7,
    )

    return extract_openrouter_text(data)


# ============================================================
# SEARCH QUERY
# ============================================================

def clean_search_query(query: str):
    if not query:
        return ""

    query = str(query)

    query = re.sub(
        r"```.*?```",
        " ",
        query,
        flags=re.S,
    )

    query = re.sub(
        r"`([^`]*)`",
        r"\1",
        query,
    )

    prefixes = [
        "поисковый запрос:",
        "search query:",
        "search:",
        "query:",
        "запрос:",
    ]

    lowered = query.lower().strip()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            query = query[
                len(prefix):
            ].strip()
            break

    query = query.strip().strip("\"'")

    query = re.sub(
        r"\s+",
        " ",
        query,
    )

    return query[:500].strip()


async def make_search_query(text: str):
    prompt = f"""
Сделай из сообщения пользователя ОДИН короткий
поисковый запрос для интернет-поисковика.

Верни только поисковый запрос.
Без объяснений.
Без markdown.
Без кавычек.

Сообщение:
{text[:2000]}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Ты генератор поисковых запросов. "
                "Возвращай только запрос."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    data = await openrouter_request(
        messages,
        model=OPENROUTER_TEXT_MODEL,
        temperature=0.2,
    )

    return clean_search_query(
        extract_openrouter_text(data)
    )


# ============================================================
# FREE SERP
# ============================================================

def normalize_search_result(
    item,
    source="Web",
):
    if not isinstance(item, dict):
        return None

    title = (
        item.get("title")
        or item.get("name")
        or item.get("heading")
        or ""
    )

    snippet = (
        item.get("snippet")
        or item.get("description")
        or item.get("content")
        or item.get("text")
        or ""
    )

    link = (
        item.get("link")
        or item.get("url")
        or item.get("href")
        or ""
    )

    date = (
        item.get("date")
        or item.get("published")
        or item.get("published_at")
        or item.get("datePublished")
        or ""
    )

    title = unescape(
        str(title)
    ).strip()

    snippet = unescape(
        str(snippet)
    ).strip()

    link = str(link).strip()
    date = str(date).strip()

    if not title and not snippet:
        return None

    return {
        "title": title[:500],
        "snippet": snippet[:2500],
        "link": link[:2000],
        "date": date[:200],
        "source": source,
    }


def extract_freeserp_items(data):
    if not isinstance(data, dict):
        return []

    items = []

    for key in (
        "results",
        "organic_results",
        "items",
    ):
        value = data.get(key)

        if isinstance(value, list):
            items.extend(value)

    web = data.get("web")

    if isinstance(web, dict):
        for key in (
            "results",
            "items",
        ):
            value = web.get(key)

            if isinstance(value, list):
                items.extend(value)

    return items


async def freeserp_search(query: str):
    query = clean_search_query(query)

    if not query:
        raise WebSearchError(
            "Empty FreeSerp query"
        )

    params = {
        "q": query,
        "output": "json",
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/140 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    timeout = httpx.Timeout(
        20.0,
        connect=10.0,
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:

            response = await client.get(
                FREE_SERP_URL,
                params=params,
                headers=headers,
            )

    except Exception as exc:
        logger.warning(
            "FreeSerp request failed: %s",
            exc,
        )

        raise WebSearchError(
            "FreeSerp request failed"
        ) from exc

    if response.status_code != 200:
        raise WebSearchError(
            f"FreeSerp HTTP {response.status_code}"
        )

    try:
        data = response.json()

    except Exception as exc:
        raise WebSearchError(
            "FreeSerp returned invalid JSON"
        ) from exc

    raw_items = extract_freeserp_items(data)

    results = []

    for item in raw_items:
        result = normalize_search_result(
            item,
            source="FreeSerp",
        )

        if result:
            results.append(result)

    unique = []
    seen = set()

    for result in results:
        key = (
            result["link"]
            or result["title"]
            or result["snippet"]
        ).lower()

        if key in seen:
            continue

        seen.add(key)
        unique.append(result)

    if not unique:
        raise WebSearchError(
            "FreeSerp returned zero usable results"
        )

    logger.info(
        "FreeSerp: %d results",
        len(unique),
    )

    return unique[:MAX_WEB_RESULTS]


# ============================================================
# DUCKDUCKGO
# ============================================================

def extract_ddg_url(href: str):
    if not href:
        return ""

    href = unescape(href)

    match = re.search(
        r"[?&]uddg=([^&]+)",
        href,
        flags=re.I,
    )

    if match:
        return unquote(
            match.group(1)
        )

    return href


def parse_duckduckgo_html(html: str):
    results = []

    title_pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"'
        r'[^>]*>(.*?)</a>',
        flags=re.I | re.S,
    )

    href_pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"'
        r'[^>]+href="([^"]+)"',
        flags=re.I | re.S,
    )

    snippet_pattern = re.compile(
        r'<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"'
        r'[^>]*>(.*?)</(?:a|div)>',
        flags=re.I | re.S,
    )

    titles = title_pattern.findall(html)
    hrefs = href_pattern.findall(html)
    snippets = snippet_pattern.findall(html)

    count = min(
        len(titles),
        len(hrefs),
    )

    for i in range(count):
        title = re.sub(
            r"<[^>]+>",
            " ",
            titles[i],
        )

        title = unescape(title)

        title = re.sub(
            r"\s+",
            " ",
            title,
        ).strip()

        href = extract_ddg_url(
            hrefs[i]
        )

        snippet = ""

        if i < len(snippets):
            snippet = re.sub(
                r"<[^>]+>",
                " ",
                snippets[i],
            )

            snippet = unescape(snippet)

            snippet = re.sub(
                r"\s+",
                " ",
                snippet,
            ).strip()

        if not title and not href:
            continue

        results.append(
            {
                "title": title[:500],
                "snippet": snippet[:2500],
                "link": href[:2000],
                "date": "",
                "source": "DuckDuckGo",
            }
        )

    return results


async def duckduckgo_search(query: str):
    query = clean_search_query(query)

    if not query:
        raise WebSearchError(
            "Empty DuckDuckGo query"
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": (
            "ru-RU,ru;q=0.9,en;q=0.8"
        ),
    }

    params = {
        "q": query,
        "kl": "wt-wt",
    }

    timeout = httpx.Timeout(
        20.0,
        connect=10.0,
    )

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:

            response = await client.get(
                DDG_URL,
                params=params,
                headers=headers,
            )

    except Exception as exc:
        raise WebSearchError(
            "DuckDuckGo request failed"
        ) from exc

    if response.status_code != 200:
        raise WebSearchError(
            f"DuckDuckGo HTTP {response.status_code}"
        )

    results = parse_duckduckgo_html(
        response.text
    )

    unique = []
    seen = set()

    for result in results:
        key = (
            result["link"]
            or result["title"]
            or result["snippet"]
        ).lower()

        if key in seen:
            continue

        seen.add(key)
        unique.append(result)

    if not unique:
        raise WebSearchError(
            "DuckDuckGo returned zero usable results"
        )

    logger.info(
        "DuckDuckGo: %d results",
        len(unique),
    )

    return unique[:MAX_WEB_RESULTS]


# ============================================================
# OPEN FOUND PAGES
# ============================================================

def clean_html_to_text(html: str):
    """
    Простой HTML -> текст.

    Не является полноценным browser parser,
    но позволяет модели читать обычные статьи,
    документацию и страницы.
    """

    if not html:
        return ""

    # Удаляем script/style/noscript
    html = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        html,
        flags=re.I | re.S,
    )

    html = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        html,
        flags=re.I | re.S,
    )

    html = re.sub(
        r"<noscript\b[^>]*>.*?</noscript>",
        " ",
        html,
        flags=re.I | re.S,
    )

    # Переводы строк для структурированных блоков
    html = re.sub(
        r"</(?:p|div|article|section|h1|h2|h3|h4|li|br)>",
        "\n",
        html,
        flags=re.I,
    )

    # Убираем остальные теги
    text = re.sub(
        r"<[^>]+>",
        " ",
        html,
    )

    text = unescape(text)

    # Убираем лишние пробелы
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )

    return text.strip()


async def fetch_page(
    client: httpx.AsyncClient,
    result: dict,
):
    url = result.get("link", "")

    if not url:
        return result

    # Не пытаемся читать потенциально странные схемы.
    if not url.startswith(
        ("http://", "https://")
    ):
        return result

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": (
            "ru-RU,ru;q=0.9,en;q=0.8"
        ),
    }

    try:
        response = await client.get(
            url,
            headers=headers,
        )

        if response.status_code != 200:
            result["page_text"] = ""
            return result

        content_type = (
            response.headers.get(
                "content-type",
                "",
            ).lower()
        )

        # Не пытаемся скармливать модели PDF,
        # картинки, видео и архивы как HTML.
        if (
            "text/html" not in content_type
            and "application/xhtml+xml"
            not in content_type
        ):
            result["page_text"] = ""
            return result

        text = clean_html_to_text(
            response.text
        )

        # Иногда сайт возвращает гигантский документ.
        result["page_text"] = text[
            :MAX_PAGE_TEXT
        ]

        return result

    except Exception as exc:
        logger.debug(
            "Page fetch failed %s: %s",
            url,
            exc,
        )

        result["page_text"] = ""

        return result


async def open_search_pages(
    results,
):
    """
    Открываем несколько лучших результатов
    параллельно.

    Если отдельная страница не открылась,
    сам результат поиска всё равно сохраняется.
    """

    if not results:
        return results

    timeout = httpx.Timeout(
        15.0,
        connect=8.0,
    )

    limits = httpx.Limits(
        max_connections=5,
        max_keepalive_connections=5,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
    ) as client:

        tasks = [
            fetch_page(
                client,
                result,
            )
            for result in results[:5]
        ]

        fetched = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

    output = []

    for original, item in zip(
        results,
        fetched,
    ):
        if isinstance(item, Exception):
            original["page_text"] = ""
            output.append(original)
        else:
            output.append(item)

    return output


# ============================================================
# FORMAT WEB CONTEXT
# ============================================================

def format_web_results(results):
    if not results:
        return ""

    blocks = []

    for index, result in enumerate(
        results[:MAX_WEB_RESULTS],
        start=1,
    ):
        title = (
            result.get("title")
            or "Без названия"
        )

        snippet = (
            result.get("snippet")
            or "Нет описания."
        )

        link = result.get("link") or ""
        date = result.get("date") or ""
        source = (
            result.get("source")
            or "Web"
        )

        page_text = (
            result.get("page_text")
            or ""
        )

        block = (
            f"[ИСТОЧНИК {index}]\n"
            f"Название: {title}\n"
            f"Источник поиска: {source}\n"
        )

        if date:
            block += f"Дата: {date}\n"

        if link:
            block += f"URL: {link}\n"

        block += (
            f"Сниппет: {snippet}\n"
        )

        if page_text:
            block += (
                "\nТЕКСТ СТРАНИЦЫ:\n"
                f"{page_text}"
            )

        blocks.append(block)

    context = "\n\n".join(
        blocks
    )

    return context[
        :MAX_WEB_CONTEXT_CHARS
    ]


# ============================================================
# MAIN WEB SEARCH
# ============================================================

async def web_search(query: str):
    query = clean_search_query(query)

    if not query:
        raise WebSearchError(
            "Search query is empty"
        )

    results = None

    # --------------------------------------------------------
    # FreeSerp
    # --------------------------------------------------------

    try:
        results = await freeserp_search(
            query
        )

    except Exception as exc:
        logger.warning(
            "FreeSerp failed: %s",
            exc,
        )

    # --------------------------------------------------------
    # DuckDuckGo fallback
    # --------------------------------------------------------

    if not results:

        try:
            results = await duckduckgo_search(
                query
            )

        except Exception as exc:
            logger.warning(
                "DuckDuckGo failed: %s",
                exc,
            )

    if not results:
        raise WebSearchError(
            "All web search providers failed"
        )

    # --------------------------------------------------------
    # Open pages
    # --------------------------------------------------------

    try:
        results = await open_search_pages(
            results
        )

    except Exception as exc:
        logger.warning(
            "Opening search pages failed: %s",
            exc,
        )

    return results


# ============================================================
# WEB ANSWER
# ============================================================

async def ask_text_with_web(
    user_id: int,
    original_text: str,
    search_query: str,
    web_results,
):
    persona, style, memory = await asyncio.to_thread(
        get_user_settings,
        user_id,
    )

    history = await asyncio.to_thread(
        get_history,
        user_id,
    )

    web_context = format_web_results(
        web_results
    )

    current_date = "2026-09-22"

    prompt = f"""
Сегодня: {current_date}

Поисковый запрос:
{search_query}

Оригинальный вопрос пользователя:
{original_text}

Найденные интернет-источники:

{web_context}

Теперь ответь пользователю.

Требования:

1. Для актуальных фактов используй найденные источники.
2. Если страница содержит более подробную информацию,
   используй её текст, а не только сниппет.
3. Сравни несколько источников, когда это возможно.
4. Учитывай даты публикаций.
5. Не выдавай свои старые знания за текущие данные.
6. Не придумывай отсутствующие сведения.
7. Если источники противоречат друг другу —
   скажи об этом.
8. Если информации недостаточно —
   прямо скажи об этом.
9. Когда уместно, добавляй ссылки на источники.
10. Не утверждай, что источник сказал то,
    чего в переданных материалах нет.

Ответ должен быть нормальным ответом человеку,
а не отчётом о работе поискового механизма.
"""

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                persona,
                style,
                memory,
                web_mode=True,
            ),
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    data = await openrouter_request(
        messages,
        model=OPENROUTER_TEXT_MODEL,
        temperature=0.5,
    )

    return extract_openrouter_text(
        data
    )


# ============================================================
# SEARCH DETECTION
# ============================================================

WEB_SEARCH_TRIGGERS = [
    "посмотри в интернете",
    "посмотри в инете",
    "посмотри интернет",
    "поищи в интернете",
    "поищи в инете",
    "поищи",
    "найди в интернете",
    "найди в инете",
    "найди",
    "проверь в интернете",
    "проверь в инете",
    "проверь",
    "загугли",
    "гугли",
    "погугли",
    "поиск в интернете",
    "поиск в инете",
    "сделай поиск",
    "последние новости",
    "свежие новости",
    "актуальная информация",
    "актуальные данные",
    "на сегодня",
    "на данный момент",
    "сейчас лучший",
    "сейчас топовый",
    "кто сейчас лучший",
    "что сейчас лучше",
    "latest",
    "today",
    "current",
    "up to date",
    "search the web",
    "search online",
    "look it up",
]


def wants_web_search(text: str):
    if not text:
        return False

    lowered = text.lower().strip()

    return any(
        trigger in lowered
        for trigger in WEB_SEARCH_TRIGGERS
    )


# ============================================================
# TELEGRAM HELPERS
# ============================================================

async def send_action(
    update: Update,
    action="typing",
):
    try:
        if update.effective_chat:
            await update.effective_chat.send_action(
                action=action
            )
    except Exception:
        pass


async def reply_text(
    update: Update,
    text: str,
):
    if not update.message:
        return

    text = str(text or "").strip()

    if not text:
        text = "Пустой ответ от модели 🤔"

    max_length = 4000

    for i in range(
        0,
        len(text),
        max_length,
    ):
        await update.message.reply_text(
            text[i:i + max_length],
            disable_web_page_preview=True,
        )


# ============================================================
# COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_user:
        return

    user_id = update.effective_user.id

    await asyncio.to_thread(
        ensure_user,
        user_id,
    )

    await reply_text(
        update,
        """
🤖 Ox Alpha AI запущен.

Я умею:
• общаться с тобой
• искать актуальную информацию
• читать найденные веб-страницы
• запоминать полезную информацию
• работать с персонажем
• менять стиль
• анализировать фото
• анализировать видео

Команды:

/reset — очистить историю
/forget — очистить память
/persona — настроить персонажа
/style — выбрать стиль

Просто пиши сообщение 👽
""".strip(),
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_user:
        return

    await asyncio.to_thread(
        clear_history,
        update.effective_user.id,
    )

    await reply_text(
        update,
        "История чата очищена 🧹",
    )


async def forget_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_user:
        return

    await asyncio.to_thread(
        set_memory,
        update.effective_user.id,
        "",
    )

    await reply_text(
        update,
        "Долгосрочная память очищена 🧠🗑️",
    )


async def persona_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.effective_user:
        return

    user_id = update.effective_user.id

    args = context.args or []

    if not args:
        persona, _, _ = await asyncio.to_thread(
            get_user_settings,
            user_id,
        )

        if persona:
            await reply_text(
                update,
                "Текущий персонаж:\n\n"
                + persona,
            )
        else:
            await reply_text(
                update,
                (
                    "Персонаж не задан.\n\n"
                    "Например:\n"
                    "/persona Ты саркастичный "
                    "техно-бот"
                ),
            )

        return

    persona = " ".join(
        args
    ).strip()

    persona = persona[:2000]

    await asyncio.to_thread(
        set_persona,
        user_id,
        persona,
    )

    await reply_text(
        update,
        "Персонаж сохранён 🎭\n\n"
        + persona,
    )


async def style_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.message:
        return

    keyboard = [
        [
            InlineKeyboardButton(
                "🧼 Clean",
                callback_data="style:Clean",
            ),
            InlineKeyboardButton(
                "⚡ Minimal",
                callback_data="style:Minimal",
            ),
        ],
        [
            InlineKeyboardButton(
                "💻 Terminal",
                callback_data="style:Terminal",
            ),
            InlineKeyboardButton(
                "😂 Meme",
                callback_data="style:Meme",
            ),
        ],
        [
            InlineKeyboardButton(
                "📚 Detailed",
                callback_data="style:Detailed",
            ),
        ],
    ]

    await update.message.reply_text(
        "Выбери стиль ответа:",
        reply_markup=InlineKeyboardMarkup(
            keyboard
        ),
    )


async def style_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    if not query:
        return

    await query.answer()

    data = query.data or ""

    if not data.startswith("style:"):
        return

    style = data.split(
        ":",
        1,
    )[1]

    if style not in STYLE_INSTRUCTIONS:
        return

    if not update.effective_user:
        return

    await asyncio.to_thread(
        set_style,
        update.effective_user.id,
        style,
    )

    try:
        await query.edit_message_text(
            f"Стиль изменён на {style} ✅"
        )
    except Exception:
        pass


# ============================================================
# MEMORY
# ============================================================

async def maybe_update_memory(
    user_id: int,
    user_text: str,
):
    lowered = user_text.lower()

    triggers = [
        "запомни",
        "запоминай",
        "помни",
        "не забывай",
        "мне нравится",
        "я люблю",
        "я предпочитаю",
        "мой проект",
        "мой бот",
        "моя игра",
        "у меня есть",
    ]

    if not any(
        trigger in lowered
        for trigger in triggers
    ):
        return

    try:
        _, _, current_memory = await asyncio.to_thread(
            get_user_settings,
            user_id,
        )

        prompt = f"""
Текущая долгосрочная память:
{current_memory or "Пусто"}

Новое сообщение пользователя:
{user_text[:2000]}

Определи, есть ли здесь полезная информация
для будущих разговоров.

Если нет — ответь строго:
NO

Если есть — верни обновлённую память
коротким списком фактов.

Не сохраняй:
- пароли;
- API-ключи;
- токены;
- точные адреса;
- лишние личные сведения;
- случайные эмоции;
- временные события.

Максимум 20 пунктов.
"""

        data = await openrouter_request(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты аккуратный менеджер "
                        "долгосрочной памяти."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model=OPENROUTER_TEXT_MODEL,
            temperature=0.2,
        )

        result = extract_openrouter_text(
            data
        ).strip()

        if result.upper() == "NO":
            return

        await asyncio.to_thread(
            set_memory,
            user_id,
            result[:5000],
        )

    except Exception as exc:
        logger.warning(
            "Memory update failed: %s",
            exc,
        )


# ============================================================
# TEXT MESSAGE
# ============================================================

async def message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.message:
        return

    if not update.effective_user:
        return

    text = update.message.text

    if not text:
        return

    text = text.strip()

    if not text:
        return

    user_id = update.effective_user.id

    await asyncio.to_thread(
        ensure_user,
        user_id,
    )

    await send_action(
        update,
        "typing",
    )

    try:

        # ====================================================
        # WEB MODE
        # ====================================================

        if wants_web_search(text):

            logger.info(
                "Web request from %s: %s",
                user_id,
                text[:300],
            )

            try:
                search_query = await make_search_query(
                    text
                )

            except Exception as exc:
                logger.warning(
                    "Could not generate search query: %s",
                    exc,
                )

                search_query = clean_search_query(
                    text
                )

            if not search_query:
                search_query = text[:500]

            try:
                web_results = await web_search(
                    search_query
                )

            except WebSearchError as exc:

                logger.error(
                    "Web search failed: %s",
                    exc,
                )

                await reply_text(
                    update,
                    (
                        "Брат, поиск сейчас полностью "
                        "отвалился 😵‍💫\n\n"
                        "Я специально не буду выдавать "
                        "старую информацию из памяти "
                        "за свежую.\n\n"
                        "Попробуй ещё раз чуть позже."
                    ),
                )

                return

            answer = await ask_text_with_web(
                user_id=user_id,
                original_text=text,
                search_query=search_query,
                web_results=web_results,
            )

        # ====================================================
        # NORMAL MODE
        # ====================================================

        else:

            answer = await ask_text(
                user_id=user_id,
                text=text,
            )

        if not answer:
            answer = (
                "Модель вернула пустой ответ 🤔"
            )

        await asyncio.to_thread(
            add_message,
            user_id,
            "user",
            text,
        )

        await asyncio.to_thread(
            add_message,
            user_id,
            "assistant",
            answer,
        )

        await reply_text(
            update,
            answer,
        )

        asyncio.create_task(
            maybe_update_memory(
                user_id,
                text,
            )
        )

    except OpenRouterError as exc:

        logger.error(
            "OpenRouter error: %s",
            exc,
            exc_info=True,
        )

        await reply_text(
            update,
            (
                "Ошибка при обращении к модели 😵\n\n"
                f"{exc}"
            ),
        )

    except Exception as exc:

        logger.error(
            "Message handler error: %s",
            exc,
            exc_info=True,
        )

        await reply_text(
            update,
            (
                "Брат, что-то сломалось внутри 😵‍💫\n\n"
                f"Ошибка: {type(exc).__name__}\n"
                f"{str(exc)[:500]}"
            ),
        )


# ============================================================
# PHOTO
# ============================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.message:
        return

    if not update.effective_user:
        return

    if not update.message.photo:
        return

    user_id = update.effective_user.id

    await send_action(
        update,
        "typing",
    )

    try:
        photo = update.message.photo[-1]

        telegram_file = await context.bot.get_file(
            photo.file_id
        )

        image_bytes = (
            await telegram_file.download_as_bytearray()
        )

        image_b64 = base64.b64encode(
            bytes(image_bytes)
        ).decode("utf-8")

        caption = (
            update.message.caption
            or "Проанализируй это изображение."
        )

        persona, style, memory = await asyncio.to_thread(
            get_user_settings,
            user_id,
        )

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    persona,
                    style,
                    memory,
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": caption,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + image_b64
                            ),
                        },
                    },
                ],
            },
        ]

        data = await openrouter_request(
            messages,
            model=OPENROUTER_MULTIMODAL_MODEL,
            temperature=0.5,
        )

        answer = extract_openrouter_text(
            data
        )

        await asyncio.to_thread(
            add_message,
            user_id,
            "user",
            "[Фото] " + caption,
        )

        await asyncio.to_thread(
            add_message,
            user_id,
            "assistant",
            answer,
        )

        await reply_text(
            update,
            answer,
        )

    except Exception as exc:

        logger.error(
            "Photo handler error: %s",
            exc,
            exc_info=True,
        )

        await reply_text(
            update,
            (
                "Не смог обработать фото 😵\n\n"
                f"{type(exc).__name__}: "
                f"{str(exc)[:500]}"
            ),
        )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

async def handle_video(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not update.message:
        return

    if not update.effective_user:
        return

    if not update.message.video:
        return

    user_id = update.effective_user.id

    await send_action(
        update,
        "typing",
    )

    try:
        video = update.message.video

        # Telegram Bot API обычно позволяет
        # скачивать файлы до 20 MB.
        if video.file_size:
            if video.file_size > (
                19.5 * 1024 * 1024
            ):
                await reply_text(
                    update,
                    (
                        "Брат, видео слишком большое 😵\n\n"
                        "Сейчас обработчик рассчитан "
                        "примерно на видео до 19.5 МБ."
                    ),
                )
                return

        telegram_file = await context.bot.get_file(
            video.file_id
        )

        video_bytes = (
            await telegram_file.download_as_bytearray()
        )

        video_b64 = base64.b64encode(
            bytes(video_bytes)
        ).decode("utf-8")

        caption = (
            update.message.caption
            or "Проанализируй это видео."
        )

        persona, style, memory = await asyncio.to_thread(
            get_user_settings,
            user_id,
        )

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    persona,
                    style,
                    memory,
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": caption,
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": (
                                "data:video/mp4;base64,"
                                + video_b64
                            ),
                        },
                    },
                ],
            },
        ]

        data = await openrouter_request(
            messages,
            model=OPENROUTER_MULTIMODAL_MODEL,
            temperature=0.5,
        )

        answer = extract_openrouter_text(
            data
        )

        await asyncio.to_thread(
            add_message,
            user_id,
            "user",
            "[Видео] " + caption,
        )

        await asyncio.to_thread(
            add_message,
            user_id,
            "assistant",
            answer,
        )

        await reply_text(
            update,
            answer,
        )

    except Exception as exc:

        logger.error(
            "Video handler error: %s",
            exc,
            exc_info=True,
        )

        await reply_text(
            update,
            (
                "Ошибка обработки видео 😵\n\n"
                f"{type(exc).__name__}: "
                f"{str(exc)[:500]}"
            ),
        )


# ============================================================
# TELEGRAM ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    error = context.error

    logger.error(
        "Unhandled Telegram error: %s",
        error,
        exc_info=True,
    )

    if not isinstance(update, Update):
        return

    try:
        if update.effective_message:
            await update.effective_message.reply_text(
                "Брат, произошла внутренняя ошибка 😵‍💫"
            )
    except Exception:
        pass


# ============================================================
# TELEGRAM WEBHOOK
# ============================================================

application = None


async def telegram_webhook(
    request: Request,
):
    global application

    incoming_secret = request.headers.get(
        "X-Telegram-Bot-Api-Secret-Token",
        "",
    )

    if WEBHOOK_SECRET:
        if incoming_secret != WEBHOOK_SECRET:
            logger.warning(
                "Invalid Telegram webhook secret"
            )

            return PlainTextResponse(
                "Forbidden",
                status_code=403,
            )

    try:
        payload = await request.json()

    except Exception:
        return PlainTextResponse(
            "Bad Request",
            status_code=400,
        )

    if application is None:
        return PlainTextResponse(
            "Service Unavailable",
            status_code=503,
        )

    try:
        update = Update.de_json(
            payload,
            application.bot,
        )

        await application.process_update(
            update
        )

    except Exception as exc:

        logger.error(
            "Telegram update error: %s",
            exc,
            exc_info=True,
        )

        return PlainTextResponse(
            "Internal Server Error",
            status_code=500,
        )

    return PlainTextResponse(
        "OK",
        status_code=200,
    )


# ============================================================
# HEALTH
# ============================================================

async def health_check(
    request: Request,
):
    return PlainTextResponse(
        "Ox Alpha AI is alive 🤖",
        status_code=200,
    )


# ============================================================
# STARLETTE
# ============================================================

routes = [
    Route(
        "/",
        health_check,
        methods=["GET"],
    ),
    Route(
        "/health",
        health_check,
        methods=["GET"],
    ),
    Route(
        "/telegram/oxalpha_webhook_7f92kLm31Qx8",
        telegram_webhook,
        methods=["POST"],
    ),
]

app = Starlette(
    routes=routes,
)


# ============================================================
# TELEGRAM SETUP
# ============================================================

async def setup_telegram():
    global application

    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "TELEGRAM_TOKEN is not configured"
        )

    application = (
        Application
        .builder()
        .token(TELEGRAM_TOKEN)
        .updater(None)
        .build()
    )

    # Commands
    application.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "reset",
            reset_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "forget",
            forget_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "persona",
            persona_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "style",
            style_command,
        )
    )

    # Style buttons
    application.add_handler(
        CallbackQueryHandler(
            style_callback,
            pattern=r"^style:",
        )
    )

    # Photo
    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo,
        )
    )

    # Video analysis
    application.add_handler(
        MessageHandler(
            filters.VIDEO,
            handle_video,
        )
    )

    # Text
    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            message_handler,
        )
    )

    application.add_error_handler(
        error_handler
    )

    await application.initialize()
    await application.start()

    logger.info(
        "Telegram application started"
    )


# ============================================================
# SET WEBHOOK
# ============================================================

async def set_telegram_webhook():
    if application is None:
        raise RuntimeError(
            "Telegram application is not initialized"
        )

    webhook_url = WEBHOOK_URL.strip()

    if not webhook_url:
        raise RuntimeError(
            "WEBHOOK_URL is empty"
        )

    logger.info(
        "Setting webhook: %s",
        webhook_url,
    )

    await application.bot.set_webhook(
        url=webhook_url,
        secret_token=WEBHOOK_SECRET,
        drop_pending_updates=True,
    )

    logger.info(
        "Telegram webhook set successfully"
    )


# ============================================================
# STARTUP
# ============================================================

async def startup():
    logger.info(
        "Starting Ox Alpha AI..."
    )

    await asyncio.to_thread(
        init_db
    )

    await setup_telegram()

    await set_telegram_webhook()

    logger.info(
        "Ox Alpha AI started successfully 🤖"
    )


async def shutdown():
    global application

    logger.info(
        "Stopping Ox Alpha AI..."
    )

    if application is not None:

        try:
            await application.stop()
        except Exception as exc:
            logger.warning(
                "Telegram stop error: %s",
                exc,
            )

        try:
            await application.shutdown()
        except Exception as exc:
            logger.warning(
                "Telegram shutdown error: %s",
                exc,
            )

    logger.info(
        "Ox Alpha AI stopped"
    )


# ============================================================
# LIFESPAN
# ============================================================

@app.on_event("startup")
async def on_startup():
    await startup()


@app.on_event("shutdown")
async def on_shutdown():
    await shutdown()


# ============================================================
# RENDER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "10000",
        )
    )

    logger.info(
        "Starting Uvicorn on port %s",
        port,
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
  )
