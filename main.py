import os
import base64
import asyncio
import logging
import re
import json

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


# =========================================================
# НАСТРОЙКИ
# =========================================================

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
HF_API_KEY = os.environ["HF_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]

FREE_SERP_URL = "https://freeserp.ai/api.php"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HF_VIDEO_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "Wan-AI/Wan2.1-T2V-1.3B"
)

RENDER_URL = os.environ.get(
    "RENDER_EXTERNAL_URL",
    "https://bot-ai-oxalpha.onrender.com",
)

PORT = int(os.environ.get("PORT", 10000))
WEBHOOK_PATH = f"/telegram/{WEBHOOK_SECRET}"


# =========================================================
# МОДЕЛИ
# =========================================================

OPENROUTER_TEXT_MODEL = (
    "nvidia/nemotron-3-ultra-550b-a55b:free"
)

OPENROUTER_MULTIMODAL_MODEL = (
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
)


# =========================================================
# ЛИМИТЫ
# =========================================================

HISTORY_LIMIT = 8
HISTORY_CHARS_LIMIT = 6000
MEMORY_LIMIT = 2500
PERSONA_LIMIT = 1000

WEB_CONTEXT_LIMIT = 4000
WEB_RESULTS_LIMIT = 5

MAX_OUTPUT_TOKENS = 1600
SEARCH_QUERY_TOKENS = 120
MEMORY_OUTPUT_TOKENS = 250

VIDEO_GEN_TIMEOUT = 280
VIDEO_GEN_MAX_RETRIES = 3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# СТИЛИ ОТВЕТОВ
# =========================================================

DEFAULT_STYLE = "clean"

STYLE_PROMPTS = {
    "clean": (
        "Отвечай чисто, по делу, дружелюбным нейтральным тоном. "
        "Обычный связный текст, без заголовков, без нумерованных "
        "разделов и жирного текста, если это не таблица или список "
        "с реально однородными пунктами."
    ),
    "minimal": (
        "Отвечай максимально коротко — 1-3 предложения, "
        "без вступлений, без воды, только суть. Никаких заголовков "
        "и разбивки на пункты."
    ),
    "terminal": (
        "Отвечай в стиле консольного/терминального вывода: "
        "моно-стиль, чётко, по пунктам, минимум эмодзи и "
        "украшений, как будто это лог программы."
    ),
    "meme": (
        "Ты живой участник переписки, а не консультант. Отвечай "
        "неформально, с юмором, сленгом и эмодзи, как будто пишешь "
        "корешу в чат. Обычным разговорным текстом, БЕЗ заголовков, "
        "БЕЗ нумерованных разделов, БЕЗ формального 'разбора по "
        "пунктам' — никогда не анализируй переписку как консультант, "
        "просто отвечай по сути так, будто сам сидишь в этом чате."
    ),
    "detailed": (
        "Отвечай подробно и развёрнуто, с пояснениями и примерами. "
        "Заголовки и списки уместны только если реально помогают "
        "структуре, а не для галочки."
    ),
    "unhinged": (
        "Ты реально угарный, конченый участник чата, а не ассистент "
        "и не консультант. Пиши как в переписке с корешами: живой "
        "разговорный сленг, открытый мат (свободно, но не через "
        "слово — не ради мата самого по себе), рваные фразы, местами "
        "КАПС для акцента на эмоции, спам эмодзи типа 💀😭🔥 в тему. "
        "Реагируй гиперболизированно и с самоиронией — если облажался "
        "или спалили косяк, комично признай это вместо того чтобы "
        "извиняться по-канцелярски. Обращайся к собеседнику "
        "неформально. Никогда не скатывайся в структурированный "
        "или деловой тон."
    ),
}

GLOBAL_RULES = (
    "Общие правила независимо от стиля:\n"
    "- Не делай формальный разбор или анализ сообщения пользователя, "
    "если тебя об этом явно не просили — отвечай сразу по сути.\n"
    "- Даже если пользователь прислал большой кусок текста, чат-лог "
    "или переписку, не превращай ответ в отчёт с заголовками и "
    "нумерацией — просто выполни то, о чём реально попросили.\n"
    "- Не используй заголовки, жирный текст и нумерованные разделы "
    "по умолчанию — только когда это правда нужно для структуры данных."
)

STYLE_LABELS = {
    "clean": "Clean 🧼",
    "minimal": "Minimal ✂️",
    "terminal": "Terminal 💻",
    "meme": "Meme 😂",
    "detailed": "Detailed 📖",
    "unhinged": "Угар 💀",
}

DEFAULT_PERSONA_DESCRIPTION = (
    "Ты дружелюбный ИИ-помощник в Telegram. "
    "Отвечай понятно, естественно и по делу. "
    "Не выдавай непроверенные факты за достоверные."
)


# =========================================================
# DATABASE
# =========================================================

def get_db():
    return psycopg.connect(DATABASE_URL)


def init_db():
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id BIGINT PRIMARY KEY,
                    memory TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cur.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                "persona TEXT DEFAULT ''"
            )

            cur.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                "style TEXT DEFAULT 'clean'"
            )

        conn.commit()

    logger.info("Database initialized")


def ensure_user(user_id: int):
    with get_db() as conn:
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


def save_message(user_id: int, role: str, content: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (user_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (user_id, role, content),
            )

        conn.commit()


def get_history(user_id: int, limit: int = HISTORY_LIMIT):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT role, content
                FROM messages
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )

            rows = cur.fetchall()

    rows.reverse()

    return rows


def get_compact_history(user_id: int):
    rows = get_history(user_id)

    result = []
    total = 0

    for role, content in reversed(rows):
        if not content:
            continue

        content = content[:1800]

        if total + len(content) > HISTORY_CHARS_LIMIT:
            break

        result.append((role, content))
        total += len(content)

    result.reverse()

    return result


def get_memory(user_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT memory FROM users WHERE user_id = %s",
                (user_id,),
            )

            row = cur.fetchone()

    if not row:
        return ""

    return (row[0] or "")[:MEMORY_LIMIT]


def set_memory(user_id: int, memory: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET memory = %s WHERE user_id = %s",
                (memory[:MEMORY_LIMIT], user_id),
            )

        conn.commit()


def reset_history(user_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM messages WHERE user_id = %s",
                (user_id,),
            )

        conn.commit()


def reset_memory(user_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET memory = '' WHERE user_id = %s",
                (user_id,),
            )

        conn.commit()


def get_persona(user_id: int) -> str:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT persona FROM users WHERE user_id = %s",
                (user_id,),
            )

            row = cur.fetchone()

    if not row:
        return ""

    return (row[0] or "")[:PERSONA_LIMIT]


def set_persona(user_id: int, persona: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET persona = %s WHERE user_id = %s",
                (persona[:PERSONA_LIMIT], user_id),
            )

        conn.commit()


def reset_persona(user_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET persona = '' WHERE user_id = %s",
                (user_id,),
            )

        conn.commit()


def get_style(user_id: int) -> str:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT style FROM users WHERE user_id = %s",
                (user_id,),
            )

            row = cur.fetchone()

    if not row or not row[0]:
        return DEFAULT_STYLE

    return row[0] if row[0] in STYLE_PROMPTS else DEFAULT_STYLE


def set_style(user_id: int, style: str):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET style = %s WHERE user_id = %s",
                (style, user_id),
            )

        conn.commit()


def should_update_memory(text: str) -> bool:
    t = text.lower().strip()

    if len(t) < 8:
        return False

    triggers = (
        "меня зовут",
        "зови меня",
        "называй меня",
        "мое имя",
        "моё имя",
        "я делаю",
        "я занимаюсь",
        "мой проект",
        "моя игра",
        "мне нравится",
        "я люблю",
        "я не люблю",
        "мне не нравится",
        "предпочитаю",
        "предпочтение",
        "запомни",
        "запоминай",
        "помни",
        "учти в будущем",
        "в дальнейшем",
        "я использую",
        "мой бот",
        "моя система",
    )

    return any(trigger in t for trigger in triggers)


# =========================================================
# SYSTEM PROMPT
# =========================================================

def build_system_prompt(user_id: int, extra: str = "") -> str:
    persona = get_persona(user_id)
    style = get_style(user_id)
    memory = get_memory(user_id)

    base = persona.strip() if persona else DEFAULT_PERSONA_DESCRIPTION

    style_instruction = STYLE_PROMPTS.get(
        style,
        STYLE_PROMPTS[DEFAULT_STYLE],
    )

    parts = [
        base,
        style_instruction,
        GLOBAL_RULES,
    ]

    if extra:
        parts.append(extra)

    parts.append(
        "Память пользователя:\n"
        + (memory or "Память пока пустая.")
    )

    return "\n\n".join(parts)


async def update_memory(user_id: int, text: str):
    if not should_update_memory(text):
        return

    try:
        current_memory = get_memory(user_id)

        prompt = f"""
Ты управляешь долговременной памятью пользователя.

Текущая память:
{current_memory or "(пусто)"}

Новое сообщение:
{text[:1200]}

Сохраняй только полезную долгосрочную информацию:
- имя и обращение;
- интересы;
- проекты;
- устойчивые предпочтения;
- полезные настройки общения.

Не сохраняй временные задачи, результаты поиска,
пароли, API-ключи, токены и секреты.
Не сохраняй персонажа и стиль ответов — они хранятся отдельно.

Если сохранять нечего, напиши:
NO_UPDATE

Иначе верни полностью обновлённую краткую память.
Без объяснений. Максимум 2500 символов.
"""

        payload = {
            "model": OPENROUTER_TEXT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты аккуратно управляешь "
                        "долговременной памятью."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": MEMORY_OUTPUT_TOKENS,
            "temperature": 0,
        }

        data = await openrouter_request(payload, 60)

        new_memory = extract_openrouter_text(data)

        if new_memory and new_memory != "NO_UPDATE":
            set_memory(user_id, new_memory)
            logger.info(
                "Memory updated for user %s",
                user_id,
            )

    except Exception:
        logger.exception("Memory update error")


# =========================================================
# OPENROUTER
# =========================================================

def extract_openrouter_text(data: dict) -> str:
    """
    Безопасно достаёт текст из ответа OpenRouter.

    Раньше здесь напрямую использовалось:
    data["choices"][0]["message"]["content"]

    Если API возвращал ошибку/другую структуру,
    возникал KeyError.

    Теперь структура проверяется явно.
    """

    if not isinstance(data, dict):
        raise RuntimeError(
            "OpenRouter returned non-object JSON"
        )

    if "error" in data:
        error = data.get("error")

        if isinstance(error, dict):
            message = (
                error.get("message")
                or error.get("code")
                or "Unknown OpenRouter error"
            )
        else:
            message = str(error)

        raise RuntimeError(
            f"OpenRouter API error: {message}"
        )

    choices = data.get("choices")

    if not isinstance(choices, list) or not choices:
        logger.error(
            "Unexpected OpenRouter response: %s",
            json.dumps(data, ensure_ascii=False)[:3000],
        )

        raise RuntimeError(
            "OpenRouter не вернул choices в ответе."
        )

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        raise RuntimeError(
            "OpenRouter вернул некорректный choice."
        )

    message = first_choice.get("message")

    if not isinstance(message, dict):
        logger.error(
            "OpenRouter choice without message: %s",
            json.dumps(first_choice, ensure_ascii=False)[:2000],
        )

        raise RuntimeError(
            "OpenRouter не вернул message."
        )

    content = message.get("content")

    if content is None:
        logger.error(
            "OpenRouter message without content: %s",
            json.dumps(message, ensure_ascii=False)[:2000],
        )

        raise RuntimeError(
            "OpenRouter не вернул content."
        )

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(str(text))

        content = "\n".join(parts)

    content = str(content).strip()

    if not content:
        raise RuntimeError(
            "OpenRouter returned an empty response."
        )

    return content


async def openrouter_request(
    payload: dict,
    timeout: int = 120,
):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": RENDER_URL,
        "X-Title": "Ox Alpha AI",
    }

    async with httpx.AsyncClient(timeout=timeout) as client:

        for attempt in range(2):

            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
            )

            if response.status_code == 429:
                logger.warning(
                    "OpenRouter rate limit, attempt %s",
                    attempt + 1,
                )

                if attempt == 0:
                    retry_after = 7

                    value = response.headers.get(
                        "retry-after"
                    )

                    if value:
                        try:
                            retry_after = max(
                                2,
                                min(
                                    int(float(value)),
                                    15,
                                ),
                            )
                        except ValueError:
                            pass

                    await asyncio.sleep(retry_after)
                    continue

            if response.is_error:
                logger.error(
                    "OpenRouter HTTP error %s: %s",
                    response.status_code,
                    response.text[:3000],
                )

            response.raise_for_status()

            try:
                data = response.json()
            except Exception as e:
                logger.error(
                    "OpenRouter returned invalid JSON: %s",
                    response.text[:3000],
                )

                raise RuntimeError(
                    "OpenRouter вернул некорректный JSON."
                ) from e

            return data

    raise RuntimeError(
        "OpenRouter request failed after retry"
    )


async def ask_text(user_id: int, text: str):
    history = get_compact_history(user_id)

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(user_id),
        }
    ]

    for role, content in history:
        messages.append({
            "role": (
                "assistant"
                if role == "model"
                else "user"
            ),
            "content": content,
        })

    messages.append({
        "role": "user",
        "content": text,
    })

    payload = {
        "model": OPENROUTER_TEXT_MODEL,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.7,
    }

    data = await openrouter_request(payload)

    answer = extract_openrouter_text(data)

    return answer.strip()


# =========================================================
# WEB SEARCH
# =========================================================

def wants_web_search(text: str) -> bool:
    t = text.lower().strip()

    phrases = (
        "в интернете",
        "в инете",
        "в сети",
        "через интернет",
        "через поиск",
        "сделай поиск",
        "проведи поиск",
        "поищи в интернете",
        "поищи в инете",
        "поищи в сети",
        "найди в интернете",
        "найди в инете",
        "найди в сети",
        "посмотри в интернете",
        "посмотри в инете",
        "посмотри в сети",
        "проверь в интернете",
        "проверь в инете",
        "проверь в сети",
        "актуальные данные",
        "актуальную информацию",
        "актуальная информация",
        "актуальные цены",
        "актуальную цену",
        "актуальная цена",
        "последние новости",
        "свежие новости",
        "что нового",
        "что сейчас",
        "сколько сейчас стоит",
        "цена сейчас",
        "цены сейчас",
        "на данный момент",
        "на сегодня",
        "сегодня",
        "сейчас",
        "в 2026",
        "за 2026 год",
        "на 2026 год",
    )

    if any(phrase in t for phrase in phrases):
        return True

    words = (
        "поищи",
        "погугли",
        "проверь",
        "поиск",
        "найди",
    )

    return any(
        re.search(
            rf"\b{re.escape(word)}\b",
            t,
        )
        for word in words
    )


async def make_search_query(text: str) -> str:
    prompt = f"""
Преобразуй запрос пользователя в короткий
и точный поисковый запрос для веб-поисковика.

Тебе НЕ нужно отвечать на вопрос.
Тебе нужно только составить поисковую строку.

Учитывай:
- тему запроса;
- нужный период времени;
- страну или язык, если они важны;
- конкретные названия;
- актуальность информации.

Если пользователь просит научные данные,
ищи именно научные публикации и авторитетные
источники.

Не добавляй объяснения.
Не используй кавычки.
Верни только одну строку поискового запроса.

Запрос пользователя:
{text[:2000]}
"""

    payload = {
        "model": OPENROUTER_TEXT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты генератор поисковых запросов. "
                    "Отвечай только поисковой строкой."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": SEARCH_QUERY_TOKENS,
        "temperature": 0,
    }

    data = await openrouter_request(
        payload,
        60,
    )

    query = extract_openrouter_text(data)

    query = query.replace("\n", " ")
    query = re.sub(r"\s+", " ", query)
    query = query.strip("\"'` ")

    if not query:
        return text[:600]

    return query[:600]


# =========================================================
# FREE SERP — ИСПРАВЛЕННАЯ ВЕРСИЯ
# =========================================================

async def freeserp_search(query: str) -> str:
    params = {
        "index": "web",
        "q": query[:600],
        "size": WEB_RESULTS_LIMIT,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "OxAlphaAI/1.0",
    }

    logger.info(
        "Sending FreeSerp request: %s",
        query[:600],
    )

    async with httpx.AsyncClient(
        timeout=30
    ) as client:

        response = await client.get(
            FREE_SERP_URL,
            params=params,
            headers=headers,
        )

    logger.info(
        "FreeSerp HTTP status: %s",
        response.status_code,
    )

    raw_text = response.text[:5000]

    if response.is_error:
        logger.error(
            "FreeSerp HTTP error %s: %s",
            response.status_code,
            raw_text,
        )

        raise RuntimeError(
            f"FreeSerp вернул HTTP {response.status_code}"
        )

    try:
        data = response.json()
    except Exception as e:
        logger.error(
            "FreeSerp returned invalid JSON: %s",
            raw_text,
        )

        raise RuntimeError(
            "FreeSerp вернул некорректный JSON."
        ) from e

    if not isinstance(data, dict):
        logger.error(
            "FreeSerp JSON is not an object: %s",
            repr(data)[:3000],
        )

        raise RuntimeError(
            "FreeSerp вернул неожиданный формат данных."
        )

    # -----------------------------------------------------
    # Проверяем возможную ошибку API
    # -----------------------------------------------------

    if data.get("error"):
        error = data.get("error")

        if isinstance(error, dict):
            error_text = (
                error.get("message")
                or error.get("code")
                or str(error)
            )
        else:
            error_text = str(error)

        logger.error(
            "FreeSerp API error: %s",
            error_text,
        )

        raise RuntimeError(
            f"FreeSerp API error: {error_text}"
        )

    # -----------------------------------------------------
    # Поддерживаем несколько возможных структур ответа
    # -----------------------------------------------------

    results = []

    direct_results = data.get("results")

    if isinstance(direct_results, list):
        results = direct_results

    elif isinstance(data.get("web"), dict):

        web_results = data["web"].get("results")

        if isinstance(web_results, list):
            results = web_results

    elif isinstance(data.get("organic"), list):
        results = data["organic"]

    # -----------------------------------------------------
    # Если результатов нет — логируем реальный JSON
    # -----------------------------------------------------

    if not results:
        logger.warning(
            "FreeSerp returned no search results. "
            "Full response: %s",
            json.dumps(
                data,
                ensure_ascii=False,
            )[:5000],
        )

        return (
            "Поисковик не вернул результатов "
            "по этому запросу."
        )

    # -----------------------------------------------------
    # Формируем контекст для модели
    # -----------------------------------------------------

    lines = ["Результаты веб-поиска:"]
    total = len(lines[0])

    for i, item in enumerate(
        results[:WEB_RESULTS_LIMIT],
        1,
    ):

        if not isinstance(item, dict):
            logger.warning(
                "FreeSerp result %s is not object: %r",
                i,
                item,
            )
            continue

        title = (
            item.get("title")
            or item.get("name")
            or "Без названия"
        )

        url = (
            item.get("url")
            or item.get("link")
            or item.get("href")
            or ""
        )

        description = (
            item.get("snippet")
            or item.get("summary")
            or item.get("description")
            or item.get("text")
            or ""
        )

        published = (
            item.get("publication_date")
            or item.get("published")
            or item.get("date")
            or ""
        )

        title = str(title).strip()[:300]
        url = str(url).strip()[:500]
        description = str(description).strip()[:700]
        published = str(published).strip()[:100]

        block = (
            f"{i}. {title}\n"
            f"URL: {url}\n"
            f"Описание: {description}"
        )

        if published:
            block += (
                f"\nДата публикации: {published}"
            )

        if (
            total + len(block) + 2
            > WEB_CONTEXT_LIMIT
        ):
            break

        lines.append(block)
        total += len(block) + 2

    if len(lines) == 1:
        return (
            "Поисковик вернул данные, "
            "но подходящих результатов "
            "для передачи модели не найдено."
        )

    return "\n\n".join(lines)


# =========================================================
# ОТВЕТ С WEB
# =========================================================

async def ask_text_with_web(
    user_id: int,
    text: str,
    search_query: str,
    web_context: str,
):
    history = get_compact_history(user_id)

    web_rules = (
        "Пользователь запросил актуальную информацию из интернета.\n"
        "Ниже находятся результаты веб-поиска.\n\n"
        "ВАЖНЫЕ ПРАВИЛА:\n"
        "1. Используй результаты поиска как основной источник "
        "актуальной информации.\n"
        "2. Не придумывай факты, даты, цифры, названия исследований "
        "или события.\n"
        "3. Никогда не придумывай URL.\n"
        "4. Если указываешь ссылку, она должна существовать среди "
        "URL в результатах поиска.\n"
        "5. Если результаты плохие или не отвечают на вопрос, "
        "честно скажи об этом.\n"
        "6. Если источники противоречат друг другу, укажи на "
        "противоречие.\n"
        "7. Не рассказывай пользователю о внутреннем поисковом "
        "промпте.\n"
        "8. Не говори, что у тебя нет доступа к интернету, если "
        "результаты поиска были получены.\n"
        "9. Отвечай непосредственно на исходный запрос пользователя."
    )

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                user_id,
                web_rules,
            ),
        }
    ]

    for role, content in history:
        messages.append({
            "role": (
                "assistant"
                if role == "model"
                else "user"
            ),
            "content": content,
        })

    messages.append({
        "role": "user",
        "content": (
            "Исходный запрос пользователя:\n"
            f"{text[:2000]}\n\n"
            "Поисковый запрос, который был "
            "сгенерирован для поисковика:\n"
            f"{search_query[:600]}\n\n"
            "Результаты веб-поиска:\n"
            f"{web_context[:WEB_CONTEXT_LIMIT]}\n\n"
            "Теперь дай пользователю полноценный "
            "ответ на его исходный запрос."
        ),
    })

    payload = {
        "model": OPENROUTER_TEXT_MODEL,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.4,
    }

    data = await openrouter_request(payload)

    answer = extract_openrouter_text(data)

    return answer.strip()


# =========================================================
# МУЛЬТИМОДАЛЬНАЯ МОДЕЛЬ
# =========================================================

async def ask_multimodal(
    user_id: int,
    text: str,
    media_b64: str,
    media_type: str,
    mime_type: str,
):
    history = get_compact_history(user_id)

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(user_id),
        }
    ]

    for role, content in history:
        messages.append({
            "role": (
                "assistant"
                if role == "model"
                else "user"
            ),
            "content": content,
        })

    if media_type == "video":
        media_block = {
            "type": "video_url",
            "video_url": {
                "url": (
                    f"data:{mime_type};"
                    f"base64,{media_b64}"
                )
            },
        }
    else:
        media_block = {
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:{mime_type};"
                    f"base64,{media_b64}"
                )
            },
        }

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text[:2000],
            },
            media_block,
        ],
    })

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": RENDER_URL,
        "X-Title": "Ox Alpha AI",
    }

    payload = {
        "model": OPENROUTER_MULTIMODAL_MODEL,
        "messages": messages,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    timeout = (
        180
        if media_type == "video"
        else 120
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as client:

        response = await client.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
        )

        if response.is_error:
            logger.error(
                "OpenRouter multimodal (%s) error %s: %s",
                media_type,
                response.status_code,
                response.text[:3000],
            )

        response.raise_for_status()

        try:
            data = response.json()
        except Exception as e:
            raise RuntimeError(
                "OpenRouter вернул некорректный JSON."
            ) from e

    answer = extract_openrouter_text(data)

    return answer.strip()


# =========================================================
# VIDEO GENERATION
# =========================================================

async def hf_generate_video(prompt: str) -> bytes:

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt[:800]
    }

    async with httpx.AsyncClient(
        timeout=VIDEO_GEN_TIMEOUT
    ) as client:

        for attempt in range(
            VIDEO_GEN_MAX_RETRIES
        ):

            response = await client.post(
                HF_VIDEO_URL,
                headers=headers,
                json=payload,
            )

            if response.status_code == 503:

                try:
                    wait_for = (
                        response.json()
                        .get(
                            "estimated_time",
                            20,
                        )
                    )
                except Exception:
                    wait_for = 20

                logger.info(
                    "HF video model is loading, "
                    "waiting %s s (attempt %s)",
                    wait_for,
                    attempt + 1,
                )

                await asyncio.sleep(
                    min(wait_for, 60)
                )

                continue

            if response.status_code == 429:
                logger.warning(
                    "HF video rate limit hit"
                )

                raise RuntimeError(
                    "Бесплатный лимит генераций видео "
                    "на Hugging Face исчерпан, "
                    "попробуй чуть позже."
                )

            if response.is_error:
                logger.error(
                    "HF video error %s: %s",
                    response.status_code,
                    response.text[:1000],
                )

            response.raise_for_status()

            content_type = (
                response.headers.get(
                    "content-type",
                    "",
                )
            )

            if (
                "video" not in content_type
                and "octet-stream" not in content_type
            ):
                logger.error(
                    "Unexpected HF video "
                    "content-type %s: %s",
                    content_type,
                    response.text[:500],
                )

                raise RuntimeError(
                    "Hugging Face вернул не видео — "
                    "возможно, модель сейчас перегружена."
                )

            return response.content

    raise RuntimeError(
        "Модель для генерации видео не прогрузилась "
        "за отведённое время, попробуй ещё раз позже."
    )


async def video_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    user_id = update.effective_user.id

    ensure_user(user_id)

    prompt = " ".join(
        context.args
    ).strip()

    if not prompt:
        await update.message.reply_text(
            "Опиши, что сгенерировать 🎬\n\n"
            "Пример: /video кот катается "
            "на скейте по неоновому городу"
        )
        return

    status_msg = await update.message.reply_text(
        "🎬 Генерирую видео, это может занять "
        "пару минут..."
    )

    await update.message.chat.send_action(
        "upload_video"
    )

    try:
        video_bytes = await hf_generate_video(
            prompt
        )

        await update.message.reply_video(
            video=video_bytes,
            caption=f"🎬 {prompt}"[:1000],
        )

        save_message(
            user_id,
            "user",
            f"[Видео-запрос] {prompt}",
        )

        save_message(
            user_id,
            "model",
            "[Сгенерировано видео]",
        )

        await status_msg.delete()

    except Exception as e:

        logger.exception(
            "Video generation error"
        )

        await status_msg.edit_text(
            "Не получилось сгенерировать видео 😵\n\n"
            f"Причина: {str(e)[:300]}"
        )


# =========================================================
# TELEGRAM HELPERS
# =========================================================

def reset_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🧹 Сбросить историю",
                callback_data="reset_history",
            ),
            InlineKeyboardButton(
                "🧠🗑️ Забыть память",
                callback_data="reset_memory",
            ),
        ]
    ])


def persona_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🗑️ Сбросить персонажа",
                callback_data="persona_reset",
            )
        ]
    ])


def style_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🧼 Clean",
                callback_data="style_clean",
            ),
            InlineKeyboardButton(
                "✂️ Minimal",
                callback_data="style_minimal",
            ),
        ],
        [
            InlineKeyboardButton(
                "💻 Terminal",
                callback_data="style_terminal",
            ),
            InlineKeyboardButton(
                "😂 Meme",
                callback_data="style_meme",
            ),
        ],
        [
            InlineKeyboardButton(
                "📖 Detailed",
                callback_data="style_detailed",
            )
        ],
        [
            InlineKeyboardButton(
                "💀 Угар",
                callback_data="style_unhinged",
            )
        ],
    ])


async def send_long_message(
    message,
    text: str,
    reply_markup=None,
):
    max_length = 4000

    if len(text) <= max_length:
        await message.reply_text(
            text,
            reply_markup=reply_markup,
        )
        return

    chunks = [
        text[i:i + max_length]
        for i in range(
            0,
            len(text),
            max_length,
        )
    ]

    for chunk in chunks[:-1]:
        await message.reply_text(chunk)

    await message.reply_text(
        chunks[-1],
        reply_markup=reply_markup,
    )


# =========================================================
# COMMANDS
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.reply_text(
        "Привет! 🧠\n\n"
        "Я Ox Alpha AI на OpenRouter.\n\n"
        "У меня есть:\n"
        "🧠 долговременная память\n"
        "💬 отдельная история для каждого пользователя\n"
        "🎭 персонаж — /persona\n"
        "🎨 стиль ответов — /style\n"
        "📸 анализ фото и видео\n"
        "📄 чтение TXT\n"
        "🌐 поиск в интернете\n"
        "🎬 генерация видео — /video <описание>\n\n"
        "Доступны /reset и /forget.",
        reply_markup=reset_keyboard(),
    )


async def reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id

    ensure_user(user_id)

    reset_history(user_id)

    await update.message.reply_text(
        "История очищена 🧹\n"
        "Долговременная память, персонаж "
        "и стиль сохранены.",
        reply_markup=reset_keyboard(),
    )


async def forget(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id

    ensure_user(user_id)

    reset_memory(user_id)

    await update.message.reply_text(
        "Долговременная память очищена 🧠🗑️"
    )


async def persona_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id

    ensure_user(user_id)

    args = context.args

    if not args:
        persona = get_persona(user_id)

        text = (
            persona
            or
            "Персонаж не задан — использую "
            "поведение по умолчанию."
        )

        await update.message.reply_text(
            f"🎭 Текущий персонаж:\n\n{text}\n\n"
            "Чтобы задать нового: "
            "/persona <описание>\n"
            "Например: /persona Ты саркастичный "
            "пиратский капитан",
            reply_markup=persona_keyboard(),
        )

        return

    if args[0].lower() == "reset":
        reset_persona(user_id)

        await update.message.reply_text(
            "Персонаж сброшен 🎭🗑️ "
            "Использую поведение по умолчанию."
        )

        return

    persona_text = " ".join(
        args
    )[:PERSONA_LIMIT]

    set_persona(
        user_id,
        persona_text,
    )

    await update.message.reply_text(
        f"Персонаж обновлён 🎭\n\n"
        f"{persona_text}"
    )


async def style_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    user_id = update.effective_user.id

    ensure_user(user_id)

    current = get_style(user_id)

    await update.message.reply_text(
        f"🎨 Текущий стиль: "
        f"{STYLE_LABELS.get(current, current)}\n\n"
        "Выбери новый стиль ответов:",
        reply_markup=style_keyboard(),
    )


async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query
    user_id = query.from_user.id

    ensure_user(user_id)

    await query.answer()

    if query.data == "reset_history":

        reset_history(user_id)

        await query.edit_message_text(
            "История очищена 🧹\n"
            "Долговременная память, персонаж "
            "и стиль сохранены."
        )

    elif query.data == "reset_memory":

        reset_memory(user_id)

        await query.edit_message_text(
            "Долговременная память очищена 🧠🗑️"
        )

    elif query.data == "persona_reset":

        reset_persona(user_id)

        await query.edit_message_text(
            "Персонаж сброшен 🎭🗑️ "
            "Использую поведение по умолчанию."
        )

    elif query.data.startswith("style_"):

        style = query.data.split(
            "_",
            1,
        )[1]

        if style in STYLE_PROMPTS:

            set_style(
                user_id,
                style,
            )

            await query.edit_message_text(
                "Стиль ответов установлен: "
                f"{STYLE_LABELS.get(style, style)} ✅"
            )


# =========================================================
# TEXT
# =========================================================

async def message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if (
        not update.message
        or not update.message.text
    ):
        return

    user_id = update.effective_user.id
    text = update.message.text

    ensure_user(user_id)

    await update.message.chat.send_action(
        "typing"
    )

    try:

        if wants_web_search(text):

            logger.info(
                "Web search requested by user %s: %s",
                user_id,
                text,
            )

            search_query = (
                await make_search_query(text)
            )

            logger.info(
                "Generated search query: %s",
                search_query,
            )

            web_context = (
                await freeserp_search(
                    search_query
                )
            )

            logger.info(
                "Web context generated successfully "
                "for user %s",
                user_id,
            )

            answer = await ask_text_with_web(
                user_id,
                text,
                search_query,
                web_context,
            )

        else:

            answer = await ask_text(
                user_id,
                text,
            )

        save_message(
            user_id,
            "user",
            text[:10000],
        )

        save_message(
            user_id,
            "model",
            answer[:10000],
        )

        await send_long_message(
            update.message,
            answer,
        )

        await update_memory(
            user_id,
            text,
        )

    except Exception as e:

        logger.exception(
            "Message error"
        )

        await update.message.reply_text(
            "Ошибка при обработке запроса 😵\n\n"
            f"Причина: {type(e).__name__}\n"
            f"Детали: {str(e)[:400]}\n\n"
            "Подробности есть в логах Render."
        )


# =========================================================
# PHOTO
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if (
        not update.message
        or not update.message.photo
    ):
        return

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.chat.send_action(
        "typing"
    )

    try:

        photo = update.message.photo[-1]

        telegram_file = (
            await context.bot.get_file(
                photo.file_id
            )
        )

        image_bytes = (
            await telegram_file.download_as_bytearray()
        )

        image_b64 = base64.b64encode(
            bytes(image_bytes)
        ).decode("utf-8")

        prompt = (
            update.message.caption
            or
            "Проанализируй это изображение."
        )

        answer = await ask_multimodal(
            user_id,
            prompt,
            image_b64,
            media_type="image",
            mime_type="image/jpeg",
        )

        save_message(
            user_id,
            "user",
            "[Фото] " + prompt,
        )

        save_message(
            user_id,
            "model",
            answer,
        )

        await send_long_message(
            update.message,
            answer,
        )

    except Exception as e:

        logger.exception(
            "Photo error"
        )

        await update.message.reply_text(
            "Не получилось обработать фото 😵\n\n"
            f"Ошибка: {type(e).__name__}\n"
            f"Детали: {str(e)[:300]}"
        )


# =========================================================
# VIDEO UNDERSTANDING
# =========================================================

VIDEO_UNDERSTANDING_MAX_BYTES = (
    95_000_000
    if os.environ.get("LOCAL_BOT_API_URL")
    else 19_500_000
)


async def video_understanding_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if (
        not update.message
        or not update.message.video
    ):
        return

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.chat.send_action(
        "typing"
    )

    try:

        video = update.message.video

        if (
            video.file_size
            and video.file_size
            > VIDEO_UNDERSTANDING_MAX_BYTES
        ):
            await update.message.reply_text(
                "Видео слишком большое для разбора 😵\n"
                "Максимум сейчас — примерно 15 МБ."
            )
            return

        telegram_file = (
            await context.bot.get_file(
                video.file_id
            )
        )

        video_bytes = (
            await telegram_file.download_as_bytearray()
        )

        video_b64 = base64.b64encode(
            bytes(video_bytes)
        ).decode("utf-8")

        mime_type = (
            video.mime_type
            or "video/mp4"
        )

        prompt = (
            update.message.caption
            or
            "Опиши, что происходит в этом видео."
        )

        answer = await ask_multimodal(
            user_id,
            prompt,
            video_b64,
            media_type="video",
            mime_type=mime_type,
        )

        save_message(
            user_id,
            "user",
            "[Видео] " + prompt,
        )

        save_message(
            user_id,
            "model",
            answer,
        )

        await send_long_message(
            update.message,
            answer,
        )

    except Exception as e:

        logger.exception(
            "Video understanding error"
        )

        await update.message.reply_text(
            "Не получилось разобрать видео 😵\n\n"
            f"Ошибка: {type(e).__name__}\n"
            f"Детали: {str(e)[:300]}"
        )


# =========================================================
# TXT
# =========================================================

async def txt_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if (
        not update.message
        or not update.message.document
    ):
        return

    document = update.message.document

    filename = (
        document.file_name
        or ""
    )

    if not filename.lower().endswith(".txt"):

        await update.message.reply_text(
            "Пока поддерживаются только "
            ".txt файлы 📄"
        )

        return

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.chat.send_action(
        "typing"
    )

    try:

        telegram_file = (
            await context.bot.get_file(
                document.file_id
            )
        )

        file_bytes = (
            await telegram_file.download_as_bytearray()
        )

        if len(file_bytes) > 500_000:

            await update.message.reply_text(
                "Файл слишком большой.\n"
                "Максимум сейчас — 500 КБ."
            )

            return

        try:

            file_text = bytes(
                file_bytes
            ).decode("utf-8")

        except UnicodeDecodeError:

            file_text = bytes(
                file_bytes
            ).decode(
                "cp1251",
                errors="replace",
            )

        caption = (
            update.message.caption
            or ""
        )

        prompt = (
            "Пользователь отправил TXT-файл.\n\n"
            f"Имя файла:\n{filename}\n\n"
            f"Комментарий:\n{caption[:2000]}\n\n"
            "Содержимое:\n\n"
            f"{file_text[:20000]}\n\n"
            "Проанализируй файл и ответь на запрос. "
            "Если отдельного запроса нет, кратко "
            "объясни содержимое файла."
        )

        answer = await ask_text(
            user_id,
            prompt,
        )

        save_message(
            user_id,
            "user",
            f"[TXT: {filename}] {caption}",
        )

        save_message(
            user_id,
            "model",
            answer,
        )

        await send_long_message(
            update.message,
            answer,
        )

    except Exception as e:

        logger.exception(
            "TXT error"
        )

        await update.message.reply_text(
            "Не получилось прочитать TXT 😵\n\n"
            f"Ошибка: {type(e).__name__}\n"
            f"Детали: {str(e)[:300]}"
        )


# =========================================================
# WEBHOOK / WEB APP
# =========================================================

LOCAL_BOT_API_URL = os.environ.get(
    "LOCAL_BOT_API_URL",
    "",
).rstrip("/")


_builder = (
    Application.builder()
    .token(TELEGRAM_TOKEN)
    .updater(None)
)


if LOCAL_BOT_API_URL:

    _builder = (
        _builder
        .base_url(
            f"{LOCAL_BOT_API_URL}/bot"
        )
        .base_file_url(
            f"{LOCAL_BOT_API_URL}/file/bot"
        )
    )

    logger.info(
        "Using local Bot API server: %s",
        LOCAL_BOT_API_URL,
    )


application = _builder.build()


async def telegram_webhook(
    request: Request,
):
    try:

        data = await request.json()

        update = Update.de_json(
            data,
            application.bot,
        )

        await application.update_queue.put(
            update
        )

        return PlainTextResponse("OK")

    except Exception:

        logger.exception(
            "Webhook error"
        )

        return PlainTextResponse(
            "ERROR",
            status_code=500,
        )


async def health(
    request: Request,
):
    return PlainTextResponse(
        "Ox Alpha AI is alive!"
    )


routes = [
    Route(
        WEBHOOK_PATH,
        telegram_webhook,
        methods=["POST"],
    ),
    Route(
        "/",
        health,
        methods=["GET"],
    ),
    Route(
        "/health",
        health,
        methods=["GET"],
    ),
]

web_app = Starlette(
    routes=routes
)


# =========================================================
# STARTUP / SHUTDOWN
# =========================================================

async def startup():

    init_db()

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        CommandHandler(
            "reset",
            reset,
        )
    )

    application.add_handler(
        CommandHandler(
            "forget",
            forget,
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

    application.add_handler(
        CommandHandler(
            "video",
            video_command,
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            button_handler
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.VIDEO,
            video_understanding_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            txt_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            message_handler,
        )
    )

    await application.initialize()
    await application.start()

    webhook_url = (
        RENDER_URL.rstrip("/")
        + WEBHOOK_PATH
    )

    await application.bot.set_webhook(
        url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
    )

    logger.info(
        "Webhook set: %s",
        webhook_url,
    )

    logger.info(
        "Ox Alpha AI started: %s",
        OPENROUTER_TEXT_MODEL,
    )


async def shutdown():

    try:

        await application.bot.delete_webhook()

    except Exception:

        logger.exception(
            "Failed to delete webhook"
        )

    await application.stop()
    await application.shutdown()


async def main():

    await startup()

    config = uvicorn.Config(
        web_app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )

    server = uvicorn.Server(config)

    try:

        await server.serve()

    finally:

        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
