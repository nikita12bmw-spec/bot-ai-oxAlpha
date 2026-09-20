import os
import base64
import asyncio
import logging
import re

import httpx
import psycopg
import uvicorn

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from telegram import (
Update,
InlineKeyboardButton,
InlineKeyboardMarkup,
)
from telegram.ext import (
Application,
CommandHandler,
MessageHandler,
CallbackQueryHandler,
ContextTypes,
filters,
)

=========================================================

НАСТРОЙКИ

=========================================================

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

FreeSerp — keyless web search

FREE_SERP_URL = "https://freeserp.ai/api.php"

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]

RENDER_URL = os.environ.get(
"RENDER_EXTERNAL_URL",
"https://bot-ai-oxalpha.onrender.com"
)

PORT = int(os.environ.get("PORT", 10000))

WEBHOOK_PATH = f"/telegram/{WEBHOOK_SECRET}"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

Основная текстовая модель

GROQ_TEXT_MODEL = "openai/gpt-oss-120b"

Vision через OpenRouter

OPENROUTER_VISION_MODEL = "openrouter/free"

=========================================================

ОГРАНИЧЕНИЯ КОНТЕКСТА

=========================================================

Раньше отправлялось до 20 сообщений.

Теперь берём меньше, чтобы не убивать TPM.

HISTORY_LIMIT = 8

Максимум символов истории, реально отправляемых модели.

HISTORY_CHARS_LIMIT = 6000

Максимальная долговременная память.

MEMORY_LIMIT = 2500

Максимальный размер веб-контекста.

WEB_CONTEXT_LIMIT = 5000

Максимум результатов поиска.

WEB_RESULTS_LIMIT = 6

Максимальный ответ обычной модели.

MAX_OUTPUT_TOKENS = 1200

Максимальный ответ модели памяти.

MEMORY_OUTPUT_TOKENS = 250

=========================================================

LOGGING

=========================================================

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(name)

=========================================================

DATABASE

=========================================================

def get_db():
return psycopg.connect(DATABASE_URL)

def init_db():

with get_db() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGINT PRIMARY KEY,
                memory TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    conn.commit()

logger.info("Database initialized!")

def ensure_user(user_id: int):

with get_db() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            INSERT INTO users (user_id)
            VALUES (%s)
            ON CONFLICT (user_id) DO NOTHING
        """, (user_id,))

    conn.commit()

def save_message(
user_id: int,
role: str,
content: str,
):

with get_db() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            INSERT INTO messages (user_id, role, content)
            VALUES (%s, %s, %s)
        """, (
            user_id,
            role,
            content,
        ))

    conn.commit()

def get_history(
user_id: int,
limit: int = HISTORY_LIMIT,
):

with get_db() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            SELECT role, content
            FROM messages
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (
            user_id,
            limit,
        ))

        rows = cur.fetchall()

rows.reverse()

return rows

def get_compact_history(user_id: int):

"""
Возвращает небольшую историю.

Ограничиваем и количество сообщений,
и общий объём текста.
"""

rows = get_history(
    user_id,
    HISTORY_LIMIT,
)

result = []
total_chars = 0

for role, content in reversed(rows):

    if not content:
        continue

    # Ограничиваем одно старое сообщение.
    content = content[:1800]

    if total_chars + len(content) > HISTORY_CHARS_LIMIT:
        break

    result.append(
        (role, content)
    )

    total_chars += len(content)

result.reverse()

return result

def get_memory(user_id: int):

with get_db() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            SELECT memory
            FROM users
            WHERE user_id = %s
        """, (user_id,))

        row = cur.fetchone()

if row:
    return (row[0] or "")[:MEMORY_LIMIT]

return ""

def set_memory(
user_id: int,
memory: str,
):

with get_db() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            UPDATE users
            SET memory = %s
            WHERE user_id = %s
        """, (
            memory[:MEMORY_LIMIT],
            user_id,
        ))

    conn.commit()

def reset_history(user_id: int):

with get_db() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            DELETE FROM messages
            WHERE user_id = %s
        """, (user_id,))

    conn.commit()

def reset_memory(user_id: int):

with get_db() as conn:

    with conn.cursor() as cur:

        cur.execute("""
            UPDATE users
            SET memory = ''
            WHERE user_id = %s
        """, (user_id,))

    conn.commit()

=========================================================

MEMORY

=========================================================

def should_update_memory(text: str) -> bool:

"""
Не запускаем отдельный запрос Groq после каждого сообщения.

Память обновляется только если сообщение похоже
на информацию, которую действительно имеет смысл помнить.
"""

t = text.lower().strip()

if len(t) < 8:
    return False

memory_triggers = (
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

if any(
    phrase in t
    for phrase in memory_triggers
):
    return True

return False

async def update_memory(
user_id: int,
text: str,
):

"""
Обновляет долговременную память.

В отличие от старой версии функция вызывается
только для потенциально полезных сообщений.
"""

if not should_update_memory(text):
    return

try:

    current_memory = get_memory(user_id)

    prompt = f"""

Ты управляешь долговременной памятью пользователя Telegram-бота.

Текущая память:
{current_memory if current_memory else "(пусто)"}

Новое сообщение:
{text[:1200]}

Сохраняй только действительно полезную
долговременную информацию.

Можно сохранять:

- имя и предпочитаемое обращение;
- долгосрочные интересы;
- проекты;
- устойчивые предпочтения;
- настройки общения;
- полезные сведения, которые пригодятся позже.

Не сохраняй:

- обычные вопросы;
- временные задачи;
- результаты поиска;
- содержимое сайтов;
- пароли;
- API-ключи;
- токены;
- секреты;
- чувствительную личную информацию.

Если сохранять нечего:
NO_UPDATE

Если есть новая полезная информация,
верни полностью обновлённую краткую память.

Без объяснений.
Максимум 2500 символов.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_TEXT_MODEL,
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

    async with httpx.AsyncClient(
        timeout=60
    ) as client:

        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=payload,
        )

        if response.is_error:

            logger.error(
                "Memory Groq error %s: %s",
                response.status_code,
                response.text[:1000],
            )

            return

        data = response.json()

    try:

        new_memory = (
            data["choices"][0]["message"]["content"]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ):

        logger.error(
            "Unexpected memory response: %s",
            data,
        )

        return

    if not new_memory:
        return

    new_memory = new_memory.strip()

    if not new_memory:
        return

    if new_memory == "NO_UPDATE":
        return

    set_memory(
        user_id,
        new_memory,
    )

    logger.info(
        "Memory updated for user %s",
        user_id,
    )

except Exception:

    logger.exception(
        "MEMORY UPDATE ERROR for user %s",
        user_id,
    )

=========================================================

GROQ REQUEST

=========================================================

async def groq_request(
payload: dict,
timeout: int = 120,
):

"""
Универсальный запрос к Groq.

При 429 ждём немного и пробуем ещё раз.
"""

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

async with httpx.AsyncClient(
    timeout=timeout
) as client:

    for attempt in range(2):

        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=payload,
        )

        if response.status_code == 429:

            logger.warning(
                "Groq rate limit 429, attempt %s",
                attempt + 1,
            )

            if attempt == 0:

                retry_after = 7

                header_value = response.headers.get(
                    "retry-after"
                )

                if header_value:

                    try:
                        retry_after = max(
                            2,
                            min(
                                int(float(header_value)),
                                15,
                            ),
                        )
                    except ValueError:
                        pass

                await asyncio.sleep(
                    retry_after
                )

                continue

        if response.is_error:

            logger.error(
                "Groq error %s: %s",
                response.status_code,
                response.text[:2000],
            )

        response.raise_for_status()

        return response.json()

raise RuntimeError(
    "Groq request failed after retry"
)

=========================================================

GROQ — ОБЫЧНЫЙ ОТВЕТ

=========================================================

async def ask_groq(
user_id: int,
text: str,
image_b64: str = None,
model: str = None,
):

model = model or GROQ_TEXT_MODEL

history = get_compact_history(user_id)
memory = get_memory(user_id)

system_prompt = (
    "Ты дружелюбный ИИ-помощник в Telegram.\n"
    "Отвечай понятно, естественно и по делу.\n"
    "Не выдавай непроверенные факты за достоверные.\n\n"
    "Долговременная память пользователя:\n"
    + (
        memory
        if memory
        else "Память пока пустая."
    )
)

messages = [
    {
        "role": "system",
        "content": system_prompt,
    }
]

for role, content in history:

    mapped_role = (
        "assistant"
        if role == "model"
        else "user"
    )

    messages.append({
        "role": mapped_role,
        "content": content,
    })

if image_b64:

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        f"{image_b64}"
                    )
                },
            },
        ],
    })

else:

    messages.append({
        "role": "user",
        "content": text,
    })

payload = {
    "model": model,
    "messages": messages,
    "max_tokens": MAX_OUTPUT_TOKENS,
    "temperature": 0.7,
}

data = await groq_request(
    payload
)

try:

    answer = (
        data["choices"][0]["message"]["content"]
    )

except (
    KeyError,
    IndexError,
    TypeError,
):

    logger.error(
        "Unexpected Groq response: %s",
        data,
    )

    raise RuntimeError(
        "Groq returned an unexpected response"
    )

if not answer:

    raise RuntimeError(
        "Groq returned an empty response"
    )

return answer.strip()

=========================================================

WEB SEARCH DETECTION

=========================================================

def wants_web_search(text: str) -> bool:

"""
Определяет, нужен ли интернет-поиск.

Теперь понимает не только "поищи",
но и запросы про актуальные/свежие данные.
"""

t = text.lower().strip()

explicit_phrases = (
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

if any(
    phrase in t
    for phrase in explicit_phrases
):
    return True

search_words = (
    "поищи",
    "погугли",
    "проверь",
    "поиск",
    "найди",
)

if any(
    re.search(
        rf"\b{re.escape(word)}\b",
        t,
    )
    for word in search_words
):
    return True

return False

=========================================================

FREE SERP SEARCH

=========================================================

async def freeserp_search(
query: str,
) -> str:

"""
Получает свежие результаты FreeSerp.

Результаты специально ограничиваются,
чтобы не раздувать запрос к Groq.
"""

params = {
    "index": "web",
    "q": query[:600],
    "size": WEB_RESULTS_LIMIT,
}

headers = {
    "Accept": "application/json",
    "User-Agent": (
        "OxAlphaAI/1.0 "
        "(Telegram bot; web search)"
    ),
}

async with httpx.AsyncClient(
    timeout=30
) as client:

    response = await client.get(
        FREE_SERP_URL,
        params=params,
        headers=headers,
    )

if response.is_error:

    logger.error(
        "FreeSerp error %s: %s",
        response.status_code,
        response.text[:1000],
    )

response.raise_for_status()

data = response.json()

results = (
    data.get("results")
    or data.get("web", {}).get(
        "results",
        [],
    )
)

if not results:

    return (
        "FreeSerp не вернул результатов "
        "по этому запросу."
    )

lines = [
    "Результаты веб-поиска:"
]

current_length = len(
    lines[0]
)

for i, item in enumerate(
    results[:WEB_RESULTS_LIMIT],
    1,
):

    title = (
        item.get("title")
        or "Без названия"
    ).strip()

    url = (
        item.get("url")
        or item.get("link")
        or ""
    ).strip()

    description = (
        item.get("snippet")
        or item.get("summary")
        or item.get("description")
        or ""
    ).strip()

    published = (
        item.get("publication_date")
        or item.get("published")
        or ""
    ).strip()

    # Не даём одному результату разрастись.
    title = title[:300]
    url = url[:500]
    description = description[:700]
    published = published[:100]

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
        current_length
        + len(block)
        + 2
        > WEB_CONTEXT_LIMIT
    ):
        break

    lines.append(block)

    current_length += (
        len(block) + 2
    )

return "\n\n".join(lines)

=========================================================

GROQ — WEB SEARCH

=========================================================

async def ask_groq_with_web(
user_id: int,
text: str,
web_context: str,
):

history = get_compact_history(user_id)
memory = get_memory(user_id)

web_context = web_context[
    :WEB_CONTEXT_LIMIT
]

system_prompt = (
    "Ты дружелюбный ИИ-помощник в Telegram.\n"
    "Пользователь запросил актуальную информацию.\n"
    "Ниже приведены результаты веб-поиска.\n"
    "Используй их как источник актуальных данных.\n"
    "Не придумывай факты, которых нет в найденных "
    "данных или которые нельзя логически вывести.\n"
    "Если информации недостаточно или источники "
    "противоречат друг другу — скажи об этом.\n"
    "Для быстро меняющихся данных указывай, "
    "что информация относится к моменту поиска.\n"
    "Если уместно, называй сайты-источники.\n\n"
    "Долговременная память пользователя:\n"
    + (
        memory
        if memory
        else "Память пока пустая."
    )
)

messages = [
    {
        "role": "system",
        "content": system_prompt,
    }
]

for role, content in history:

    mapped_role = (
        "assistant"
        if role == "model"
        else "user"
    )

    messages.append({
        "role": mapped_role,
        "content": content,
    })

messages.append({
    "role": "user",
    "content": (
        "Запрос пользователя:\n"
        f"{text[:1500]}\n\n"
        "Результаты веб-поиска:\n"
        f"{web_context}\n\n"
        "Ответь на запрос пользователя, "
        "опираясь прежде всего на найденные данные."
    ),
})

payload = {
    "model": GROQ_TEXT_MODEL,
    "messages": messages,
    "max_tokens": MAX_OUTPUT_TOKENS,
    "temperature": 0.4,
}

data = await groq_request(
    payload
)

try:

    answer = (
        data["choices"][0]["message"]["content"]
    )

except (
    KeyError,
    IndexError,
    TypeError,
):

    logger.error(
        "Unexpected Groq web response: %s",
        data,
    )

    raise RuntimeError(
        "Groq returned an unexpected response"
    )

if not answer:

    raise RuntimeError(
        "Groq returned an empty response"
    )

return answer.strip()

=========================================================

TELEGRAM HELPERS

=========================================================

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

    await message.reply_text(
        chunk
    )

await message.reply_text(
    chunks[-1],
    reply_markup=reply_markup,
)

=========================================================

COMMANDS

=========================================================

async def start(
update: Update,
context: ContextTypes.DEFAULT_TYPE,
):

user_id = update.effective_user.id

ensure_user(user_id)

await update.message.reply_text(
    "Привет! 🧠\n\n"
    "Я Ox Alpha AI (на Groq).\n\n"
    "У меня есть:\n"
    "🧠 долговременная память\n"
    "💬 отдельная история для каждого пользователя\n"
    "📸 анализ фото\n"
    "📄 чтение TXT\n"
    "🌐 поиск в интернете\n\n"
    "Кнопки ниже — быстрый сброс.\n"
    "Также доступны команды /reset и /forget",
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
    "Долговременная память сохранена.",
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
        "Долговременная память сохранена."
    )

elif query.data == "reset_memory":

    reset_memory(user_id)

    await query.edit_message_text(
        "Долговременная память очищена 🧠🗑️"
    )

=========================================================

TEXT

=========================================================

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

        web_context = await freeserp_search(
            text
        )

        answer = await ask_groq_with_web(
            user_id,
            text,
            web_context,
        )

    else:

        answer = await ask_groq(
            user_id,
            text,
        )

    # Сохраняем историю.
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

    # Память теперь обновляется
    # только для потенциально полезных сообщений.
    try:

        await update_memory(
            user_id,
            text,
        )

    except Exception:

        logger.exception(
            "MEMORY UPDATE ERROR"
        )

    await send_long_message(
        update.message,
        answer,
    )

except Exception as e:

    logger.exception(
        "MESSAGE ERROR"
    )

    await update.message.reply_text(
        "Ошибка при обращении к модели 😵\n\n"
        f"Причина: {type(e).__name__}\n"
        "Подробности есть в логах Render."
    )

=========================================================

PHOTO

=========================================================

async def ask_openrouter_vision(
user_id: int,
text: str,
image_b64: str,
):

history = get_compact_history(user_id)
memory = get_memory(user_id)

system_prompt = (
    "Ты дружелюбный ИИ-помощник в Telegram.\n"
    "Отвечай понятно, естественно и по делу.\n"
    "Не выдавай непроверенные факты за достоверные.\n\n"
    "Долговременная память пользователя:\n"
    + (
        memory
        if memory
        else "Память пока пустая."
    )
)

messages = [
    {
        "role": "system",
        "content": system_prompt,
    }
]

for role, content in history:

    mapped_role = (
        "assistant"
        if role == "model"
        else "user"
    )

    messages.append({
        "role": mapped_role,
        "content": content,
    })

messages.append({
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": text[:2000],
        },
        {
            "type": "image_url",
            "image_url": {
                "url": (
                    "data:image/jpeg;base64,"
                    f"{image_b64}"
                )
            },
        },
    ],
})

headers = {
    "Authorization": (
        f"Bearer {OPENROUTER_API_KEY}"
    ),
    "Content-Type": "application/json",
    "HTTP-Referer": RENDER_URL,
    "X-Title": "Ox Alpha AI",
}

payload = {
    "model": OPENROUTER_VISION_MODEL,
    "messages": messages,
    "max_tokens": MAX_OUTPUT_TOKENS,
}

async with httpx.AsyncClient(
    timeout=120
) as client:

    response = await client.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
    )

    if response.is_error:

        logger.error(
            "OpenRouter vision error %s: %s",
            response.status_code,
            response.text[:2000],
        )

    response.raise_for_status()

    data = response.json()

try:

    answer = (
        data["choices"][0]["message"]["content"]
    )

except (
    KeyError,
    IndexError,
    TypeError,
):

    logger.error(
        "Unexpected OpenRouter vision response: %s",
        data,
    )

    raise RuntimeError(
        "OpenRouter returned an unexpected response"
    )

if not answer:

    raise RuntimeError(
        "OpenRouter returned an empty response"
    )

return answer.strip()

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

    telegram_file = await context.bot.get_file(
        photo.file_id
    )

    image_bytes = (
        await telegram_file.download_as_bytearray()
    )

    image_b64 = base64.b64encode(
        bytes(image_bytes)
    ).decode("utf-8")

    prompt = (
        update.message.caption
        or "Проанализируй это изображение."
    )

    answer = await ask_openrouter_vision(
        user_id,
        prompt,
        image_b64,
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
        "PHOTO ERROR"
    )

    await update.message.reply_text(
        "Не получилось обработать фото 😵\n\n"
        f"Ошибка: {type(e).__name__}"
    )

=========================================================

TXT

=========================================================

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

filename = document.file_name or ""

if not filename.lower().endswith(".txt"):

    await update.message.reply_text(
        "Пока поддерживаются только .txt файлы 📄"
    )

    return

user_id = update.effective_user.id

ensure_user(user_id)

await update.message.chat.send_action(
    "typing"
)

try:

    telegram_file = await context.bot.get_file(
        document.file_id
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

    # Не даём TXT раздувать контекст
    # до безумных размеров.
    file_text = file_text[:20000]

    prompt = (
        "Пользователь отправил TXT-файл.\n\n"
        f"Имя файла:\n{filename}\n\n"
        f"Комментарий:\n{caption[:2000]}\n\n"
        "Содержимое:\n\n"
        f"{file_text}\n\n"
        "Проанализируй файл и ответь "
        "на запрос пользователя. "
        "Если отдельного запроса нет, "
        "кратко объясни содержимое файла."
    )

    answer = await ask_groq(
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
        "TXT ERROR"
    )

    await update.message.reply_text(
        "Не получилось прочитать TXT 😵\n\n"
        f"Ошибка: {type(e).__name__}"
    )

=========================================================

WEBHOOK / WEB APP

=========================================================

application = (
Application.builder()
.token(TELEGRAM_TOKEN)
.updater(None)
.build()
)

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

    return PlainTextResponse(
        "OK"
    )

except Exception:

    logger.exception(
        "WEBHOOK ERROR"
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

=========================================================

STARTUP / SHUTDOWN

=========================================================

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
    "Telegram webhook set: %s",
    webhook_url,
)

logger.info(
    "Ox Alpha AI started with model: %s",
    GROQ_TEXT_MODEL,
)

logger.info(
    "FreeSerp web search: enabled "
    "(no API key required)"
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

=========================================================

RUN

=========================================================

async def main():

await startup()

config = uvicorn.Config(
    web_app,
    host="0.0.0.0",
    port=PORT,
    log_level="info",
)

server = uvicorn.Server(
    config
)

try:

    await server.serve()

finally:

    await shutdown()

if name == "main":
asyncio.run(main())
