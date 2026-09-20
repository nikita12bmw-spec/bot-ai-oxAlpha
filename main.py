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


# =========================================================
# НАСТРОЙКИ
# =========================================================

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

# FreeSerp — бесплатный keyless-поиск, используется только когда пользователь явно просит поиск в интернете.
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

# Текст — GPT-OSS 120B, открытая модель OpenAI на Groq, доступна на твоём
# аккаунте прямо сейчас (в отличие от Llama 3.3/4, которые пока закрыты):
GROQ_TEXT_MODEL = "openai/gpt-oss-120b"
# Фото — на Groq у аккаунта нет ни одной vision-модели, поэтому фото
# отдельно идёт через OpenRouter (бесплатный роутер сам подбирает
# рабочую vision-модель под капотом):
OPENROUTER_VISION_MODEL = "openrouter/free"
# Видео не понимает ни Groq, ни бесплатный OpenRouter — для видео нужен
# отдельный платный провайдер (например Gemini через OpenRouter).


HISTORY_LIMIT = 20


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# DATABASE
# =========================================================

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


def save_message(user_id: int, role: str, content: str):

    with get_db() as conn:

        with conn.cursor() as cur:

            cur.execute("""
                INSERT INTO messages (user_id, role, content)
                VALUES (%s, %s, %s)
            """, (user_id, role, content))

        conn.commit()


def get_history(user_id: int, limit: int = HISTORY_LIMIT):
    """Раздельная история на каждого пользователя (по user_id)."""

    with get_db() as conn:

        with conn.cursor() as cur:

            cur.execute("""
                SELECT role, content
                FROM messages
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (user_id, limit))

            rows = cur.fetchall()

    rows.reverse()

    return rows


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
        return row[0] or ""

    return ""


def set_memory(user_id: int, memory: str):

    with get_db() as conn:

        with conn.cursor() as cur:

            cur.execute("""
                UPDATE users
                SET memory = %s
                WHERE user_id = %s
            """, (memory, user_id))

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


# =========================================================
# GROQ
# =========================================================

async def ask_groq(
    user_id: int,
    text: str,
    image_b64: str = None,
    model: str = None,
):

    model = model or GROQ_TEXT_MODEL

    history = get_history(user_id)
    memory = get_memory(user_id)

    system_prompt = (
        "Ты дружелюбный ИИ-помощник в Telegram.\n"
        "Отвечай понятно, естественно и по делу.\n"
        "Не выдавай непроверенные факты за достоверные.\n\n"
        "Долговременная память пользователя:\n"
        + (memory if memory else "Память пока пустая.")
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    for role, content in history:
        # Groq/OpenAI формат: "assistant" вместо "model"
        mapped_role = "assistant" if role == "model" else "user"
        messages.append({"role": mapped_role, "content": content})

    if image_b64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    },
                },
            ],
        })
    else:
        messages.append({"role": "user", "content": text})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=120) as client:

        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=payload,
        )

        if response.is_error:
            logger.error(
                "Groq error %s: %s",
                response.status_code,
                response.text,
            )

        response.raise_for_status()

        data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        logger.error("Unexpected Groq response: %s", data)
        raise RuntimeError("Groq returned an unexpected response")

    if not answer:
        raise RuntimeError("Groq returned an empty response")

    return answer


# =========================================================
# FREE SERP SEARCH
# =========================================================

def wants_web_search(text: str) -> bool:
    """Поиск запускается только по явной просьбе пользователя."""
    t = text.lower().strip()

    explicit_phrases = (
        "в интернете", "в инете", "в сети",
        "через интернет", "через поиск",
        "сделай поиск", "проведи поиск",
        "поищи в интернете", "поищи в инете", "поищи в сети",
        "найди в интернете", "найди в инете", "найди в сети",
        "посмотри в интернете", "посмотри в инете", "посмотри в сети",
        "проверь в интернете", "проверь в инете", "проверь в сети",
        "актуальные данные в интернете",
        "актуальную информацию в интернете",
        "актуальная информация в интернете",
        "актуальные цены в интернете",
    )

    if any(phrase in t for phrase in explicit_phrases):
        return True

    # Короткие явные команды поиска тоже запускают поиск.
    search_words = ("поищи", "погугли", "проверь", "поиск")
    return any(re.search(rf"\b{re.escape(word)}\b", t) for word in search_words)


async def freeserp_search(query: str) -> str:
    """Получает свежие результаты FreeSerp без API-ключа."""
    params = {
        "index": "web",
        "q": query[:600],
        "size": 8,
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "OxAlphaAI/1.0 (Telegram bot; web search)",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            FREE_SERP_URL,
            params=params,
            headers=headers,
        )

    if response.is_error:
        logger.error("FreeSerp error %s: %s", response.status_code, response.text[:1000])

    response.raise_for_status()

    data = response.json()

    # FreeSerp возвращает результаты в JSON; поддерживаем оба возможных
    # названия массива, чтобы формат ответа был устойчивее к изменениям.
    results = data.get("results") or data.get("web", {}).get("results", [])

    if not results:
        return "FreeSerp не вернул результатов по этому запросу."

    lines = ["Результаты веб-поиска FreeSerp:"]
    for i, item in enumerate(results[:8], 1):
        title = (item.get("title") or "Без названия").strip()
        url = (item.get("url") or item.get("link") or "").strip()
        description = (
            item.get("snippet")
            or item.get("summary")
            or item.get("description")
            or ""
        ).strip()
        published = (item.get("publication_date") or item.get("published") or "").strip()

        block = (
            f"{i}. {title}\n"
            f"URL: {url}\n"
            f"Описание: {description}"
        )
        if published:
            block += f"\nДата публикации: {published}"
        lines.append(block)

    return "\n\n".join(lines)


async def ask_groq_with_web(user_id: int, text: str, web_context: str):
    """Отвечает на запрос, используя результаты FreeSerp."""
    history = get_history(user_id)
    memory = get_memory(user_id)

    system_prompt = (
        "Ты дружелюбный ИИ-помощник в Telegram.\n"
        "Пользователь явно попросил поискать информацию в интернете.\n"
        "Используй результаты веб-поиска FreeSerp как источник актуальной информации.\n"
        "Не придумывай факты, которых нет в результатах поиска.\n"
        "Если источники противоречат друг другу или данных недостаточно — прямо скажи об этом.\n"
        "Для цен и других быстро меняющихся данных указывай, что информация актуальна на момент поиска, "
        "и по возможности называй источник/сайт.\n\n"
        "Долговременная память пользователя:\n"
        + (memory if memory else "Память пока пустая.")
    )

    messages = [{"role": "system", "content": system_prompt}]

    for role, content in history:
        mapped_role = "assistant" if role == "model" else "user"
        messages.append({"role": mapped_role, "content": content})

    messages.append({
        "role": "user",
        "content": (
            f"Запрос пользователя:\n{text}\n\n"
            f"Данные веб-поиска:\n{web_context}\n\n"
            "Ответь на запрос пользователя, опираясь на найденные данные."
        ),
    })

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_TEXT_MODEL,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=payload,
        )

        if response.is_error:
            logger.error(
                "Groq web-search error %s: %s",
                response.status_code,
                response.text,
            )

        response.raise_for_status()

        data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        logger.error("Unexpected Groq response: %s", data)
        raise RuntimeError("Groq returned an unexpected response")

    if not answer:
        raise RuntimeError("Groq returned an empty response")

    return answer


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


async def send_long_message(message, text: str, reply_markup=None):

    max_length = 4000

    if len(text) <= max_length:
        await message.reply_text(text, reply_markup=reply_markup)
        return

    chunks = [
        text[i:i + max_length]
        for i in range(0, len(text), max_length)
    ]

    for chunk in chunks[:-1]:
        await message.reply_text(chunk)

    await message.reply_text(chunks[-1], reply_markup=reply_markup)


# =========================================================
# COMMANDS
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.reply_text(
        "Привет! 🧠\n\n"
        "Я Ox Alpha AI (на Groq).\n\n"
        "У меня есть:\n"
        "🧠 долговременная память\n"
        "💬 отдельная история для каждого пользователя\n"
        "📸 анализ фото\n"
        "📄 чтение TXT\n\n"
        "Кнопки ниже — быстрый сброс.\n"
        "Также доступны команды /reset и /forget",
        reply_markup=reset_keyboard(),
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    ensure_user(user_id)
    reset_history(user_id)

    await update.message.reply_text(
        "История очищена 🧹\n"
        "Долговременная память сохранена.",
        reply_markup=reset_keyboard(),
    )


async def forget(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    ensure_user(user_id)
    reset_memory(user_id)

    await update.message.reply_text(
        "Долговременная память очищена 🧠🗑️",
        reply_markup=reset_keyboard(),
    )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

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


# =========================================================
# TEXT
# =========================================================

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text

    ensure_user(user_id)

    await update.message.chat.send_action("typing")

    try:

        if wants_web_search(text):
            logger.info("FreeSerp search requested by user %s: %s", user_id, text)
            web_context = await freeserp_search(text)
            answer = await ask_groq_with_web(user_id, text, web_context)
        else:
            answer = await ask_groq(user_id, text)

        save_message(user_id, "user", text)
        save_message(user_id, "model", answer)

        await update_memory(user_id, text)

        await send_long_message(update.message, answer)

    except Exception as e:

        logger.exception("MESSAGE ERROR")

        await update.message.reply_text(
            "Ошибка при обращении к модели 😵\n\n"
            f"Причина: {type(e).__name__}\n"
            "Подробности есть в логах Render."
        )


# =========================================================
# PHOTO (через OpenRouter — на Groq нет доступных vision-моделей)
# =========================================================

async def ask_openrouter_vision(user_id: int, text: str, image_b64: str):

    history = get_history(user_id)
    memory = get_memory(user_id)

    system_prompt = (
        "Ты дружелюбный ИИ-помощник в Telegram.\n"
        "Отвечай понятно, естественно и по делу.\n"
        "Не выдавай непроверенные факты за достоверные.\n\n"
        "Долговременная память пользователя:\n"
        + (memory if memory else "Память пока пустая.")
    )

    messages = [{"role": "system", "content": system_prompt}]

    for role, content in history:
        mapped_role = "assistant" if role == "model" else "user"
        messages.append({"role": mapped_role, "content": content})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                },
            },
        ],
    })

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": RENDER_URL,
        "X-Title": "Ox Alpha AI",
    }

    payload = {
        "model": OPENROUTER_VISION_MODEL,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=120) as client:

        response = await client.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
        )

        if response.is_error:
            logger.error(
                "OpenRouter vision error %s: %s",
                response.status_code,
                response.text,
            )

        response.raise_for_status()

        data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        logger.error("Unexpected OpenRouter vision response: %s", data)
        raise RuntimeError("OpenRouter returned an unexpected response")

    if not answer:
        raise RuntimeError("OpenRouter returned an empty response")

    return answer


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message or not update.message.photo:
        return

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.chat.send_action("typing")

    try:

        photo = update.message.photo[-1]

        telegram_file = await context.bot.get_file(photo.file_id)

        image_bytes = await telegram_file.download_as_bytearray()

        image_b64 = base64.b64encode(bytes(image_bytes)).decode("utf-8")

        prompt = update.message.caption or "Проанализируй это изображение."

        answer = await ask_openrouter_vision(
            user_id,
            prompt,
            image_b64=image_b64,
        )

        save_message(user_id, "user", "[Фото] " + prompt)
        save_message(user_id, "model", answer)

        await send_long_message(update.message, answer)

    except Exception as e:

        logger.exception("PHOTO ERROR")

        await update.message.reply_text(
            "Не получилось обработать фото 😵\n\n"
            f"Ошибка: {type(e).__name__}"
        )


# =========================================================
# TXT
# =========================================================

async def txt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message or not update.message.document:
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

    await update.message.chat.send_action("typing")

    try:

        telegram_file = await context.bot.get_file(document.file_id)

        file_bytes = await telegram_file.download_as_bytearray()

        if len(file_bytes) > 500_000:
            await update.message.reply_text(
                "Файл слишком большой.\n"
                "Максимум сейчас — 500 КБ."
            )
            return

        try:
            file_text = bytes(file_bytes).decode("utf-8")
        except UnicodeDecodeError:
            file_text = bytes(file_bytes).decode("cp1251", errors="replace")

        caption = update.message.caption or ""

        prompt = f"""Пользователь отправил TXT-файл.

Имя файла:
{filename}

Комментарий:
{caption}

Содержимое:

{file_text}

Проанализируй файл и ответь на запрос пользователя.
Если отдельного запроса нет, кратко объясни содержимое файла."""

        answer = await ask_groq(user_id, prompt)

        save_message(user_id, "user", f"[TXT: {filename}] {caption}")
        save_message(user_id, "model", answer)

        await send_long_message(update.message, answer)

    except Exception as e:

        logger.exception("TXT ERROR")

        await update.message.reply_text(
            "Не получилось прочитать TXT 😵\n\n"
            f"Ошибка: {type(e).__name__}"
        )


# =========================================================
# WEBHOOK / WEB APP
# =========================================================

application = (
    Application.builder()
    .token(TELEGRAM_TOKEN)
    .updater(None)
    .build()
)


async def telegram_webhook(request: Request):

    try:

        data = await request.json()

        update = Update.de_json(data, application.bot)

        await application.update_queue.put(update)

        return PlainTextResponse("OK")

    except Exception:

        logger.exception("WEBHOOK ERROR")

        return PlainTextResponse("ERROR", status_code=500)


async def health(request: Request):

    return PlainTextResponse("Ox Alpha AI is alive!")


routes = [
    Route(WEBHOOK_PATH, telegram_webhook, methods=["POST"]),
    Route("/", health, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
]

web_app = Starlette(routes=routes)


# =========================================================
# STARTUP / SHUTDOWN
# =========================================================

async def startup():

    init_db()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("forget", forget))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, txt_handler))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    )

    await application.initialize()
    await application.start()

    webhook_url = RENDER_URL.rstrip("/") + WEBHOOK_PATH

    await application.bot.set_webhook(
        url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
    )

    logger.info("Telegram webhook set: %s", webhook_url)
    logger.info("Ox Alpha AI started with model: %s", GROQ_TEXT_MODEL)
    logger.info("FreeSerp web search: enabled (no API key required)")


async def shutdown():

    try:
        await application.bot.delete_webhook()
    except Exception:
        logger.exception("Failed to delete webhook")

    await application.stop()
    await application.shutdown()


# =========================================================
# RUN
# =========================================================

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
