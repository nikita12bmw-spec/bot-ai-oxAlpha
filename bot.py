import os
import asyncio
import logging

import psycopg
import uvicorn

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from google import genai
from google.genai import types


# =========================================================
# НАСТРОЙКИ
# =========================================================

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]

RENDER_URL = os.environ.get(
    "RENDER_EXTERNAL_URL",
    "https://bot-ai-oxalpha.onrender.com"
)

PORT = int(os.environ.get("PORT", 10000))

WEBHOOK_PATH = f"/telegram/{WEBHOOK_SECRET}"

MODEL = "gemini-2.5-flash"


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# GEMINI
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

google_search_tool = types.Tool(
    google_search=types.GoogleSearch()
)


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


def save_message(
    user_id: int,
    role: str,
    content: str
):

    with get_db() as conn:

        with conn.cursor() as cur:

            cur.execute("""
                INSERT INTO messages
                (user_id, role, content)
                VALUES (%s, %s, %s)
            """, (
                user_id,
                role,
                content
            ))

        conn.commit()


def get_history(
    user_id: int,
    limit: int = 20
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
                limit
            ))

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


def set_memory(
    user_id: int,
    memory: str
):

    with get_db() as conn:

        with conn.cursor() as cur:

            cur.execute("""
                UPDATE users
                SET memory = %s
                WHERE user_id = %s
            """, (
                memory,
                user_id
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


# =========================================================
# GEMINI REQUEST
# =========================================================

async def telegram_webhook(request):
    print("🔥 WEBHOOK REQUEST RECEIVED", flush=True)

    try:
        data = await request.json()
        print("📦 UPDATE RECEIVED", flush=True)

        update = Update.de_json(data, application.bot)

        if update.message:
            print(
                f"👤 MESSAGE: {update.message.text!r}",
                flush=True
            )

        await application.update_queue.put(update)

        print("✅ UPDATE PUT INTO QUEUE", flush=True)

        return JSONResponse({"ok": True})

    except Exception as e:
        print(f"❌ WEBHOOK ERROR: {type(e).__name__}: {e}", flush=True)
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=500
        )

    history = get_history(user_id)
    memory = get_memory(user_id)

    contents = []

    # История пользователя
    for role, content in history:

        if role == "user":

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=content
                        )
                    ]
                )
            )

        elif role == "model":

            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=content
                        )
                    ]
                )
            )

    # Текущий запрос
    current_parts = []

    if extra_parts:
        current_parts.extend(extra_parts)

    current_parts.append(
        types.Part.from_text(
            text=text
        )
    )

    contents.append(
        types.Content(
            role="user",
            parts=current_parts
        )
    )

    system_prompt = f"""
Ты дружелюбный ИИ-помощник в Telegram.

Отвечай понятно, естественно и по делу.

Если вопрос требует свежей или актуальной информации,
используй Google Search.

Не выдавай непроверенные факты за достоверные.

У тебя есть долговременная память пользователя.
Используй её только тогда, когда она относится
к текущему разговору.

Долговременная память пользователя:

{memory if memory else "Память пока пустая."}
"""

    try:

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[google_search_tool],
                system_instruction=system_prompt,
            ),
        )

        if not response.text:

            raise RuntimeError(
                "Gemini returned an empty response"
            )

        return response.text

    except Exception as e:

        logger.exception(
            "GEMINI ERROR"
        )

        raise e


# =========================================================
# MEMORY
# =========================================================

async def update_memory(
    user_id: int,
    user_text: str
):

    old_memory = get_memory(user_id)

    memory_triggers = [
        "запомни",
        "помни",
        "меня зовут",
        "я люблю",
        "мне нравится",
        "я предпочитаю",
        "мой проект",
        "моя игра",
        "я делаю",
        "я использую",
    ]

    text_lower = user_text.lower()

    if not any(
        word in text_lower
        for word in memory_triggers
    ):
        return

    prompt = f"""
Ты управляешь долговременной памятью пользователя.

Старая память:
{old_memory}

Новое сообщение:
{user_text}

Обнови память пользователя.

Правила:

- сохраняй только полезные долгосрочные факты;
- сохраняй предпочтения и проекты пользователя;
- не сохраняй случайную болтовню;
- не сохраняй пароли, токены или секретные ключи;
- не удаляй полезные старые факты;
- пиши максимально кратко;
- не добавляй пояснения.

Верни только новую память.
"""

    try:

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt
        )

        new_memory = response.text.strip()

        if new_memory:

            set_memory(
                user_id,
                new_memory
            )

            logger.info(
                "Memory updated for user %s",
                user_id
            )

    except Exception:

        logger.exception(
            "MEMORY ERROR"
        )


# =========================================================
# TELEGRAM HELPERS
# =========================================================

async def send_long_message(
    message,
    text: str
):

    max_length = 4000

    if len(text) <= max_length:

        await message.reply_text(text)
        return

    for i in range(
        0,
        len(text),
        max_length
    ):

        await message.reply_text(
            text[i:i + max_length]
        )


# =========================================================
# COMMANDS
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.reply_text(
        "Привет! 🧠\n\n"
        "Я Ox Alpha AI.\n\n"
        "У меня есть:\n"
        "🌐 интернет через Google Search\n"
        "🧠 долговременная память\n"
        "💬 отдельная история пользователей\n"
        "📸 анализ фото\n"
        "📄 чтение TXT\n\n"
        "/reset — очистить историю\n"
        "/forget — забыть память"
    )


async def reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    ensure_user(user_id)

    reset_history(user_id)

    await update.message.reply_text(
        "История очищена 🧹\n"
        "Долговременная память сохранена."
    )


async def forget(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    ensure_user(user_id)

    reset_memory(user_id)

    await update.message.reply_text(
        "Долговременная память очищена 🧠🗑️"
    )


# =========================================================
# TEXT
# =========================================================

async def message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
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

        answer = await ask_gemini(
            user_id,
            text
        )

        save_message(
            user_id,
            "user",
            text
        )

        save_message(
            user_id,
            "model",
            answer
        )

        await update_memory(
            user_id,
            text
        )

        await send_long_message(
            update.message,
            answer
        )

    except Exception as e:

        logger.exception(
            "MESSAGE ERROR"
        )

        await update.message.reply_text(
            "Ошибка при обращении к Gemini 😵\n\n"
            f"Причина: {type(e).__name__}\n"
            "Подробности есть в логах Render."
        )


# =========================================================
# PHOTO
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
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

        prompt = (
            update.message.caption
            or "Проанализируй это изображение."
        )

        image_part = types.Part.from_bytes(
            data=bytes(image_bytes),
            mime_type="image/jpeg"
        )

        answer = await ask_gemini(
            user_id,
            prompt,
            extra_parts=[image_part]
        )

        save_message(
            user_id,
            "user",
            "[Фото] " + prompt
        )

        save_message(
            user_id,
            "model",
            answer
        )

        await send_long_message(
            update.message,
            answer
        )

    except Exception as e:

        logger.exception(
            "PHOTO ERROR"
        )

        await update.message.reply_text(
            "Не получилось обработать фото 😵\n\n"
            f"Ошибка: {type(e).__name__}"
        )


# =========================================================
# TXT
# =========================================================

async def txt_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
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
                errors="replace"
            )

        caption = (
            update.message.caption
            or ""
        )

        prompt = f"""
Пользователь отправил TXT-файл.

Имя файла:
{filename}

Комментарий:
{caption}

Содержимое:

{file_text}

Проанализируй файл и ответь на запрос пользователя.

Если отдельного запроса нет,
кратко объясни содержимое файла.
"""

        answer = await ask_gemini(
            user_id,
            prompt
        )

        save_message(
            user_id,
            "user",
            f"[TXT: {filename}] {caption}"
        )

        save_message(
            user_id,
            "model",
            answer
        )

        await send_long_message(
            update.message,
            answer
        )

    except Exception as e:

        logger.exception(
            "TXT ERROR"
        )

        await update.message.reply_text(
            "Не получилось прочитать TXT 😵\n\n"
            f"Ошибка: {type(e).__name__}"
        )


# =========================================================
# WEBHOOK
# =========================================================

application = (
    Application.builder()
    .token(TELEGRAM_TOKEN)
    .updater(None)
    .build()
)


async def telegram_webhook(
    request: Request
):

    try:

        data = await request.json()

        update = Update.de_json(
            data,
            application.bot
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
            status_code=500
        )


async def health(
    request: Request
):

    return PlainTextResponse(
        "Ox Alpha AI is alive!"
    )


# =========================================================
# WEB APP
# =========================================================

routes = [
    Route(
        WEBHOOK_PATH,
        telegram_webhook,
        methods=["POST"]
    ),

    Route(
        "/",
        health,
        methods=["GET"]
    ),

    Route(
        "/health",
        health,
        methods=["GET"]
    ),
]

web_app = Starlette(
    routes=routes
)


# =========================================================
# STARTUP
# =========================================================

async def startup():

    init_db()

    application.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    application.add_handler(
        CommandHandler(
            "reset",
            reset
        )
    )

    application.add_handler(
        CommandHandler(
            "forget",
            forget
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            txt_handler
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            message_handler
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
        allowed_updates=Update.ALL_TYPES
    )

    logger.info(
        "Telegram webhook set: %s",
        webhook_url
    )

    logger.info(
        "Ox Alpha AI started!"
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


# =========================================================
# RUN
# =========================================================

async def main():

    await startup()

    config = uvicorn.Config(
        web_app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

    server = uvicorn.Server(config)

    try:

        await server.serve()

    finally:

        await shutdown()


if __name__ == "__main__":

    asyncio.run(main())
