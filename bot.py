import os
import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import psycopg

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

MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

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

    print("Database initialized!")


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


def get_history(user_id: int, limit: int = 20):
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
# GEMINI
# =========================================================

async def ask_gemini(
    user_id: int,
    text: str,
    extra_parts=None
):

    history = get_history(user_id)
    memory = get_memory(user_id)

    contents = []

    # История
    for role, content in history:

        if role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=content)
                    ]
                )
            )

        elif role == "model":
            contents.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text=content)
                    ]
                )
            )

    # Текущий запрос
    current_parts = []

    if extra_parts:
        current_parts.extend(extra_parts)

    current_parts.append(
        types.Part.from_text(text=text)
    )

    contents.append(
        types.Content(
            role="user",
            parts=current_parts
        )
    )

    system_prompt = """
Ты дружелюбный ИИ-помощник в Telegram.

Отвечай понятно и по делу.

Если вопрос требует свежей или актуальной информации,
используй Google Search.

Не выдавай непроверенные факты за достоверные.

У тебя есть долговременная память о пользователе.
Используй её только если она действительно относится
к текущему разговору.

Долговременная память пользователя:
""" + (memory if memory else "Память пока пустая.")

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=[google_search_tool],
            system_instruction=system_prompt,
        ),
    )

    return response.text


# =========================================================
# LONG-TERM MEMORY
# =========================================================

async def update_memory(user_id: int, user_text: str):

    old_memory = get_memory(user_id)

    # Не вызываем отдельный запрос для обычной болтовни.
    memory_words = [
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

    if not any(word in user_text.lower() for word in memory_words):
        return

    prompt = f"""
Ты управляешь долговременной памятью Telegram-бота.

Старая память:
{old_memory}

Новое сообщение пользователя:
{user_text}

Обнови память.

Правила:
- сохраняй только полезные долгосрочные факты;
- не сохраняй случайную болтовню;
- не удаляй полезные старые факты без причины;
- пиши кратко;
- обычный текст без пояснений;
- если нового полезного факта нет, верни старую память.
"""

    try:

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL,
            contents=prompt,
        )

        new_memory = response.text.strip()

        if new_memory:
            set_memory(user_id, new_memory)

    except Exception as e:
        print("MEMORY ERROR:", repr(e))


# =========================================================
# TELEGRAM HELPERS
# =========================================================

async def send_long_message(message, text):

    max_length = 4000

    if len(text) <= max_length:
        await message.reply_text(text)
        return

    for i in range(0, len(text), max_length):
        await message.reply_text(
            text[i:i + max_length]
        )


# =========================================================
# COMMANDS
# =========================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.reply_text(
        "Привет! 🧠\n\n"
        "Я ИИ-бот на Gemini.\n"
        "У меня есть интернет через Google Search, "
        "память, история диалога и анализ файлов.\n\n"
        "Просто напиши сообщение или отправь фото/.txt 📸📄"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id

    ensure_user(user_id)
    reset_history(user_id)

    await update.message.reply_text(
        "История текущего диалога очищена 🧹\n"
        "Долговременная память сохранена."
    )


async def forget(update: Update, context: ContextTypes.DEFAULT_TYPE):

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

    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    text = update.message.text

    ensure_user(user_id)

    await update.message.chat.send_action("typing")

    try:

        answer = await ask_gemini(
            user_id=user_id,
            text=text
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

        print("ERROR:", repr(e))

        await update.message.reply_text(
            "Произошла ошибка 😵\n"
            "Попробуй ещё раз."
        )


# =========================================================
# PHOTO
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message or not update.message.photo:
        return

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.chat.send_action("typing")

    try:

        photo = update.message.photo[-1]

        file = await context.bot.get_file(
            photo.file_id
        )

        image_bytes = await file.download_as_bytearray()

        caption = update.message.caption

        if caption:
            prompt = caption
        else:
            prompt = "Проанализируй это изображение."

        image_part = types.Part.from_bytes(
            data=bytes(image_bytes),
            mime_type="image/jpeg"
        )

        answer = await ask_gemini(
            user_id=user_id,
            text=prompt,
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

        print("PHOTO ERROR:", repr(e))

        await update.message.reply_text(
            "Не получилось обработать фото 😵"
        )


# =========================================================
# TXT FILE
# =========================================================

async def txt_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message or not update.message.document:
        return

    document = update.message.document

    filename = document.file_name or ""

    if not filename.lower().endswith(".txt"):
        await update.message.reply_text(
            "Пока я умею читать только .txt файлы 📄"
        )
        return

    user_id = update.effective_user.id

    ensure_user(user_id)

    await update.message.chat.send_action("typing")

    try:

        file = await context.bot.get_file(
            document.file_id
        )

        file_bytes = await file.download_as_bytearray()

        # Ограничиваем размер текста
        if len(file_bytes) > 500_000:
            await update.message.reply_text(
                "Файл слишком большой. "
                "Максимум сейчас — 500 КБ."
            )
            return

        try:
            text_content = bytes(file_bytes).decode("utf-8")

        except UnicodeDecodeError:
            text_content = bytes(file_bytes).decode(
                "cp1251",
                errors="replace"
            )

        caption = update.message.caption or ""

        prompt = f"""
Пользователь отправил TXT-файл.

Имя файла:
{filename}

Комментарий пользователя:
{caption}

Содержимое файла:

{text_content}

Проанализируй файл и ответь на запрос пользователя.
Если отдельного запроса нет — кратко объясни, что находится в файле.
"""

        answer = await ask_gemini(
            user_id=user_id,
            text=prompt
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

        print("TXT ERROR:", repr(e))

        await update.message.reply_text(
            "Не получилось прочитать TXT-файл 😵"
        )


# =========================================================
# RENDER HEALTH CHECK
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        self.send_response(200)
        self.end_headers()

        self.wfile.write(
            b"Bot is alive!"
        )

    def log_message(self, format, *args):
        return


def start_health_server():

    port = int(
        os.environ.get("PORT", 10000)
    )

    server = HTTPServer(
        ("0.0.0.0", port),
        HealthHandler
    )

    print(
        f"Health server started on port {port}"
    )

    server.serve_forever()


# =========================================================
# MAIN
# =========================================================

async def main():

    init_db()

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

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

    print("Telegram bot started!")

    await application.initialize()
    await application.start()
    await application.updater.start_polling()

    await asyncio.Event().wait()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    health_thread = threading.Thread(
        target=start_health_server,
        daemon=True
    )

    health_thread.start()

    asyncio.run(main())
