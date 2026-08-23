import asyncio
import base64
import logging
import os

import asyncpg
from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, BufferedInputFile
from openai import AsyncOpenAI

# ==================== НАСТРОЙКИ ====================
# Токен бота от @BotFather
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8757713372:AAGDgGmXGvbJvylJeNBEKPxrqsH52gdDKaQ")

# API-ключ ИИ — провайдер TokenRa (tokenra.io)
AI_API_KEY = os.getenv("AI_API_KEY", "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")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1")
AI_MODEL = os.getenv("AI_MODEL", "stealth/ox-alpha")

# Строка подключения к PostgreSQL (даёт Neon/Render при создании базы)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_MrXCLfIg0G9e@ep-wispy-forest-ayyzao6p-pooler.c-5.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

# Имя и характер бота
BOT_NAME = "Ox Alpha"
SYSTEM_PROMPT = (
    f"Тебя зовут {BOT_NAME}. Ты дружелюбный и полезный ИИ-ассистент, "
    "который отвечает кратко, по делу и с лёгким чувством юмора."
)

# Сколько последних сообщений хранить в истории на пользователя
MAX_HISTORY = 10
# =====================================================

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
ai_client = AsyncOpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)

db_pool: asyncpg.Pool | None = None


async def init_db():
    """Подключаемся к БД и создаём таблицу, если её ещё нет."""
    global db_pool
    if not DATABASE_URL:
        raise RuntimeError(
            "Не задан DATABASE_URL — укажите строку подключения к PostgreSQL."
        )
    db_pool = await asyncpg.create_pool(DATABASE_URL, ssl="require")
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_user "
            "ON chat_history (user_id, id)"
        )


async def get_history(user_id: int) -> list[dict]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM chat_history "
            "WHERE user_id = $1 ORDER BY id DESC LIMIT $2",
            user_id,
            MAX_HISTORY * 2,
        )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


async def save_message(user_id: int, role: str, content: str):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_history (user_id, role, content) VALUES ($1, $2, $3)",
            user_id,
            role,
            content,
        )
        # чистим старые сообщения, оставляя только последние MAX_HISTORY*2
        await conn.execute(
            """
            DELETE FROM chat_history
            WHERE user_id = $1 AND id NOT IN (
                SELECT id FROM chat_history
                WHERE user_id = $1
                ORDER BY id DESC LIMIT $2
            )
            """,
            user_id,
            MAX_HISTORY * 2,
        )


async def clear_history(user_id: int):
    if db_pool is None:
        return
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM chat_history WHERE user_id = $1", user_id)


async def ask_ai(user_id: int, user_text: str, image_base64: str | None = None) -> str:
    """Отправляет запрос к ИИ с учётом истории переписки из БД.
    Если передано image_base64 — отправляет вместе с ним изображение
    (запрос к картинке в историю не сохраняется, только текст)."""

    if image_base64:
        user_content = [
            {"type": "text", "text": user_text or "Что на этом изображении?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ]
    else:
        user_content = user_text

    if db_pool is None:
        # БД недоступна — работаем без памяти истории, чтобы бот не падал
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = await ai_client.chat.completions.create(
            model=AI_MODEL, messages=messages, temperature=0.7
        )
        return response.choices[0].message.content

    history = await get_history(user_id)
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_content}]
    )

    response = await ai_client.chat.completions.create(
        model=AI_MODEL,
        messages=messages,
        temperature=0.7,
    )
    answer = response.choices[0].message.content

    # В историю сохраняем только текстовую часть (без самой картинки —
    # чтобы не раздувать базу данных base64-строками)
    await save_message(user_id, "user", user_text or "[изображение]")
    await save_message(user_id, "assistant", answer)

    return answer


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await clear_history(message.from_user.id)
    await message.answer(
        f"Привет! Я {BOT_NAME} 🤖\n"
        "Пиши мне что угодно — отвечу с помощью ИИ.\n\n"
        "Команды:\n"
        "/reset — очистить историю диалога"
    )


@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    await clear_history(message.from_user.id)
    await message.answer("История диалога очищена ✅")


TELEGRAM_MESSAGE_LIMIT = 3500  # с запасом от лимита Telegram в 4096 символов


async def send_answer(message: Message, answer: str):
    """Отправляет ответ пользователю. Если он слишком длинный для обычного
    сообщения — отправляет его файлом .txt."""
    if len(answer) <= TELEGRAM_MESSAGE_LIMIT:
        await message.answer(answer)
        return

    file = BufferedInputFile(answer.encode("utf-8"), filename="response.txt")
    await message.answer_document(
        file,
        caption="Ответ получился слишком длинным для сообщения в Telegram — "
        "прикладываю файлом 📄",
    )


async def process_and_reply(message: Message, user_text: str, image_base64: str | None = None):
    """Общая логика: показать 'печатает', дождаться ответа ИИ (с таймаутом
    и предупреждением о задержке), удалить временное сообщение, ответить.
    Обёрнуто так, что любая непредвиденная ошибка внутри всё равно
    приведёт к какому-то ответу пользователю, а не к молчаливому зависанию."""
    chat_id = message.chat.id
    placeholder = None

    try:
        try:
            await bot.send_chat_action(chat_id, "typing")
        except Exception:
            pass

        placeholder = await message.answer("✍️ Печатаю ответ...")

        task = asyncio.create_task(
            ask_ai(message.from_user.id, user_text, image_base64)
        )
        elapsed = 0
        warned = False

        while not task.done():
            await asyncio.sleep(4)
            elapsed += 4

            try:
                await bot.send_chat_action(chat_id, "typing")
            except Exception:
                pass  # сетевой сбой на "печатает" не должен рушить весь процесс

            if elapsed >= 30 and not warned:
                warned = True
                try:
                    await bot.edit_message_text(
                        "⏳ Модель думает дольше обычного, ещё немного подождите...",
                        chat_id=chat_id,
                        message_id=placeholder.message_id,
                    )
                except Exception:
                    pass

            if elapsed >= 180:
                task.cancel()
                try:
                    await task
                except Exception:
                    pass
                break

        try:
            answer = task.result()
        except asyncio.CancelledError:
            answer = (
                "⚠️ ИИ не ответил за отведённое время (3 мин). "
                "Попробуйте ещё раз или задайте вопрос покороче."
            )
        except Exception as e:
            logging.exception("Ошибка при обращении к ИИ")
            answer = f"⚠️ Произошла ошибка при обращении к ИИ: {e}"

        try:
            await bot.delete_message(chat_id, placeholder.message_id)
        except Exception:
            pass

        await send_answer(message, answer)

    except Exception:
        # Последний рубеж: что бы ни случилось выше, пользователь должен
        # получить хоть какой-то ответ, а не вечное "печатаю ответ..."
        logging.exception("Непредвиденная ошибка в process_and_reply")
        if placeholder is not None:
            try:
                await bot.delete_message(chat_id, placeholder.message_id)
            except Exception:
                pass
        try:
            await message.answer(
                "⚠️ Произошла непредвиденная ошибка. Попробуйте ещё раз."
            )
        except Exception:
            pass


@dp.message(F.text)
async def handle_message(message: Message):
    await process_and_reply(message, message.text)


@dp.message(F.photo)
async def handle_photo(message: Message):
    # Берём фото в максимальном разрешении (последнее в списке)
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)
    image_base64 = base64.b64encode(file_bytes.read()).decode("utf-8")

    # Подпись к фото (если есть) используется как вопрос про изображение
    caption = message.caption or ""
    await process_and_reply(message, caption, image_base64=image_base64)


async def handle_health(request):
    """Крошечный веб-эндпоинт — нужен только чтобы Render считал сервис
    'веб-сервисом' (бесплатный тариф доступен только для них) и чтобы
    UptimeRobot мог пинговать его и не давать засыпать."""
    return web.Response(text="Ox Alpha bot is alive")


async def run_web_server():
    port = int(os.getenv("PORT", "10000"))
    app = web.Application()
    app.router.add_get("/", handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logging.info(f"Веб-сервер для health-пинга запущен на порту {port}")


async def main():
    # Сначала поднимаем веб-сервер — Render должен увидеть открытый порт
    # ещё до того, как мы попытаемся подключиться к базе данных.
    await run_web_server()

    try:
        await init_db()
        logging.info("База данных подключена.")
    except Exception:
        logging.exception(
            "Не удалось подключиться к базе данных. Проверьте DATABASE_URL. "
            "Бот запустится, но история чатов работать не будет."
        )

    logging.info("Бот запускается...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
