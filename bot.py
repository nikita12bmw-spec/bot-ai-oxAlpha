import asyncio
import logging
import os

import asyncpg
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from openai import AsyncOpenAI

# ==================== НАСТРОЙКИ ====================
# Токен бота от @BotFather
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "ВАШ_TELEGRAM_ТОКЕН")

# API-ключ ИИ — провайдер TokenRa (tokenra.io)
AI_API_KEY = os.getenv("AI_API_KEY", "ВАШ_API_КЛЮЧ_С_TOKENRA")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://tokenra.io/v1")
AI_MODEL = os.getenv("AI_MODEL", "artsdance-2-5-pro-260801")

# Строка подключения к PostgreSQL (даёт Neon/Render при создании базы)
DATABASE_URL = os.getenv("DATABASE_URL", "")

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
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM chat_history WHERE user_id = $1", user_id)


async def ask_ai(user_id: int, user_text: str) -> str:
    """Отправляет запрос к ИИ с учётом истории переписки из БД."""
    history = await get_history(user_id)
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_text}]
    )

    response = await ai_client.chat.completions.create(
        model=AI_MODEL,
        messages=messages,
        temperature=0.7,
    )
    answer = response.choices[0].message.content

    await save_message(user_id, "user", user_text)
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


@dp.message(F.text)
async def handle_message(message: Message):
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        answer = await ask_ai(message.from_user.id, message.text)
    except Exception as e:
        logging.exception("Ошибка при обращении к ИИ")
        answer = f"⚠️ Произошла ошибка при обращении к ИИ: {e}"

    await message.answer(answer)


async def main():
    await init_db()
    logging.info("База данных подключена, бот запускается...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
