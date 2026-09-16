import os
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer

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

=========================

НАСТРОЙКИ

=========================

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

Google Search

google_search_tool = types.Tool(
google_search=types.GoogleSearch()
)

=========================

GOOGLE SEARCH + GEMINI

=========================

async def ask_gemini(text: str) -> str:
response = await asyncio.to_thread(
client.models.generate_content,
model=MODEL,
contents=text,
config=types.GenerateContentConfig(
tools=[google_search_tool],
system_instruction=(
"Ты дружелюбный ИИ-помощник в Telegram. "
"Отвечай понятно и по делу. "
"Если вопрос требует свежей или актуальной информации, "
"используй Google Search. "
"Не утверждай непроверенные факты как достоверные."
),
),
)

return response.text

=========================

TELEGRAM

=========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
await update.message.reply_text(
"Привет! 🧠\n\n"
"Я ИИ-бот на Gemini с доступом к Google Search 🌐\n"
"Просто напиши мне вопрос."
)

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
if not update.message or not update.message.text:
return

text = update.message.text

# Показываем пользователю, что бот думает
await update.message.chat.send_action("typing")

try:
    answer = await ask_gemini(text)

    # Telegram ограничивает размер одного сообщения
    max_length = 4000

    if len(answer) <= max_length:
        await update.message.reply_text(answer)
    else:
        for i in range(0, len(answer), max_length):
            await update.message.reply_text(
                answer[i:i + max_length]
            )

except Exception as e:
    print("ERROR:", repr(e))

    await update.message.reply_text(
        "Произошла ошибка при обращении к Gemini 😵\n"
        "Попробуй ещё раз через несколько секунд."
    )

=========================

RENDER HEALTH CHECK

=========================

class HealthHandler(BaseHTTPRequestHandler):

def do_GET(self):
    self.send_response(200)
    self.end_headers()
    self.wfile.write(b"Bot is alive!")

def log_message(self, format, *args):
    return

def start_health_server():
port = int(os.environ.get("PORT", 10000))

server = HTTPServer(
    ("0.0.0.0", port),
    HealthHandler
)

print(f"Health server started on port {port}")

server.serve_forever()

=========================

ЗАПУСК

=========================

async def main():

application = (
    Application.builder()
    .token(TELEGRAM_TOKEN)
    .build()
)

application.add_handler(
    CommandHandler("start", start)
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

# Бот работает постоянно
await asyncio.Event().wait()

if name == "main":

# Запускаем health-check Render
import threading

health_thread = threading.Thread(
    target=start_health_server,
    daemon=True
)

health_thread.start()

asyncio.run(main())
