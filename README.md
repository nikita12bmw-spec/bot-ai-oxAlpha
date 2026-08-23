# Ox Alpha — Telegram-бот с ИИ (бесплатный сервер + постоянная БД)

## ⚠️ Важный момент про Render

У Render **больше нет бесплатного тарифа для Background Worker**
(теперь минимум $7/мес). Бесплатный тариф остался только у
**Web Service**. Поэтому бот сделан так, чтобы Render воспринимал
его как веб-сервис: внутри крутится крошечный веб-эндпоинт
(просто отвечает "OK"), а рядом в фоне работает сам Telegram-бот.

Побочный эффект: бесплатные веб-сервисы Render засыпают после
15 минут без обращений. Чтобы бот не засыпал — подключаем бесплатный
сервис **UptimeRobot**, который будет пинговать его каждые 5–10 минут
(шаг 4 ниже).

---

## Что понадобится (всё бесплатно, всё с телефона через браузер)

1. **GitHub** — хранить код.
2. **Neon** (neon.tech) — бесплатная PostgreSQL, не истекает.
3. **Render** (render.com) — держать бота запущенным.
4. **UptimeRobot** (uptimerobot.com) — не давать боту засыпать.

---

## Шаг 1. Залить код на GitHub

1. github.com → войти/зарегистрироваться.
2. **+** → **New repository** → назвать `ox-alpha-bot` → **Private** →
   **Create repository**.
3. **Add file → Upload files** → загрузить `bot.py`,
   `requirements.txt`, `render.yaml`, `README.md`.
4. **Commit changes**.

## Шаг 2. Создать базу данных на Neon

1. neon.tech → **Sign up**.
2. Создать проект (можно оставить как есть — у вас он уже создан,
   называется "Оксалфаи").
3. На странице проекта найти **Connection string** (не команду
   `psql`, а именно строку вида `postgresql://...`) и скопировать —
   это и есть `DATABASE_URL`.

## Шаг 3. Развернуть бота на Render как Web Service

1. render.com → **New +** → **Web Service**.
2. Подключить репозиторий `ox-alpha-bot`.
3. Настройки:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python bot.py`
   - **Instance Type**: **Free**
4. В разделе **Environment Variables** добавить:
   - `TELEGRAM_TOKEN` — токен от @BotFather
   - `AI_API_KEY` — ключ с tokenra.io
   - `AI_MODEL` — `artsdance-2-5-pro-260801`
   - `AI_BASE_URL` — `https://tokenra.io/v1`
   - `DATABASE_URL` — строка подключения из Neon
5. **Create Web Service** / **Deploy**. Через 1–2 минуты статус
   станет **Live**.
6. Render выдаст вам публичный URL вида
   `https://ox-alpha-bot.onrender.com` — он понадобится в шаге 4.
7. Проверить: открыть этот URL в браузере — должно показать
   "Ox Alpha bot is alive". Затем написать боту в Telegram `/start`.

## Шаг 4. Не дать боту заснуть (UptimeRobot)

1. uptimerobot.com → **Sign up**.
2. **+ Add New Monitor**.
3. Тип — **HTTP(s)**.
4. URL — тот, что выдал Render (`https://ox-alpha-bot.onrender.com`).
5. Интервал проверки — 5 минут.
6. Сохранить. Готово — теперь бот не будет засыпать.

## Логи и проверка

Render → сервис `ox-alpha-bot` → вкладка **Logs**. Там видно
"База данных подключена, бот запускается..." и все сообщения/ошибки.

## Обновление кода

Загрузить изменённые файлы на GitHub через **Add file → Upload
files** (с заменой) → **Commit**. Render сам пересоберёт и
перезапустит бота.

## Если нужна 100% надёжность без обходных путей

Можно просто взять платный **Starter** тариф для Background Worker
($7/мес) — тогда не нужен ни веб-эндпоинт, ни UptimeRobot, всё
работает "из коробки" без риска что где-то моргнёт при пробуждении.

## Если у вас другой ИИ-провайдер

Поменять `AI_BASE_URL`, `AI_MODEL`, `AI_API_KEY` в переменных
окружения Render — код трогать не нужно, если провайдер совместим
с OpenAI API (как TokenRa).
