# 🧠 SeoWriterBot OpenAI Proxy

Простой Flask-прокси для генерации статей через OpenAI GPT-4. Используется в системе SeoWriterBot как промежуточное звено между сайтом на Bitrix и OpenAI API.

## 🚀 Назначение

Этот сервис:

- Принимает `POST`-запрос с массивом `chunks` (частей данных по жилому комплексу);
- Формирует диалог для OpenAI ChatCompletion API;
- Получает сгенерированную статью от GPT-4;
- Возвращает результат в формате JSON.

## 📦 Структура проекта

```bash
.
├── main.py              # Flask-приложение
├── requirements.txt     # Зависимости (Flask + OpenAI SDK)
└── .gitignore           # Исключения (опционально)
