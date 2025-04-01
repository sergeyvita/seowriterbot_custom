from flask import Flask, request, jsonify
from openai import OpenAI
import os
import re

app = Flask(__name__)

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        custom_input = data.get("custom_input", "")
        chunks = data.get("chunks", [])

        if not custom_input and chunks:
            custom_input = "\n".join([chunk.strip() for chunk in chunks if isinstance(chunk, str)])

        custom_input = custom_input.strip()
        
        if not custom_input:
            return jsonify({"error": "Нет текста для генерации"}), 400

        full_prompt = f"""
Ты — профессиональный маркетинговый копирайтер. 

На основе предоставленного текста создай качественную, увлекательную переработанную статью, ориентированную на аудиторию, которая хочет купить квартиру — для себя, детей, инвестиций, сдачи в аренду, по ипотеке, в рассрочку или за наличные. При упоминании в тексте про банковские ставки, то учитывай, что если говорится об ипотечных ставках свыше 15% годовых, то это высокие ставки. Сейчас ЦБ РФ держит ключевую ставку в 21% это высокая ключевая ставка, рынок ожидает снижения. Семейная льготная ипотека на данный момент составляет 6%.

Статья должна быть **живой и убедительно-позитивной**, с ярким вступлением, аргументами, примерами и фактами (если уместно). Переработай текст, усилив аргументацию, добавив стиль, структуру и SEO.

**Требования:**
- Объём — не менее 3900 символов
- Без эмодзи
- Все важные мысли выделяй жирным или курсивом (Markdown)
- Включай высокочастотные ключевые слова по теме (покупка квартиры, ипотека, новостройки и пр.)
- В конце статьи добавь **ссылку на сайт Ассоциации Застройщиков** — https://ap-r.ru
- Не используй скобки, markdown-ссылки. Можно использовать формулировку “Официальный сайт Ассоциации застройщиков”

Также сгенерируй:

2. ✏️ *Заголовок элемента* — до 100 символов

3. 📈 *META TITLE* — до 60 символов

4. 🔑 *META KEYWORDS* — не менее 25 ключевых фраз через запятую

5. 📝 *META DESCRIPTION* — до 500 символов, не должно обрываться

Исходный текст:
{custom_input}

Ответ верни строго в формате:

===ELEMENT_NAME===
{{заголовок элемента}}

===META_TITLE===
{{мета тайтл}}

===META_KEYWORDS===
{{ключевые слова}}

===META_DESCRIPTION===
{{мета описание}}

===ARTICLE===
{{полный текст статьи}}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=1
        )

        content = response.choices[0].message.content.strip()

        def extract_block(tag):
            match = re.search(rf"==={tag}===\s*(.+?)(?=(?:===|$))", content, re.DOTALL)
            return match.group(1).strip() if match else ""

        # Извлекаем данные из ответа
        element_name = extract_block("ELEMENT_NAME")
        meta_title = extract_block("META_TITLE")
        meta_keywords = extract_block("META_KEYWORDS")
        meta_description = extract_block("META_DESCRIPTION")
        article_text = extract_block("ARTICLE")

            # Удаляем кавычки у ЖК и прочие нестабильные элементы
        cleaned_article_text = re.sub(r"[«»\"“”]", "", article_text)
            
            # Краткое смысловое описание (в будущем можно заменить автосаммари)
        article_summary = (
            "This article helps readers choose the best residential complex in Russia based on lifestyle, budget, "
            "and preferences — from cozy family options to premium urban living. "
            "It explains which type of housing fits different goals: comfort, investment, location, or seaside life."
        )
            # 🔥 Генерация картинки по статье
        image_prompt = (
            f"A clean, photo-realistic image inspired by a real estate article in Russia."
            f"The image should show a warm, modern residential area in a Russian city — with new apartment buildings, trees, and families outside. "
            f"A peaceful, sunny day. Children playing, couples walking, and families enjoying time together in a courtyard. "
            f"The scene should feel safe, optimistic, and modern, like a real estate advertisement for urban living. "
            f"The mood is positive, welcoming, and filled with light."
            
        )    

        image_url = None
        try:
            image_response = client.images.generate(
                model="dall-e-3",
                prompt=image_prompt,
                n=1,
                size="1792x1024",
                quality="hd",
                style="natural"
            )
            image_url = image_response.data[0].url
        except Exception as e:
            print("Ошибка генерации изображения:", e)

        # Возвращаем всё вместе
        return jsonify({
            "element_name": element_name,
            "meta_title": meta_title,
            "meta_keywords": meta_keywords,
            "meta_description": meta_description,
            "article": article_text,
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
