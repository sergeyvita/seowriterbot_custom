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

На основе предоставленного текста создай качественную, увлекательную переработанную статью, ориентированную на аудиторию, которая хочет купить квартиру — для себя, детей, инвестиций, сдачи в аренду, по ипотеке, в рассрочку или за наличные.

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

        # 🔥 Генерация картинки по заголовку
        image_prompt = (
            f"A magical, atmospheric illustration in the exact style of Disney animated concept art, "
            f"inspired by the meaning and content of the following article about real estate in Russia:\n\n"
            f"{article_text[:500]}\n\n"
            f"The illustration should metaphorically and visually convey the core idea of the article — "
            f"whether it's about buying a home, financial analytics, commercial real estate, urban development, "
            f"investments, or economic trends.\n\n"
            f"The composition may include, depending on the article's theme, cities, neighborhoods, and characters "
            f"interacting with the environment: families, experts, builders, dreamers, or buyers. "
            f"The color palette should be warm, golden, and inspiring. Magical lighting is encouraged. "
            f"The visual style should be expressive, painterly, soft, and highly detailed, "
            f"like scenes from Disney movies such as Tangled, Encanto, or Frozen.\n\n"
            f"The characters should have light skin tones and soft European facial features, "
            f"resembling the classical characters from Disney fairy tales. Faces should appear friendly, open, "
            f"and kind — evoking a sense of trust, joy, and emotional connection.\n\n"
            f"The illustration must not resemble a photo or render, but should feel like a warm, storybook painting "
            f"that emotionally resonates with the viewer."
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
