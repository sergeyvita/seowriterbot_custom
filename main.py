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
        custom_input = data.get("custom_input", "").strip()

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
            temperature=0.8
        )

        content = response.choices[0].message.content.strip()

        def extract_block(tag):
            match = re.search(rf"==={tag}===\s*(.+?)(?=(?:===|$))", content, re.DOTALL)
            return match.group(1).strip() if match else ""

        result = {
            "element_name": extract_block("ELEMENT_NAME"),
            "meta_title": extract_block("META_TITLE"),
            "meta_keywords": extract_block("META_KEYWORDS"),
            "meta_description": extract_block("META_DESCRIPTION"),
            "article": extract_block("ARTICLE")
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
