
import os

import cohere

# Ключ API лежит в файле .env рядом со скриптом (в сам код ключ не пишем)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    for _line in open(_env_path):
        _line = _line.strip()
        if _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

co = cohere.ClientV2(os.environ["CO_API_KEY"])

# Документы берём из файлов в папке docs/ рядом со скриптом
_docs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
documents = []
for _file_name in sorted(os.listdir(_docs_folder)):
    if _file_name.endswith(".md"):
        with open(os.path.join(_docs_folder, _file_name), encoding="utf-8") as _f:
            documents.append(
                {"id": _file_name, "data": {"title": _file_name, "text": _f.read()}}
            )

print(f"Загружено документов: {len(documents)}")

# Вопрос — такой, на который в этих документах есть ответ
query = "If my bank fails, how much of my money is protected, and are my stocks covered too?"

# Generate the response
response = co.chat(
    model="command-a-plus-05-2026",
    messages=[{"role": "user", "content": query}],
    documents=documents,
)

# Безопасный вывод ответа и хода мысли модели
for content_item in response.message.content:
  # Если это блок рассуждений (если хотите посмотреть, как думает модель)
  if hasattr(content_item, "thinking") and content_item.thinking:
    print(f"[Мысли модели]: {content_item.thinking}\n")

  # Если это итоговый текст ответа
  if hasattr(content_item, "text") and content_item.text:
    print(f"[Ответ]: {content_item.text}\n")

if response.message.citations:
  print("Цитаты:")
  for citation in response.message.citations:
    print(citation, "\n")
