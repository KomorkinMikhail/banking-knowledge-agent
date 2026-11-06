
import math
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

# --- Шаг 3: модель получает не файлы целиком, а только куски, близкие к вопросу ---

# 1) Читаем файлы из docs/ и режем каждый на куски по абзацам
_docs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
pieces = []  # каждый кусок: id, title (имя файла), text
for _file_name in sorted(os.listdir(_docs_folder)):
    if _file_name.endswith(".md"):
        with open(os.path.join(_docs_folder, _file_name), encoding="utf-8") as _f:
            _text = _f.read()
        for _i, _para in enumerate(_text.split("\n\n")):
            _para = _para.strip()
            if len(_para) > 40:  # совсем короткие обрывки (заголовки) пропускаем
                pieces.append({"id": f"{_file_name}#{_i}", "title": _file_name, "text": _para})

print(f"Кусков всего: {len(pieces)}")

# Вопрос — тот же, что и на шаге 2: ответ проверяем по документу про CDIC
query = "If my bank fails, how much of my money is protected, and are my stocks covered too?"


# 2) Превращаем тексты в числа (embeddings): похожий смысл — похожие числа
def _embed(texts, input_type):
    _vectors = []
    for _start in range(0, len(texts), 96):  # API принимает не больше 96 текстов за раз
        _resp = co.embed(
            model="embed-v4.0",
            texts=texts[_start : _start + 96],
            input_type=input_type,
            embedding_types=["float"],
        )
        _emb = _resp.embeddings
        _vectors.extend(getattr(_emb, "float", None) or getattr(_emb, "float_"))
    return _vectors


_piece_vectors = _embed([p["text"] for p in pieces], "search_document")
_query_vector = _embed([query], "search_query")[0]


# 3) Близость вопроса к куску — обычная арифметика (косинус между списками чисел)
def _cosine(a, b):
    _dot = sum(x * y for x, y in zip(a, b))
    _na = math.sqrt(sum(x * x for x in a))
    _nb = math.sqrt(sum(y * y for y in b))
    return _dot / (_na * _nb)


_scored = sorted(
    ((_cosine(_query_vector, _vec), _piece) for _vec, _piece in zip(_piece_vectors, pieces)),
    key=lambda t: t[0],
    reverse=True,
)

TOP = 5
print(f"Выбрано кусков: {TOP}")
for _score, _piece in _scored[:TOP]:
    print(f"  [{_score:.2f}] {_piece['title']}: {_piece['text'][:70]}...")

# В модель отправляем только выбранные куски
documents = [
    {"id": _p["id"], "data": {"title": _p["title"], "text": _p["text"]}}
    for _s, _p in _scored[:TOP]
]

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
