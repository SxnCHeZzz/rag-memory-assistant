# RAG Memory Assistant

Локальная система вопросов и ответов на основе Retrieval-Augmented Generation (RAG) с поддержкой долговременной памяти пользователя.

Проект реализован в рамках курсовой работы и демонстрирует построение локального RAG-пайплайна без использования внешних LLM API.

## Возможности

* загрузка документов (`.txt`, `.pdf`);
* автоматический чанкинг и индексация документов;
* гибридный retrieval (dense embeddings + BM25);
* долговременная память пользователя;
* поддержка контекста текущего диалога;
* локальная генерация ответов через Ollama;
* debug endpoints для проверки retrieval.

---

## Архитектура

```text
Client
   │
   ▼

FastAPI API

   │
   ├── Document Retrieval
   │      ├── Embeddings
   │      ├── BM25
   │      └── Qdrant
   │
   ├── Memory Service
   │
   └── Ollama (Qwen2.5 7B)

```

### Потоки данных

**Документы**

```text
upload
  ↓
chunking
  ↓
embeddings
  ↓
Qdrant indexing
```

**Вопрос пользователя**

```text
question
   ↓
retrieval
   ↓
prompt assembly
   ↓
LLM generation
```

**Память**

```text
message
   ↓
fact extraction
   ↓
memory storage
   ↓
memory retrieval
   ↓
prompt
```

---

## Технологический стек

| Компонент           | Технология            |
| ------------------- | --------------------- |
| API                 | FastAPI               |
| Векторное хранилище | Qdrant Embedded       |
| LLM                 | Ollama + Qwen2.5 7B   |
| Embeddings          | sentence-transformers |
| Retrieval           | Dense + BM25 + RRF    |
| Configuration       | Pydantic Settings     |

---

## Требования

Моя конфигурация:

* Python 3.11+
* 32 GB RAM
* GPU с 8 GB VRAM 
* установленный Ollama

---

## Установка

### 1. Клонирование проекта

```bash
git clone https://github.com/SxnCHeZzz/rag-memory-assistant.git

cd rag-memory-assistant
```

### 2. Создание окружения

```bash
python -m venv .venv

.venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Установка модели

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

---

## Запуск

```powershell
.\start.ps1
```

После запуска API будет доступно:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Проверка работоспособности

> **Примечание для Windows:** в PowerShell 5.1 команды `curl.exe` с JSON могут не работать из-за особенностей обработки кавычек. Рекомендуется использовать **PowerShell 7** (`pwsh`) или экранировать кавычки: `\"question\"`.

### 1. Health check

```powershell
curl.exe http://127.0.0.1:8000/health
```

Ожидаемый результат: `{"ollama":true,"qdrant":"embedded"}`

### 2. Загрузка документа

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents" -F "file=@demo_documents/fastapi.txt"
```

Ожидаемый результат: `{"filename":"fastapi.txt","chunks_count":4}`

### 3. Проверка retrieval

```powershell
curl.exe -X POST "http://127.0.0.1:8000/debug/retrieval" `
  -H "Content-Type: application/json" `
  -d '{"question":"Какие преимущества FastAPI?","top_k":5}'
```

Ожидаемый результат: `documents` содержит чанки из `fastapi.txt`.

### 4. Вопрос к системе

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ask" `
  -H "Content-Type: application/json" `
  -d '{"question":"Какие преимущества FastAPI?","session_id":"test","top_k":5}'
```

Ожидаемый результат: полный ответ с `sources: ["fastapi.txt"]`.

### 5. Создание памяти

```powershell
curl.exe -X POST "http://127.0.0.1:8000/memory" `
  -H "Content-Type: application/json" `
  -d '{"user_id":"test","category":"preference","text":"Люблю краткие ответы"}'
```

Ожидаемый результат: запись с `id`, `user_id`, `category`, `text`.

### 6. Проверка памяти

```powershell
curl.exe -X POST "http://127.0.0.1:8000/ask" `
  -H "Content-Type: application/json" `
  -d '{"question":"Расскажи о FastAPI","session_id":"test","top_k":5}'
```

Ожидаемый результат: `memories_used` содержит `"Люблю краткие ответы"`, ответ короткий.

## Дополнительные сценарии тестирования

Ниже приведены дополнительные сценарии проверки пользовательской памяти и изоляции данных между сессиями.

### 7. Проверка персональных фактов (имя и место учебы)

Проверим сохранение нескольких связанных фактов пользователя.

```powershell
# сохраняем информацию о пользователе

$body=@{
user_id="test"
category="fact"
text="Меня зовут Александр и я учусь в Южном федеральном университете"
}|ConvertTo-Json -Compress

curl.exe -X POST "http://127.0.0.1:8000/memory" `
-H "Content-Type: application/json" `
-d $body


# задаем вопрос

$body=@{
question="Как меня зовут и где я учусь?"
session_id="test"
}|ConvertTo-Json -Compress

curl.exe -X POST "http://127.0.0.1:8000/ask" `
-H "Content-Type: application/json" `
-d $body
```

Ожидаемый результат:

* память пользователя попадает в `memories_used`;
* система использует сохраненный факт при генерации ответа;
* ответ содержит имя пользователя и место обучения.

---

### 8. Проверка семантического поиска по памяти

Проверим, может ли retrieval найти релевантную память при изменении формулировки вопроса.

```powershell
$body=@{
question="В каком образовательном учреждении я учусь?"
session_id="test"
}|ConvertTo-Json -Compress

curl.exe -X POST "http://127.0.0.1:8000/ask" `
-H "Content-Type: application/json" `
-d $body
```

Ожидаемый результат:

* retrieval использует векторный поиск памяти;
* в `memories_used` появляется сохраненная запись;
* ответ содержит информацию об университете.

---

### 9. Проверка изоляции пользовательских данных

Проверим, что память одного пользователя недоступна другим сессиям.

```powershell
$body=@{
question="Как меня зовут и где я учусь?"
session_id="ivan"
}|ConvertTo-Json -Compress

curl.exe -X POST "http://127.0.0.1:8000/ask" `
-H "Content-Type: application/json" `
-d $body
```

Ожидаемый результат:

* `sources` и `memories_used` пустые;
* система не использует данные пользователя `test`;
* ответ сообщает об отсутствии необходимого контекста.

---

Эти тесты позволяют проверить:

* работу долговременной памяти;
* семантический retrieval пользовательских фактов;
* изоляцию пользовательских данных между сессиями.

```
```


### Альтернатива: Swagger UI

Откройте `http://127.0.0.1:8000/docs` — интерактивная документация API. Нажмите **Try it out** на любом endpoint, заполните параметры и нажмите **Execute**.

---

## Основные endpoints

| Endpoint                        | Назначение         |
| ------------------------------- | ------------------ |
| POST `/ask`                     | задать вопрос      |
| POST `/documents`               | загрузить документ |
| POST `/memory`                  | добавить память    |
| GET `/memory`                   | получить память    |
| DELETE `/memory/{id}`           | удалить память     |
| POST `/conversation/{id}/clear` | очистить сессию    |
| POST `/debug/retrieval`         | inspect retrieval  |
| GET `/health`                   | состояние системы  |

---

## Ограничения

* система рассчитана на локальный запуск;
* embedded Qdrant не предназначен для высокой нагрузки;
* память пользователя извлекается rule-based подходом;
* диалоговая память хранится в оперативной памяти;
* производительность зависит от локального оборудования.

---

## Структура проекта

```text
project/

├── app/
├── data/
├── demo_documents/
├── scripts/
├── tests/
├── start.ps1
├── requirements.txt
└── README.md
```

---

## Статус проекта

Проект реализован как учебный инженерный прототип для исследования Retrieval-Augmented Generation и пользовательской памяти в локальных диалоговых системах.
