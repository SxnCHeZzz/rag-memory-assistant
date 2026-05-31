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
git clone <repo>

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

### Health check

```text
GET /health
```

Ожидаемый результат:

```json
{
  "ollama": true,
  "qdrant": "embedded"
}
```

### Загрузка документа

```powershell
curl -X POST ^
"http://127.0.0.1:8000/documents" ^
-F "file=@demo_documents/fastapi.txt"
```

### Вопрос к системе

```powershell
curl -X POST ^
"http://127.0.0.1:8000/ask" ^
-H "Content-Type: application/json" ^
-d "{\"question\":\"Что такое FastAPI?\",\"session_id\":\"test\"}"
```

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
