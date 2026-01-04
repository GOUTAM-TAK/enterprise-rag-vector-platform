```md
# Enterprise RAG Vector Platform  
**Low-Latency, Real-Time Retrieval-Augmented Generation with Pinecone & WebSocket Streaming**

---

## Overview

**Enterprise RAG Vector Platform** is a production-grade backend system demonstrating how to build a **high-performance Retrieval-Augmented Generation (RAG) platform** with:

- **FastAPI (async-first, lifespan-managed)**
- **Pinecone (vector database)**
- **Local embeddings (SentenceTransformers)**
- **Real-time WebSocket streaming**
- **Dynamic NVIDIA LLM model switching**
- **Strict configuration & enterprise code structure**

The platform is designed to **separate ingestion and retrieval concerns**, preload all heavy dependencies at startup, and deliver **minimal end-to-end latency** for chat-based RAG use cases.

This repository is intended for **learning, experimentation, and portfolio showcase**, following real-world enterprise backend standards.

---

## Key Features

### Ingestion Pipeline (Offline / Async)
- PDF & DOCX document ingestion
- Async-safe document parsing
- Token-aware chunking (SentenceTransformer tokenizer)
- Local embedding generation (CPU/GPU auto-detect)
- Batched upserts to Pinecone
- Namespace isolation per document

### Retrieval Pipeline (Real-Time)
- WebSocket-based chat interface
- Query embedding using preloaded model
- Pinecone similarity search
- Context-aware prompt construction
- Token-by-token streaming response to UI

### Dynamic LLM Model Switching
- NVIDIA LLM APIs
- Multiple models preloaded at application startup
- Runtime model selection per request
- No model reload or warm-up during chat

### Latency Optimization
- All heavy resources initialized once during lifespan
- Shared objects reused via `app.state`
- No per-request model loading
- No per-request Pinecone initialization
- Async execution with background threads for ML workloads

---

## High-Level Architecture

```

Client (UI)
│
│  WebSocket (real-time)
▼
FastAPI WebSocket Endpoint
│
▼
Chat Service Layer
│
├─ Embed query (shared embedder)
├─ Pinecone similarity search
├─ Context aggregation
▼
RAG Chain
│
├─ Prompt construction (query + context)
├─ Streaming LLM inference
▼
WebSocket stream → UI

```

---

## Project Structure

```

enterprise-rag-vector-platform/
│
├── main.py
│
├── app/
│   ├── api/
│   │   ├── router.py
│   │   └── endpoints/
│   │       ├── ingestion.py
│   │       ├── ws_chat.py
│   │       └── health.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── logger.py
│   │
│   ├── rag_core/
│   │   ├── embeddings/
│   │   │   ├── embedder.py
│   │   │   └── tokenizer.py
│   │   ├── ingestion/
│   │   │   ├── loader.py
│   │   │   └── chunker.py
│   │   ├── vectorstore/
│   │   │   └── pinecone_client.py
│   │   ├── prompt/
│   │   │   └── prompt_builder.py
│   │   ├── chain/
│   │   │   └── rag_chain.py
│   │   └── llm/
│   │       ├── nvidia_client.py
│   │       └── llm_registry.py
│   │
│   ├── service/
│   │   ├── ingestion_service.py
│   │   └── chat_service.py
│   │
│   └── utils/
│
├── data/
├── logs/
├── tests/
├── .env
├── requirements.txt
└── README.md

```

---

## Ingestion Pipeline

### Flow
1. Upload document (PDF / DOCX)
2. Async text extraction
3. Token-aware chunking
4. Local embedding generation
5. Batched vector upsert to Pinecone
6. Namespace created per document

### Endpoint
```

POST /api/v1/ingest/pdf

````

### Sample Response
```json
{
  "document_id": "uuid",
  "chunks": 96,
  "status": "INGESTED"
}
````

---

## Real-Time Retrieval (WebSocket)

### WebSocket Endpoint

```
ws://localhost:9666/api/v1/ws/chat
```

### Client → Server Event

```json
{
  "event_type": "chat_request",
  "payload": {
    "query": "Explain Retrieval Augmented Generation",
    "namespace": "<document_id>",
    "model": "meta/llama3-70b-instruct"
  }
}
```

### Server → Client Streaming

```json
{ "event_type": "chat_stream", "token": "Retrieval " }
{ "event_type": "chat_stream", "token": "Augmented " }
{ "event_type": "chat_stream", "token": "Generation " }
{ "event_type": "chat_complete" }
```

---

## Dynamic NVIDIA LLM Model Management

### Environment Configuration

```env
NVIDIA_API_KEY=nvapi-xxxxxxxx
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

NVIDIA_MODELS=meta/llama3-70b-instruct,meta/llama3-8b-instruct
NVIDIA_DEFAULT_MODEL=meta/llama3-70b-instruct
```

### Behavior

* All models are registered and prepared at startup
* UI can fetch available models
* Switching models does not reload or reinitialize clients
* Streaming continues seamlessly

---

## Latency Optimization Strategy

| Optimization Area     | Approach                  |
| --------------------- | ------------------------- |
| Application lifecycle | FastAPI lifespan          |
| Embeddings            | Single in-memory model    |
| Vector DB             | Singleton Pinecone client |
| LLM                   | Preloaded LLM registry    |
| Execution             | Async + thread offloading |
| Transport             | WebSocket streaming       |

---

## Running the Application

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Server

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 9666
```

---

## Health Check

```
GET /api/v1/health
```

---

## Design Principles

* Clear separation of concerns
* Strict configuration validation (Pydantic v2)
* Async-first, non-blocking design
* Streaming over request/response
* Reuse of infrastructure components
* Provider-agnostic LLM architecture

---

## Future Enhancements

* Conversation memory per WebSocket
* Hybrid retrieval (BM25 + vector)
* Model fallback and routing
* Token usage and cost metrics
* Authenticated WebSocket sessions
* Multi-tenant namespace management

---

## Disclaimer

This project is built for **educational and demonstration purposes**, inspired by real-world enterprise GenAI backend architectures.

---

## Author

**Goutam Tak**
Associate Software Engineer
Focus Areas:

* Enterprise Backend Development
* Retrieval-Augmented Generation (RAG)
* GenAI Platform Architecture

```
