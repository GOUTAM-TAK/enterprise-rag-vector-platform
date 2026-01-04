from prometheus_client import Counter, Histogram, Gauge

# -------------------------
# WebSocket Metrics
# -------------------------
ACTIVE_WS_CONNECTIONS = Gauge(
    "rag_active_ws_connections",
    "Number of active WebSocket connections",
)

# -------------------------
# RAG Request Metrics
# -------------------------
CHAT_REQUESTS_TOTAL = Counter(
    "rag_chat_requests_total",
    "Total number of chat requests",
    ["model"],
)

CHAT_ERRORS_TOTAL = Counter(
    "rag_chat_errors_total",
    "Total number of chat errors",
)

# -------------------------
# Latency Metrics
# -------------------------
EMBEDDING_LATENCY = Histogram(
    "rag_embedding_latency_seconds",
    "Embedding generation latency",
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Vector retrieval latency",
)

LLM_FIRST_TOKEN_LATENCY = Histogram(
    "rag_llm_first_token_latency_seconds",
    "Time to first token from LLM",
)

CHAT_TOTAL_LATENCY = Histogram(
    "rag_chat_total_latency_seconds",
    "End-to-end chat latency",
)

# -------------------------
# Context Quality Metrics
# -------------------------
RETRIEVED_CONTEXTS = Histogram(
    "rag_retrieved_contexts_count",
    "Number of contexts retrieved",
    buckets=(0, 1, 2, 3, 5, 8, 13),
)
