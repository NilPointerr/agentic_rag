# Agentic RAG Pipeline Review

## Scope
Reviewed the current pipeline implementation under `app/` for ingestion, retrieval, agent orchestration, API quality, reliability, and production readiness.

## Critical Findings (Fix First)

1. Unbounded agent loop can hang requests
- File: [rag_agent.py](/home/dev62/Documents/agentic_rag/app/agent/rag_agent.py#L48)
- Issue: `while True` has no max tool-call/iteration guard.
- Risk: Infinite loop on repeated tool calls, request timeout, cost spike.
- Improve:
  - Add `MAX_AGENT_STEPS` (for example 4-6).
  - If exceeded, return a safe fallback answer with diagnostics logged.

2. Ingestion endpoint converts client errors into 500
- File: [routes.py](/home/dev62/Documents/agentic_rag/app/api/routes.py#L46)
- Issue: broad `except Exception` wraps everything, including `HTTPException`.
- Risk: Wrong status codes and poor API behavior for users.
- Improve:
  - Add explicit `except HTTPException: raise`.
  - Keep generic exception handling for unexpected failures only.

3. Query endpoint is sync and blocks event loop with network/LLM work
- File: [routes.py](/home/dev62/Documents/agentic_rag/app/api/routes.py#L90)
- Issue: endpoint is synchronous while performing external calls.
- Risk: Lower throughput and latency under concurrent load.
- Improve:
  - Make `/query` async and offload blocking calls to threadpool or adopt async clients.

4. No namespace/document identity in vector records
- File: [embedder.py](/home/dev62/Documents/agentic_rag/app/ingestion/embedder.py#L17)
- Issue: only stores raw chunk text, random id, no source metadata.
- Risk: cannot filter by doc, deduplicate, trace citations, or delete by document.
- Improve:
  - Store metadata: `source`, `doc_id`, `chunk_id`, `page`, `ingested_at`.
  - Use deterministic IDs like `doc_id:chunk_idx`.

## High-Impact Quality Improvements

5. Retrieval ignores configured `TOP_K` and uses average score only
- File: [retriever.py](/home/dev62/Documents/agentic_rag/app/retriever/retriever.py#L6)
- Issue:
  - hardcoded default `top_k=3` instead of config.
  - confidence based on average score across top-k.
- Risk: weak relevance gating; one bad hit can distort decision.
- Improve:
  - default to `settings.TOP_K`.
  - use top-1 score or weighted score.
  - optionally apply minimum-score filter per match before context assembly.

6. Prompt-only tool ordering is brittle
- File: [rag_agent.py](/home/dev62/Documents/agentic_rag/app/agent/rag_agent.py#L13)
- Issue: relies on model compliance for “vector first, then web”.
- Risk: model can violate policy in edge cases.
- Improve:
  - Enforce orchestration in code: call vector search first, then conditionally web search.
  - Keep LLM for reasoning/answer synthesis, not control flow.

7. Logging may leak sensitive content and increase cost
- File: [groq_client.py](/home/dev62/Documents/agentic_rag/app/llm/groq_client.py#L8)
- Issue: logs full `messages` payload.
- Risk: PII leakage and large logs.
- Improve:
  - log metadata only (`message_count`, token estimates, tool calls, latency).
  - redact user content by default.

8. Index creation at import time
- File: [pinecone_client.py](/home/dev62/Documents/agentic_rag/app/vectorstore/pinecone_client.py#L8)
- Issue: index list/create runs during module import.
- Risk: slower cold start, hard-to-test side effects.
- Improve:
  - move to explicit startup/init function.
  - fail fast with clear startup error if dependencies unavailable.

9. Duplicate imports/router declarations and dead model in routes
- File: [routes.py](/home/dev62/Documents/agentic_rag/app/api/routes.py#L1)
- Issue: repeated imports, `router` initialized twice, unused `IngestRequest`.
- Risk: maintainability and readability degradation.
- Improve:
  - clean imports, single router declaration, remove dead request model.

## Reliability, Security, and Ops Gaps

10. Upload handling lacks safeguards
- File: [routes.py](/home/dev62/Documents/agentic_rag/app/api/routes.py#L55)
- Gaps:
  - filename not sanitized,
  - no size limit,
  - no content-type verification beyond extension.
- Improve:
  - sanitize filenames (`Path(file.filename).name`),
  - enforce max size and MIME/content sniffing,
  - reject encrypted/corrupt PDFs with clear error.

11. No citation-ready response format
- Files: [retriever.py](/home/dev62/Documents/agentic_rag/app/retriever/retriever.py#L22), [rag_agent.py](/home/dev62/Documents/agentic_rag/app/agent/rag_agent.py#L73)
- Issue: only plain text context is returned to LLM.
- Improve:
  - return structured passages with source metadata.
  - expose citations in API response.

12. No test coverage for core path
- Issue: no tests for chunking, retrieval threshold behavior, tool routing, endpoint contract.
- Improve:
  - add unit tests for chunking/retrieval,
  - integration tests for `/ingest` and `/query` with mocked Pinecone/Groq/DDGS.

13. Missing runtime controls
- Gaps: no request timeout controls, no retry/backoff strategy, no rate limiting.
- Improve:
  - set explicit LLM/search timeouts,
  - retry transient failures with backoff,
  - add per-IP or token-based rate limits.

## Recommended Refactor (Pragmatic)

1. Split orchestration layers:
- `RetrieverService` (vector query, rerank, context packaging)
- `ToolService` (web search wrapper with timeout/retry)
- `AnswerService` (LLM synthesis from structured evidence)
- `AgentOrchestrator` (deterministic flow + fallback policy)

2. Define data contracts:
- `RetrievedPassage`, `ToolResult`, `AnswerWithCitations`.

3. Add observability:
- request id, per-step latency, token usage, retrieval hit-rate, web fallback rate.

## 30-60-90 Implementation Plan

### Next 3 days (stability)
- Add max agent steps and robust exception taxonomy in API.
- Clean `routes.py` duplication/dead code.
- Respect `TOP_K` config and improve retrieval score gating.

### Next 1-2 weeks (quality)
- Add metadata-rich vector schema + deterministic IDs.
- Add citation-capable response format.
- Add test suite for critical paths with mocks.

### Next 1 month (production readiness)
- Introduce reranking (cross-encoder) for better precision.
- Add async-safe clients/timeouts/retries/rate limits.
- Add dashboard metrics for retrieval and answer quality.

## Quick Wins Summary
- Highest ROI now: bounded agent loop, proper HTTP error handling, metadata-rich storage, deterministic orchestration.
- These changes reduce outages/hallucinations and make future improvements (citations, evals, reranking) much easier.
