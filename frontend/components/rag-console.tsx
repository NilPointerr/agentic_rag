"use client";

import { ChangeEvent, FormEvent, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type IngestResponse = {
  message: string;
  file_name?: string;
  chunks_created: number;
};

type QuerySource = {
  text?: string;
  score?: number;
  source_file?: string;
  source_path?: string;
  source_url?: string;
  page_number?: number;
  page_url?: string;
  title?: string;
  snippet?: string;
  source_type?: string;
};

type QueryResponse = {
  query: string;
  answer: string;
  sources?: QuerySource[];
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ||
  "http://localhost:8000";

const defaultAnswer =
  "Ask a question after ingesting a document and the response will appear here with headings, lists, tables, and source links.";
const defaultIngest =
  "Upload a PDF to the `/ingest` endpoint. Once it is processed, you can query the backend from the workspace on the right.";

function joinUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

function toAbsoluteUrl(url: string) {
  if (/^https?:\/\//i.test(url)) {
    return url;
  }

  return joinUrl(url);
}

function readErrorMessage(payload: unknown, fallback: string) {
  if (
    payload &&
    typeof payload === "object" &&
    "detail" in payload &&
    typeof (payload as { detail?: unknown }).detail === "string"
  ) {
    return (payload as { detail: string }).detail;
  }

  return fallback;
}

function readRequestError(error: unknown, action: string) {
  if (error instanceof TypeError && error.message === "Failed to fetch") {
    return `Cannot reach the backend at ${API_BASE_URL}. Start the FastAPI server and verify the API URL before trying to ${action}.`;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return `Unable to ${action}.`;
}

export function RagConsole() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [ingestLoading, setIngestLoading] = useState(false);
  const [queryLoading, setQueryLoading] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [queryStatus, setQueryStatus] = useState<string | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [ingestResult, setIngestResult] = useState(defaultIngest);
  const [answer, setAnswer] = useState(defaultAnswer);
  const [sources, setSources] = useState<QuerySource[]>([]);

  const backendReady = !ingestError && !queryError;
  const fileLabel = selectedFile ? selectedFile.name : "No PDF selected yet";

  async function handleIngest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setIngestError("Choose a PDF file first.");
      setIngestStatus(null);
      return;
    }

    setIngestLoading(true);
    setIngestError(null);
    setIngestStatus("Uploading PDF and starting ingestion...");
    setIngestResult("Working...");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(joinUrl("/ingest"), {
        method: "POST",
        body: formData
      });

      const payload = (await response.json()) as IngestResponse | { detail?: string };

      if (!response.ok) {
        throw new Error(readErrorMessage(payload, "Ingestion failed."));
      }

      const ingestPayload = payload as IngestResponse;

      setIngestStatus("PDF ingested successfully.");
      setIngestResult(
        JSON.stringify(
          {
            file_name: ingestPayload.file_name ?? selectedFile.name,
            message: ingestPayload.message,
            chunks_created: ingestPayload.chunks_created
          },
          null,
          2
        )
      );
    } catch (error) {
      setIngestStatus(null);
      setIngestError(readRequestError(error, "upload the PDF"));
      setIngestResult("The upload did not complete.");
    } finally {
      setIngestLoading(false);
    }
  }

  async function handleQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setQueryError("Enter a question first.");
      setQueryStatus(null);
      return;
    }

    setQueryLoading(true);
    setQueryError(null);
    setQueryStatus("Generating answer from the RAG agent...");
    setAnswer("Working...");
    setSources([]);

    try {
      const response = await fetch(joinUrl("/query"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ query: trimmedQuestion })
      });

      const payload = (await response.json()) as QueryResponse | { detail?: string };

      if (!response.ok) {
        throw new Error(readErrorMessage(payload, "Query failed."));
      }

      const queryPayload = payload as QueryResponse;

      setQueryStatus("Answer received.");
      setAnswer(queryPayload.answer || "No answer returned.");
      setSources(queryPayload.sources ?? []);
    } catch (error) {
      setQueryStatus(null);
      setQueryError(readRequestError(error, "send the query"));
      setAnswer("The question could not be processed.");
      setSources([]);
    } finally {
      setQueryLoading(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    setSelectedFile(event.target.files?.[0] ?? null);
    setIngestError(null);
    setIngestStatus(null);
  }

  function getSourceHref(source: QuerySource) {
    const href = source.page_url || source.source_url;
    return href ? toAbsoluteUrl(href) : null;
  }

  return (
    <main className="studio-shell">
      <section className="hero-band">
        <div className="hero-copy-block">
          <p className="eyebrow">Agentic RAG Console</p>
          <h1>Sharper workflow. Better reading surface. Same APIs.</h1>
          <p className="hero-copy">
            This interface gives the query experience more space and moves the
            operational controls into a compact rail. It still talks directly to
            <code> /ingest</code> and <code>/query</code>.
          </p>
        </div>

        <div className="hero-status">
          <div className="hero-chip">
            <span className={`status-dot ${backendReady ? "live" : "warning"}`} />
            {backendReady ? "Frontend ready" : "Check backend connection"}
          </div>
          <div className="hero-endpoint">
            <span>API base</span>
            <strong>{API_BASE_URL}</strong>
          </div>
        </div>
      </section>

      <section className="studio-grid">
        <aside className="control-rail">
          <div className="rail-card rail-card-accent">
            <p className="rail-label">Document Intake</p>
            <h2>Upload one PDF to prime the workspace.</h2>
            <p className="rail-copy">
              The ingest flow stores chunked document content in your vector
              pipeline so the answer panel can use it immediately.
            </p>

            <form className="rail-form" onSubmit={handleIngest}>
              <label className="upload-zone">
                <input
                  accept=".pdf,application/pdf"
                  name="file"
                  onChange={handleFileChange}
                  type="file"
                />
                <span className="upload-kicker">Selected file</span>
                <strong>{fileLabel}</strong>
                <em>Choose PDF</em>
              </label>

              <button disabled={ingestLoading} type="submit">
                {ingestLoading ? "Ingesting..." : "Run Ingest"}
              </button>
            </form>

            {ingestStatus ? <p className="notice success">{ingestStatus}</p> : null}
            {ingestError ? <p className="notice error">{ingestError}</p> : null}
          </div>

          <div className="rail-card">
            <p className="rail-label">Pipeline Snapshot</p>
            <pre className="result-block compact">{ingestResult}</pre>
          </div>

          <div className="rail-card">
            <p className="rail-label">Available Routes</p>
            <div className="route-list">
              <div className="route-item">
                <span>POST</span>
                <strong>/ingest</strong>
              </div>
              <div className="route-item">
                <span>POST</span>
                <strong>/query</strong>
              </div>
            </div>
          </div>
        </aside>

        <section className="workspace">
          <div className="workspace-panel composer-panel">
            <div className="panel-topline">
              <p className="rail-label">Query Workspace</p>
              <span className="panel-badge">
                {queryLoading ? "Thinking..." : "Ready"}
              </span>
            </div>

            <div className="workspace-header">
              <div>
                <h2>Ask the system a focused question.</h2>
                <p>
                  Use the larger editor and reading surface for longer answers,
                  tables, citations, and web-backed summaries.
                </p>
              </div>
            </div>

            <form className="query-form" onSubmit={handleQuery}>
              <label className="query-field">
                <span>Question</span>
                <textarea
                  name="query"
                  onChange={(event) => {
                    setQuestion(event.target.value);
                    setQueryError(null);
                    setQueryStatus(null);
                  }}
                  placeholder="Ask about the uploaded PDF, compare topics, or request a structured answer..."
                  rows={8}
                  value={question}
                />
              </label>

              <div className="query-actions">
                <button disabled={queryLoading} type="submit">
                  {queryLoading ? "Sending..." : "Ask Query"}
                </button>
                <p className="helper-copy">
                  Markdown in the response is rendered automatically.
                </p>
              </div>
            </form>

            {queryStatus ? <p className="notice success">{queryStatus}</p> : null}
            {queryError ? <p className="notice error">{queryError}</p> : null}
          </div>

          <div className="workspace-panel answer-panel">
            <div className="panel-topline">
              <p className="rail-label">Answer Surface</p>
              <span className="answer-state">
                {answer === defaultAnswer ? "Waiting for question" : "Latest response"}
              </span>
            </div>

            <div className="markdown-body result-card spacious">
              <ReactMarkdown
                components={{
                  a: ({ node: _node, ...props }) => (
                    <a {...props} rel="noreferrer" target="_blank" />
                  )
                }}
                remarkPlugins={[remarkGfm]}
              >
                {answer}
              </ReactMarkdown>
            </div>

            <div className="sources-panel">
              <div className="panel-topline panel-topline-compact">
                <p className="rail-label">Retrieved Sources</p>
                <span className="answer-state">
                  {sources.length ? `${sources.length} attached` : "No source metadata"}
                </span>
              </div>

              {sources.length ? (
                <div className="sources-list">
                  {sources.map((source, index) => {
                    const href = getSourceHref(source);
                    const title =
                      source.source_file || source.title || `Source ${index + 1}`;

                    return (
                      <article className="source-card" key={`${title}-${index}`}>
                        <div className="source-meta">
                          <strong>{title}</strong>
                          <span>
                            {source.page_number
                              ? `Page ${source.page_number}`
                              : source.source_type === "web"
                                ? "Web result"
                                : "Stored chunk"}
                          </span>
                          {typeof source.score === "number" ? (
                            <span>Score {source.score.toFixed(3)}</span>
                          ) : null}
                        </div>

                        <p className="source-snippet">
                          {source.snippet || source.text || "No preview available."}
                        </p>

                        {href ? (
                          <a href={href} rel="noreferrer" target="_blank">
                            Open source
                          </a>
                        ) : (
                          <span className="source-path">
                            {source.source_path || "Link unavailable"}
                          </span>
                        )}
                      </article>
                    );
                  })}
                </div>
              ) : (
                <p className="helper-copy source-empty">
                  Ingest a PDF again after this update, then ask a question to see file,
                  page, and link metadata here.
                </p>
              )}
            </div>
          </div>
        </section>
      </section>
    </main>
  );
}
