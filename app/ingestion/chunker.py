import nltk


def chunk_text(text, chunk_size=500, overlap=100):
    """Split plain text into overlapping character-based chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks


def chunk_pdf_pages(
    pages: list[dict],
    source_file: str,
    source_path: str,
    source_url: str,
    chunk_size=500,
    overlap=100,
):
    """Split page text into chunks while preserving source metadata."""
    chunks: list[dict] = []

    for page in pages:
        page_text = page.get("text", "").strip()
        if not page_text:
            continue

        start = 0
        chunk_index = 0

        while start < len(page_text):
            end = start + chunk_size
            chunk_text_value = page_text[start:end].strip()

            if chunk_text_value:
                chunks.append(
                    {
                        "text": chunk_text_value,
                        "source_file": source_file,
                        "source_path": source_path,
                        "source_url": source_url,
                        "page_number": page.get("page_number"),
                        "chunk_index": chunk_index,
                    }
                )

            start += chunk_size - overlap
            chunk_index += 1

    return chunks


def sentence_chunk(text, max_sentences=5, overlap=1):
    """Split text into overlapping sentence-based chunks."""
    sentences = nltk.sent_tokenize(text)
    chunks = []

    step = max_sentences - overlap

    for i in range(0, len(sentences), step):
        chunk = sentences[i:i + max_sentences]
        if not chunk:
            break
        chunks.append(" ".join(chunk))

    return chunks
