import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=500, overlap=100):
    """Split plain text into overlapping character-based chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for page in pages:
        page_text = page.get("text", "").strip()
        if not page_text:
            continue

        for chunk_index, chunk_text_value in enumerate(
            splitter.split_text(page_text)
        ):
            cleaned_text = chunk_text_value.strip()
            if not cleaned_text:
                continue

            chunks.append(
                {
                    "text": cleaned_text,
                    "source_file": source_file,
                    "source_path": source_path,
                    "source_url": source_url,
                    "page_number": page.get("page_number"),
                    "chunk_index": chunk_index,
                }
            )

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
