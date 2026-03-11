def _simple_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text using a simple character window fallback."""

    if not text:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_text(text: str) -> list[str]:
    """Split a long document string into overlapping chunks for retrieval."""

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        return splitter.split_text(text)
    except Exception:
        return _simple_chunk_text(text, chunk_size=500, chunk_overlap=100)


def sentence_chunk(text: str, max_sentences: int = 5, overlap: int = 1) -> list[str]:
    """Split text by sentence windows with overlap."""

    import nltk

    sentences = nltk.sent_tokenize(text)
    chunks: list[str] = []

    step = max_sentences - overlap
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + max_sentences]
        if not chunk:
            break
        chunks.append(" ".join(chunk))

    return chunks
