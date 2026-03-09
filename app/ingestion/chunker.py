import nltk

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def sentence_chunk(text, max_sentences=5, overlap=1):
    sentences = nltk.sent_tokenize(text)
    chunks = []

    step = max_sentences - overlap

    for i in range(0, len(sentences), step):
        chunk = sentences[i:i + max_sentences]
        if not chunk:
            break
        chunks.append(" ".join(chunk))

    return chunks