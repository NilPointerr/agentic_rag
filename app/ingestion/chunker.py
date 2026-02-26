import nltk

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks


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