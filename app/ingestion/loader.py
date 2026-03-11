import os


def load_documents(directory: str) -> list[str]:
    """Load plain-text documents from a directory."""

    documents: list[str] = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append(file.read())
    return documents
