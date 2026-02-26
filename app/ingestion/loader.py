import os

def load_documents(directory: str):
    documents = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as f:
                documents.append(f.read())
    
    return documents
