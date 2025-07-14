from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
import os

# Define folder with synthetic policies
DATA_FOLDER = Path("../data/")

# Helper: Extract structured metadata from top of doc
def extract_metadata(text):
    category = re.search(r"Category:\s*(.*)", text)
    policy_type = re.search(r"Policy Type:\s*(.*)", text)
    brand = re.search(r"Brand:\s*(.*)", text)

    return {
        "category": category.group(1).strip() if category else "Unknown",
        "policy_type": policy_type.group(1).strip() if policy_type else "Unknown",
        "brand": brand.group(1).strip() if brand else "Unknown"
    }

# Chunking strategy
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Final processed docs
processed_docs = []

# Iterate through all .txt files in /data
for file_path in DATA_FOLDER.glob("*.txt"):
    loader = TextLoader(file_path)
    docs = loader.load()

    for doc in docs:
        metadata = extract_metadata(doc.page_content)

        # Chunk the full document
        chunks = splitter.split_text(doc.page_content)

        for i, chunk in enumerate(chunks):
            processed_docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": file_path.name,
                    "chunk_id": i,
                    **metadata
                }
            ))

print(f"✅ Processed {len(processed_docs)} chunks across {len(list(DATA_FOLDER.glob('*.txt')))} documents.")

# Example output
for d in processed_docs[:2]:
    print("\n---")
    print("📄 Chunk:", d.page_content)
    print("🧷 Metadata:", d.metadata)
