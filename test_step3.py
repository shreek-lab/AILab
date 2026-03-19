# test_step3.py
from src.loader   import load_documents, chunk_documents
from src.embedder import get_embedding_model, build_vectorstore, load_vectorstore

print("=" * 50)
print("STEP 2 — Loading and chunking")
print("=" * 50)
docs   = load_documents("documents")
chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=100)

print("=" * 50)
print("STEP 3 — Embedding and saving FAISS index")
print("=" * 50)
embeddings  = get_embedding_model()
vectorstore = build_vectorstore(chunks, embeddings)

print("=" * 50)
print("Testing similarity search on saved index")
print("=" * 50)

# Reload from disk (simulates restarting the app)
vs, loaded_chunks = load_vectorstore(embeddings)

# Run a test query
query   = "how does binary search work"
results = vs.similarity_search(query, k=2)

print(f"\nQuery: '{query}'")
print(f"Top {len(results)} results:\n")
for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  Source : {doc.metadata['filename']}")
    print(f"  Chunk  : {doc.metadata['chunk_id']}")
    print(f"  Preview: {doc.page_content[:150]}...")
    print()