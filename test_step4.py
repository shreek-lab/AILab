# test_step4.py — tests the full retrieval pipeline
from src.loader    import load_documents, chunk_documents
from src.embedder  import get_embedding_model, build_vectorstore, load_vectorstore, vectorstore_exists
from src.retriever import build_hybrid_retriever, retrieve, print_results

# ── Load embedding model ──────────────────────────────────────────────────────
embeddings = get_embedding_model()

# ── Load or build index ───────────────────────────────────────────────────────
if vectorstore_exists():
    print("Index found — loading from disk (no re-embedding)")
    vectorstore, chunks = load_vectorstore(embeddings)
else:
    print("No index found — building from scratch")
    docs        = load_documents("documents")
    chunks      = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks, embeddings)

# ── Build hybrid retriever ────────────────────────────────────────────────────
retriever = build_hybrid_retriever(
    chunks        = chunks,
    vectorstore   = vectorstore,
    bm25_weight   = 0.3,
    vector_weight = 0.7,
    k             = 10
)

# ── Run test queries ──────────────────────────────────────────────────────────
test_queries = [
    "how does binary search work",           # semantic query
    "fibonacci",                             # exact keyword query
    "time complexity O(log n)",              # technical term query
    "what happens when target is not found", # conceptual query
]

for query in test_queries:
    print(f"\n{'='*60}")
    results = retrieve(retriever, query, top_k=3)
    print_results(results, query)