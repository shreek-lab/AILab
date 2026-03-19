# test_step5.py — tests the full RAG chain end to end
from src.loader    import load_documents, chunk_documents
from src.embedder  import get_embedding_model, build_vectorstore, \
                          load_vectorstore, vectorstore_exists
from src.retriever import build_hybrid_retriever
from src.chain     import get_llm, build_rag_chain, ask

# ── Setup ─────────────────────────────────────────────────────────────────────
embeddings = get_embedding_model()

if vectorstore_exists():
    vectorstore, chunks = load_vectorstore(embeddings)
else:
    docs        = load_documents("documents")
    chunks      = chunk_documents(docs)
    vectorstore = build_vectorstore(chunks, embeddings)

# CORRECT — unpack properly
retriever = build_hybrid_retriever(chunks, vectorstore)
if isinstance(retriever, tuple):
    retriever = retriever[0]
llm       = get_llm()
chain     = build_rag_chain(llm)

# ── Single question ───────────────────────────────────────────────────────────
answer, sources = ask(
    query        = "How does binary search work and what is its time complexity?",
    retriever    = retriever,
    chain        = chain,
    show_sources = True
)

# ── Multi-turn: follow-up question ────────────────────────────────────────────
history = [("How does binary search work?", answer)]

answer2, _ = ask(
    query        = "Can you show me what happens when the target is not found?",
    retriever    = retriever,
    chain        = chain,
    chat_history = history,   # passes previous turn
    show_sources = False
)