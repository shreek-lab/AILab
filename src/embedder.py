# =============================================================================
# src/embedder.py
# Embeds document chunks using a local sentence-transformers model
# Saves and loads FAISS index to/from disk
# No API key needed — runs 100% locally
# =============================================================================

import pickle
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # local model, ~90MB download once
VECTORSTORE_DIR   = "vectorstore"         # folder to save FAISS index
FAISS_INDEX_FILE  = "faiss_index"         # FAISS saves 2 files with this name
CHUNKS_FILE       = "chunks.pkl"          # we also save raw chunks for BM25


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads the local sentence-transformers embedding model.
    First call downloads the model (~90MB) — subsequent calls load from cache.

    Model: all-MiniLM-L6-v2
      - 384 dimensions
      - 256 max tokens
      - Fast, lightweight, good quality for code + technical text
    """
    console.print(f"\n[bold]Loading embedding model:[/bold] {EMBEDDING_MODEL}")
    console.print("[dim]First run downloads ~90MB — subsequent runs load from cache[/dim]")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have a GPU
        encode_kwargs={
            "normalize_embeddings": True,  # normalise so cosine = dot product
            "batch_size": 32,              # embed 32 chunks at a time
        }
    )

    console.print(f"[green]Embedding model ready[/green]")
    return embeddings


def build_vectorstore(chunks: List[Document],
                      embeddings: HuggingFaceEmbeddings,
                      vectorstore_dir: str = VECTORSTORE_DIR) -> FAISS:
    """
    Embeds all chunks and builds a FAISS index.
    Saves both the FAISS index and raw chunks to disk.

    Why save raw chunks separately?
      - FAISS stores vectors only
      - BM25 (Step 4) needs the raw text
      - We save chunks.pkl so BM25 can be rebuilt from disk too

    Args:
        chunks         : list of Document objects from chunk_documents()
        embeddings     : embedding model from get_embedding_model()
        vectorstore_dir: folder to save index files

    Returns:
        FAISS vectorstore object ready for similarity search
    """
    if not chunks:
        console.print("[red]No chunks to embed. Run loader first.[/red]")
        return None

    save_path = Path(vectorstore_dir)
    save_path.mkdir(exist_ok=True)

    console.print(f"\n[bold]Embedding {len(chunks)} chunks...[/bold]")
    console.print("[dim]This may take 1-2 minutes on first run[/dim]\n")

    # ── Embed all chunks and build FAISS index ────────────────────────────────
    # FAISS.from_documents():
    #   1. calls embeddings.embed_documents() on all chunk texts
    #   2. gets back a list of 384-dim float vectors
    #   3. builds an HNSW index over those vectors in memory
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding...", total=len(chunks))

        # Process in batches so progress bar updates
        batch_size = 32
        all_texts    = [c.page_content for c in chunks]
        all_metadata = [c.metadata     for c in chunks]

        # Build from first batch then add rest
        first_batch_docs = chunks[:batch_size]
        vectorstore = FAISS.from_documents(first_batch_docs, embeddings)
        progress.advance(task, min(batch_size, len(chunks)))

        # Add remaining batches
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_documents(batch)
            progress.advance(task, len(batch))

    # ── Save FAISS index to disk ──────────────────────────────────────────────
    # Saves two files:
    #   vectorstore/faiss_index.faiss  ← the actual vector index
    #   vectorstore/faiss_index.pkl    ← docstore (maps IDs to Documents)
    vectorstore.save_local(
        folder_path=str(save_path),
        index_name=FAISS_INDEX_FILE
    )
    console.print(f"\n[green]FAISS index saved →[/green] {save_path}/{FAISS_INDEX_FILE}.faiss")

    # ── Save raw chunks for BM25 ──────────────────────────────────────────────
    chunks_path = save_path / CHUNKS_FILE
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    console.print(f"[green]Raw chunks saved →[/green] {chunks_path}")

    # ── Print stats ───────────────────────────────────────────────────────────
    _print_embedding_stats(chunks, vectorstore_dir)

    return vectorstore


def load_vectorstore(embeddings: HuggingFaceEmbeddings,
                     vectorstore_dir: str = VECTORSTORE_DIR) -> Tuple[FAISS, List[Document]]:
    """
    Loads a previously saved FAISS index from disk.
    Also loads the raw chunks needed by BM25.

    Returns:
        (vectorstore, chunks) tuple
        or (None, []) if no saved index found
    """
    save_path   = Path(vectorstore_dir)
    index_path  = save_path / f"{FAISS_INDEX_FILE}.faiss"
    chunks_path = save_path / CHUNKS_FILE

    # Check if saved index exists
    if not index_path.exists():
        console.print("[yellow]No saved index found.[/yellow] Run indexing first.")
        return None, []

    console.print(f"\n[bold]Loading saved index from:[/bold] {vectorstore_dir}/")

    # Load FAISS index
    # allow_dangerous_deserialization=True is required by LangChain
    # safe here because WE created this file ourselves
    vectorstore = FAISS.load_local(
        folder_path=str(save_path),
        embeddings=embeddings,
        index_name=FAISS_INDEX_FILE,
        allow_dangerous_deserialization=True
    )
    console.print(f"[green]FAISS index loaded[/green]")

    # Load raw chunks
    chunks = []
    if chunks_path.exists():
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        console.print(f"[green]Loaded {len(chunks)} chunks for BM25[/green]")
    else:
        console.print("[yellow]chunks.pkl not found — BM25 will be unavailable[/yellow]")

    return vectorstore, chunks


def vectorstore_exists(vectorstore_dir: str = VECTORSTORE_DIR) -> bool:
    """Quick check — has the index been built yet?"""
    return (Path(vectorstore_dir) / f"{FAISS_INDEX_FILE}.faiss").exists()


def _print_embedding_stats(chunks: List[Document], vectorstore_dir: str):
    """Prints embedding statistics."""
    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chars   = total_chars // len(chunks) if chunks else 0

    console.print(f"""
[bold cyan]Embedding complete[/bold cyan]
  Total chunks embedded : [green]{len(chunks)}[/green]
  Total characters      : [green]{total_chars:,}[/green]
  Average chunk size    : [green]{avg_chars} chars[/green]
  Index saved to        : [green]{vectorstore_dir}/[/green]
""")