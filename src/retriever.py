# =============================================================================
# src/retriever.py
# Hybrid retriever = BM25 keyword search + FAISS vector search
# Results merged using Reciprocal Rank Fusion (RRF)
# Also adds: score threshold + MMR option
# =============================================================================

import pickle
from pathlib import Path
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rich.console                 import Console
from rich.table                   import Table

console = Console()


def build_bm25_retriever(chunks: List[Document],
                          k: int = 10) -> BM25Retriever:
    """
    Builds a BM25 keyword retriever from raw chunks.

    BM25 scores documents by:
      - Term frequency  : how often query word appears in chunk
      - Inverse doc freq: how rare that word is across all chunks
    Rare words that appear often = high BM25 score

    Why k=10 here (not 5)?
      We cast a wide net at this stage.
      RRF fusion + score threshold narrows it down later.

    Args:
        chunks: raw Document chunks from loader
        k     : number of results to return per query

    Returns:
        BM25Retriever ready to use
    """
    console.print(f"  Building BM25 index over {len(chunks)} chunks...")

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k

    console.print(f"  [green]BM25 ready[/green] (k={k})")
    return retriever


def build_faiss_retriever(vectorstore: FAISS,
                           k: int = 10):
    """
    Wraps the FAISS vectorstore as a LangChain retriever.

    Uses standard similarity search (cosine distance).
    k=10 to match BM25 — wide net, narrowed after fusion.

    Args:
        vectorstore: FAISS index from embedder.py
        k          : number of results to return per query

    Returns:
        VectorStoreRetriever ready to use
    """
    console.print(f"  Building FAISS retriever...")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    console.print(f"  [green]FAISS retriever ready[/green] (k={k})")
    return retriever


def build_hybrid_retriever(chunks: List[Document],
                          vectorstore: FAISS,
                          bm25_weight: float = 0.3,
                          vector_weight: float = 0.7,
                          k: int = 10):
    """
    Custom hybrid retriever: runs BM25 and FAISS, merges results using RRF.
    Returns a tuple (bm25_retriever, faiss_retriever, weights, k)
    """
    console.print(f"\n[bold]Building hybrid retriever...[/bold]")
    console.print(f"  Weights: BM25={bm25_weight} | Vector={vector_weight}")

    bm25_retriever = build_bm25_retriever(chunks, k=k)
    faiss_retriever = build_faiss_retriever(vectorstore, k=k)

    console.print(f"[bold green]Hybrid retriever ready[/bold green] "
                  f"[dim](BM25 {int(bm25_weight*100)}% + "
                  f"Vector {int(vector_weight*100)}%)[/dim]\n")
    return (bm25_retriever, faiss_retriever, bm25_weight, vector_weight, k)


def retrieve(hybrid_retriever,
             query: str,
             top_k: int = 5,
             score_threshold: Optional[float] = None) -> List[Document]:
    """
    Runs a query through the custom hybrid retriever and returns top_k results.
    Uses Reciprocal Rank Fusion (RRF) to merge BM25 and FAISS results.
    """
    bm25_retriever, faiss_retriever, bm25_weight, vector_weight, k = hybrid_retriever

    # Get results from both retrievers
    bm25_results = bm25_retriever._get_relevant_documents(query, run_manager=None)
    faiss_results = faiss_retriever._get_relevant_documents(query, run_manager=None)

    # RRF fusion: score = weight / (60 + rank)
    def rrf_scores(results, weight):
        return {doc.metadata.get('chunk_id', id(doc)): weight / (60 + rank)
                for rank, doc in enumerate(results)}

    bm25_scores = rrf_scores(bm25_results, bm25_weight)
    faiss_scores = rrf_scores(faiss_results, vector_weight)

    # Merge scores
    all_docs = {doc.metadata.get('chunk_id', id(doc)): doc for doc in bm25_results + faiss_results}
    combined_scores = {}
    for doc_id in all_docs:
        combined_scores[doc_id] = bm25_scores.get(doc_id, 0) + faiss_scores.get(doc_id, 0)

    # Sort by combined RRF score
    sorted_docs = sorted(all_docs.values(), key=lambda d: combined_scores[d.metadata.get('chunk_id', id(d))], reverse=True)

    # Trim to top_k
    results = sorted_docs[:top_k]

    # Optional: drop results below score threshold (not raw score, but RRF score)
    if score_threshold is not None:
        before = len(results)
        results = [doc for doc in results if combined_scores[doc.metadata.get('chunk_id', id(doc))] >= score_threshold]
        dropped = before - len(results)
        if dropped > 0:
            console.print(f"  [yellow]Score threshold dropped {dropped} weak chunk(s)[/yellow]")

    return results


def _filter_by_threshold(docs: List[Document],
                          query: str,
                          threshold: float) -> List[Document]:
    """
    Filters out documents with low relevance scores.

    Note: EnsembleRetriever doesn't expose raw scores directly.
    We use a simple heuristic: keep top results up to threshold*len.
    For production, use FAISS similarity_search_with_score() directly.

    A practical approach: keep only top ceil(threshold * total) results.
    threshold=0.6 on 5 results → keep top 3
    threshold=0.8 on 5 results → keep top 4
    """
    import math
    keep = max(1, math.ceil(threshold * len(docs)))
    return docs[:keep]


def print_results(results: List[Document], query: str):
    """
    Prints retrieved chunks in a readable table format.
    Useful for debugging retrieval quality.
    """
    if not results:
        console.print(f"[yellow]No results found for:[/yellow] '{query}'")
        return

    table = Table(
        title=f"Retrieved chunks for: '{query}'",
        show_header=True,
        header_style="bold cyan",
        show_lines=True
    )
    table.add_column("#",        width=3,  style="dim")
    table.add_column("File",     width=20, style="white")
    table.add_column("Chunk ID", width=25, style="dim")
    table.add_column("Preview",  style="white")

    for i, doc in enumerate(results, 1):
        # Truncate preview to 120 chars
        preview = doc.page_content.replace("\n", " ")[:120] + "..."
        table.add_row(
            str(i),
            doc.metadata.get("filename",  "?"),
            doc.metadata.get("chunk_id",  "?"),
            preview
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]Returned {len(results)} chunk(s)[/dim]\n")