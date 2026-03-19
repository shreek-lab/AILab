# =============================================================================
# main.py — CLI interface for the document search system
#
# Commands:
#   python main.py index    → load documents, chunk, embed, save FAISS index
#   python main.py search   → interactive chat loop with hybrid RAG
#   python main.py status   → show what's currently indexed
#
# Usage:
#   1. Drop your .py .md .txt files into documents/
#   2. python main.py index
#   3. python main.py search
# =============================================================================

import sys
from pathlib import Path
from typing  import List, Tuple

import typer
from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text
from rich.rule    import Rule

# ── Local imports ─────────────────────────────────────────────────────────────
from src.loader    import load_documents, chunk_documents
from src.embedder  import (get_embedding_model, build_vectorstore,
                            load_vectorstore, vectorstore_exists)
from src.retriever import build_hybrid_retriever
from src.chain     import get_llm, build_rag_chain, ask

# ── Setup ─────────────────────────────────────────────────────────────────────
app     = typer.Typer(help="DocSearch — RAG-powered technical document search")
console = Console()

# ── Config ────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR   = "documents"
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 100
BM25_WEIGHT     = 0.3
VECTOR_WEIGHT   = 0.7
TOP_K           = 5


# =============================================================================
# index command
# =============================================================================

@app.command()
def index(
    docs_dir: str = typer.Option(
        DOCUMENTS_DIR,
        "--docs", "-d",
        help="Folder containing documents to index"
    ),
    chunk_size: int = typer.Option(
        CHUNK_SIZE,
        "--chunk-size", "-c",
        help="Max characters per chunk"
    ),
    chunk_overlap: int = typer.Option(
        CHUNK_OVERLAP,
        "--overlap", "-o",
        help="Overlap between consecutive chunks"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force re-indexing even if index already exists"
    )
):
    """
    Index documents from the documents/ folder.

    Loads all .py .md .txt .js etc files, chunks them,
    embeds using local sentence-transformers model,
    and saves FAISS index to vectorstore/.

    Run this once after adding or changing documents.
    """
    _print_header()
    console.print(Rule("[bold]INDEXING[/bold]", style="cyan"))

    # ── Check if already indexed ──────────────────────────────────────────────
    if vectorstore_exists(VECTORSTORE_DIR) and not force:
        console.print("\n[yellow]Index already exists.[/yellow]")
        console.print("Use [bold]--force[/bold] to re-index: "
                      "[dim]python main.py index --force[/dim]")
        console.print("Or run [bold]python main.py search[/bold] to start searching.\n")
        raise typer.Exit()

    # ── Step 1: Load documents ────────────────────────────────────────────────
    console.print(Rule("Step 1 — Loading documents", style="dim"))
    docs = load_documents(docs_dir)

    if not docs:
        console.print("[red]No documents found. Add files to documents/ first.[/red]")
        raise typer.Exit(1)

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    console.print(Rule("Step 2 — Chunking", style="dim"))
    chunks = chunk_documents(
        docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # ── Step 3: Embed + save ──────────────────────────────────────────────────
    console.print(Rule("Step 3 — Embedding + saving index", style="dim"))
    embeddings  = get_embedding_model()
    vectorstore = build_vectorstore(chunks, embeddings, VECTORSTORE_DIR)

    # ── Done ──────────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[green]Indexing complete![/green]\n\n"
        f"  Documents : [cyan]{len(docs)}[/cyan]\n"
        f"  Chunks    : [cyan]{len(chunks)}[/cyan]\n"
        f"  Saved to  : [cyan]{VECTORSTORE_DIR}/[/cyan]\n\n"
        f"Now run: [bold]python main.py search[/bold]",
        title="Done",
        border_style="green"
    ))


# =============================================================================
# search command
# =============================================================================

@app.command()
def search(
    top_k: int = typer.Option(
        TOP_K,
        "--top-k", "-k",
        help="Number of chunks to retrieve per query"
    ),
    show_sources: bool = typer.Option(
        True,
        "--sources/--no-sources",
        help="Show retrieved source chunks"
    ),
    model: str = typer.Option(
        "tinyllama",
        "--model", "-m",
        help="Ollama model to use (must be pulled)"
    )
):
    """
    Start interactive search chat loop.

    Type your question and get answers grounded in your documents.
    Commands inside the chat:
      /exit   or  q  → quit
      /clear         → clear conversation history
      /sources       → toggle showing sources on/off
      /help          → show commands
    """
    _print_header()
    console.print(Rule("[bold]SEARCH[/bold]", style="cyan"))

    # ── Check index exists ────────────────────────────────────────────────────
    if not vectorstore_exists(VECTORSTORE_DIR):
        console.print("[red]No index found.[/red] Run indexing first:")
        console.print("  [bold]python main.py index[/bold]\n")
        raise typer.Exit(1)

    # ── Load everything ───────────────────────────────────────────────────────
    console.print(Rule("Loading pipeline", style="dim"))

    embeddings            = get_embedding_model()
    vectorstore, chunks   = load_vectorstore(embeddings, VECTORSTORE_DIR)
    result                = build_hybrid_retriever(chunks, vectorstore,
                                                    BM25_WEIGHT, VECTOR_WEIGHT)
    retriever             = result[0] if isinstance(result, tuple) else result
    llm                   = get_llm(model=model)
    chain                 = build_rag_chain(llm)

    # ── Print ready message ───────────────────────────────────────────────────
    console.print(Panel(
        f"[green]Pipeline ready![/green]\n\n"
        f"  Model     : [cyan]{model}[/cyan]\n"
        f"  Top-k     : [cyan]{top_k}[/cyan]\n"
        f"  BM25      : [cyan]{int(BM25_WEIGHT*100)}%[/cyan] | "
        f"Vector: [cyan]{int(VECTOR_WEIGHT*100)}%[/cyan]\n\n"
        f"Type your question below.\n"
        f"[dim]Commands: /exit  /clear  /sources  /help[/dim]",
        title="DocSearch Ready",
        border_style="cyan"
    ))

    # ── Chat loop ─────────────────────────────────────────────────────────────
    chat_history: List[Tuple[str, str]] = []
    show_src = show_sources

    while True:
        try:
            # ── Get user input ────────────────────────────────────────────────
            console.print()
            query = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not query:
                continue

            # ── Handle commands ───────────────────────────────────────────────
            if query.lower() in ("/exit", "q", "quit", "exit"):
                console.print("\n[dim]Goodbye![/dim]\n")
                break

            elif query.lower() == "/clear":
                chat_history = []
                console.print("[green]Conversation history cleared.[/green]")
                continue

            elif query.lower() == "/sources":
                show_src = not show_src
                state = "ON" if show_src else "OFF"
                console.print(f"[green]Sources display: {state}[/green]")
                continue

            elif query.lower() == "/help":
                _print_help()
                continue

            elif query.lower() == "/history":
                _print_history(chat_history)
                continue

            # ── Run RAG pipeline ──────────────────────────────────────────────
            answer, source_docs = ask(
                query        = query,
                retriever    = retriever,
                chain        = chain,
                chat_history = chat_history,
                top_k        = top_k,
                show_sources = show_src
            )

            # ── Save to history ───────────────────────────────────────────────
            # Keep last 10 turns to avoid context window overflow
            chat_history.append((query, answer))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Type /exit to quit.[/dim]")
            continue

        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            console.print("[dim]Try rephrasing your question.[/dim]")
            continue


# =============================================================================
# status command
# =============================================================================

@app.command()
def status():
    """
    Show what is currently indexed.
    Displays document count, chunk count, and index size on disk.
    """
    _print_header()
    console.print(Rule("[bold]STATUS[/bold]", style="cyan"))

    # Check documents folder
    docs_path = Path(DOCUMENTS_DIR)
    vs_path   = Path(VECTORSTORE_DIR)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Item",   style="white", width=30)
    table.add_column("Status", style="green")

    # Documents folder
    if docs_path.exists():
        files = list(docs_path.rglob("*"))
        files = [f for f in files if f.is_file()]
        table.add_row("Documents folder", f"{len(files)} file(s) found")
    else:
        table.add_row("Documents folder",
                      "[red]Not found — create documents/[/red]")

    # Vector index
    faiss_file = vs_path / "faiss_index.faiss"
    if faiss_file.exists():
        size_mb = faiss_file.stat().st_size / (1024 * 1024)
        table.add_row("FAISS index", f"[green]Exists[/green] ({size_mb:.2f} MB)")
    else:
        table.add_row("FAISS index",
                      "[yellow]Not built — run: python main.py index[/yellow]")

    # Chunks file
    chunks_file = vs_path / "chunks.pkl"
    if chunks_file.exists():
        import pickle
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        table.add_row("Indexed chunks", f"{len(chunks)} chunks")
    else:
        table.add_row("Indexed chunks", "[yellow]Not found[/yellow]")

    # Ollama
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2)
        models = [m["name"] for m in r.json().get("models", [])]
        table.add_row("Ollama models", ", ".join(models) if models else "none pulled")
    except Exception:
        table.add_row("Ollama", "[red]Not running[/red]")

    console.print(table)


# =============================================================================
# Helpers
# =============================================================================

def _print_header():
    console.print()
    console.print(Panel(
        "[bold cyan]DocSearch[/bold cyan]\n"
        "[dim]RAG-powered technical document search[/dim]\n"
        "[dim]Stack: LangChain + FAISS + BM25 + Ollama[/dim]",
        border_style="cyan",
        padding=(0, 2)
    ))
    console.print()


def _print_help():
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")

    table.add_row("/exit or q", "Quit the search session")
    table.add_row("/clear",     "Clear conversation history")
    table.add_row("/sources",   "Toggle showing retrieved sources")
    table.add_row("/history",   "Show conversation history")
    table.add_row("/help",      "Show this help message")

    console.print(Panel(table, title="Commands", border_style="dim"))


def _print_history(history: List[Tuple[str, str]]):
    if not history:
        console.print("[dim]No conversation history yet.[/dim]")
        return
    for i, (q, a) in enumerate(history, 1):
        console.print(f"[dim]{i}. Q: {q[:80]}[/dim]")
        console.print(f"[dim]   A: {a[:80]}...[/dim]")

@app.command()
def evaluate(
    num_questions: int = typer.Option(
        5, "--questions", "-n",
        help="Number of test questions to generate"
    ),
    save: bool = typer.Option(
        True, "--save/--no-save",
        help="Save results to evaluation_results.json"
    ),
    model: str = typer.Option(
        "tinyllama", "--model", "-m",
        help="Ollama model to use"
    )
):
    """
    Evaluate RAG pipeline quality using RAGAS metrics.
    """
    from src.evaluator import (generate_test_questions, run_pipeline_on_testset,
                                evaluate_with_ragas, print_scores, save_results)

    _print_header()
    console.print(Rule("[bold]EVALUATION[/bold]", style="cyan"))

    if not vectorstore_exists(VECTORSTORE_DIR):
        console.print("[red]No index found.[/red] Run: python main.py index")
        raise typer.Exit(1)

    embeddings          = get_embedding_model()
    vectorstore, chunks = load_vectorstore(embeddings, VECTORSTORE_DIR)
    result              = build_hybrid_retriever(chunks, vectorstore,
                                                  BM25_WEIGHT, VECTOR_WEIGHT)
    retriever           = result[0] if isinstance(result, tuple) else result
    llm                 = get_llm(model=model)
    chain               = build_rag_chain(llm)

    test_pairs = generate_test_questions(chunks, llm, num_questions)

    if not test_pairs:
        console.print("[red]Could not generate test questions.[/red]")
        raise typer.Exit(1)

    results = run_pipeline_on_testset(test_pairs, retriever, chain)
    scores  = evaluate_with_ragas(results)

    print_scores(scores, len(results))

    if save:
        save_results(scores, results)

@app.command()
def crew(
    query: str = typer.Argument(..., help="Your question for the crew"),
    model: str = typer.Option("tinyllama", "--model", "-m",
                               help="Ollama model to use"),
):
    """
    Run a query through the CrewAI multi-agent pipeline.

    3 agents work in sequence:
      1. Query Analyst   — expands and analyses your question
      2. Doc Retriever   — finds relevant chunks using hybrid search
      3. Answer Writer   — synthesises a grounded cited answer
    """
    from src.crew import run_crew

    _print_header()
    console.print(Rule("[bold]CREW MODE[/bold]", style="cyan"))

    if not vectorstore_exists(VECTORSTORE_DIR):
        console.print("[red]No index found.[/red] Run: python main.py index")
        raise typer.Exit(1)

    embeddings          = get_embedding_model()
    vectorstore, chunks = load_vectorstore(embeddings, VECTORSTORE_DIR)
    result              = build_hybrid_retriever(chunks, vectorstore,
                                                  BM25_WEIGHT, VECTOR_WEIGHT)
    retriever           = result[0] if isinstance(result, tuple) else result

    run_crew(query=query, retriever=retriever,
             chunks=chunks, model=model)   

@app.command()
def crew_eval(
    query: str = typer.Argument(..., help="Question to ask the crew"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Chunks to retrieve")
):
    """
    Run a query through CrewAI and score the answer using
    faithfulness, answer_relevancy, context_precision, context_recall.
    """
    from src.crew      import run_crew
    from src.evaluator import _heuristic_scores, print_scores

    _print_header()
    console.print(Rule("[bold]CREW EVALUATION[/bold]", style="cyan"))

    if not vectorstore_exists(VECTORSTORE_DIR):
        console.print("[red]No index found.[/red] Run: python main.py index")
        raise typer.Exit(1)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    console.print("\n[bold]Loading pipeline...[/bold]")
    embeddings          = get_embedding_model()
    vectorstore, chunks = load_vectorstore(embeddings, VECTORSTORE_DIR)
    result              = build_hybrid_retriever(chunks, vectorstore,
                                                  BM25_WEIGHT, VECTOR_WEIGHT)
    retriever           = result[0] if isinstance(result, tuple) else result

    # ── Run CrewAI ────────────────────────────────────────────────────────────
    console.print(Rule("Running CrewAI", style="teal"))
    crew_answer = run_crew(
        query     = query,
        retriever = retriever,
        chunks    = chunks,
        model     = model
    )

    # ── Retrieve contexts for scoring ─────────────────────────────────────────
    # We retrieve the same chunks the crew would have used
    # and use them as the reference context for scoring
    console.print(Rule("Scoring CrewAI answer", style="yellow"))
    crew_docs     = retriever.invoke(query)[:top_k]
    crew_contexts = [d.page_content for d in crew_docs]

    # ── Score ─────────────────────────────────────────────────────────────────
    scores = _heuristic_scores([{
        "question":     query,
        "answer":       crew_answer,
        "contexts":     crew_contexts,
        "ground_truth": crew_answer,
    }])

    # ── Print scores ──────────────────────────────────────────────────────────
    print_scores(scores, num_questions=1)

    # ── Save ──────────────────────────────────────────────────────────────────
    import json
    from datetime import datetime
    from pathlib  import Path

    output = {
        "timestamp":    datetime.now().isoformat(),
        "query":        query,
        "model":        model,
        "answer":       crew_answer,
        "contexts_used": crew_contexts,
        "scores":       scores,
    }
    Path("crew_eval_results.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    console.print(f"\n[dim]Results saved → crew_eval_results.json[/dim]")                  
# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    app()