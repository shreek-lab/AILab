# =============================================================================
# src/loader.py
# Loads documents from the /documents folder and splits into chunks
# Supports: .py .md .txt .js .ts .java .cpp .c .go .rs
# =============================================================================

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from rich.console import Console
from rich.table import Table

console = Console()

# ── Supported file types ──────────────────────────────────────────────────────
# Maps file extension → LangChain Language enum (for smart code splitting)
# Files not in this map are treated as plain text
EXTENSION_MAP = {
    ".py":   Language.PYTHON,
    ".js":   Language.JS,
    ".ts":   Language.JS,
    ".java": Language.JAVA,
    ".cpp":  Language.CPP,
    ".c":    Language.CPP,
    ".go":   Language.GO,
    ".rs":   Language.RUST,
    ".md":   None,   # markdown → plain text splitter
    ".txt":  None,   # plain text splitter
}


def load_documents(documents_dir: str = "documents") -> List[Document]:
    """
    Walks the documents/ folder and loads all supported files.
    Each file becomes one or more LangChain Document objects.

    Returns a list of Document objects with:
      - page_content : the raw text of the file
      - metadata     : source path, file type, file name
    """
    docs_path = Path(documents_dir)

    # Check the folder exists
    if not docs_path.exists():
        console.print(f"[red]Error:[/red] '{documents_dir}' folder not found.")
        console.print("Create it and add your code/doc files inside.")
        return []

    documents = []
    found_files = []

    # Walk all files recursively
    for file_path in sorted(docs_path.rglob("*")):
        if file_path.is_file() and file_path.suffix in EXTENSION_MAP:
            found_files.append(file_path)

    if not found_files:
        console.print(f"[yellow]No supported files found in '{documents_dir}/'[/yellow]")
        console.print(f"Supported: {', '.join(EXTENSION_MAP.keys())}")
        return []

    console.print(f"\n[bold]Found {len(found_files)} file(s) to load...[/bold]")

    for file_path in found_files:
        try:
            # Read raw text
            text = file_path.read_text(encoding="utf-8", errors="ignore")

            if not text.strip():
                console.print(f"  [yellow]Skipped (empty):[/yellow] {file_path.name}")
                continue

            # Create a Document with useful metadata
            doc = Document(
                page_content=text,
                metadata={
                    "source":    str(file_path),          # full path
                    "filename":  file_path.name,          # e.g. loader.py
                    "filetype":  file_path.suffix,        # e.g. .py
                    "language":  file_path.suffix.lstrip(".").upper(),
                }
            )
            documents.append(doc)
            console.print(f"  [green]Loaded:[/green] {file_path.name} "
                          f"[dim]({len(text)} chars)[/dim]")

        except Exception as e:
            console.print(f"  [red]Failed:[/red] {file_path.name} → {e}")

    console.print(f"\n[bold green]Loaded {len(documents)} document(s)[/bold green]")
    return documents


def chunk_documents(documents: List[Document],
                    chunk_size: int = 512,
                    chunk_overlap: int = 100) -> List[Document]:
    """
    Splits documents into overlapping chunks.

    For code files  → uses language-aware splitter
                      (splits on class/function boundaries first)
    For text files  → uses recursive character splitter
                      (splits on paragraphs → lines → words)

    Args:
        documents    : list of Document objects from load_documents()
        chunk_size   : max tokens per chunk (default 512)
        chunk_overlap: overlap between consecutive chunks (default 100)
                       prevents answers being split at chunk boundaries

    Returns:
        List of smaller Document objects, each with chunk metadata added
    """
    all_chunks = []

    for doc in documents:
        ext      = doc.metadata.get("filetype", ".txt")
        language = EXTENSION_MAP.get(ext)

        # ── Choose splitter based on file type ──
        if language is not None:
            # Code-aware splitter — respects function/class boundaries
            # Tries to split on: class → function → block → line → character
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            # Plain text splitter — splits on paragraphs → lines → words
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )

        # Split this document into chunks
        chunks = splitter.split_documents([doc])

        # Add chunk position metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)
            chunk.metadata["chunk_id"]    = (
                f"{doc.metadata['filename']}::chunk_{i}"
            )

        all_chunks.extend(chunks)

    # ── Print summary table ───────────────────────────────────────────────────
    _print_chunk_summary(all_chunks)
    print("All chunkss data isssssss",all_chunks[0].metadata)
    return all_chunks


def _print_chunk_summary(chunks: List[Document]):
    """Prints a nice summary table of what was chunked."""
    # Group by source file
    from collections import defaultdict
    file_counts = defaultdict(int)
    file_sizes  = defaultdict(list)

    for chunk in chunks:
        name = chunk.metadata.get("filename", "unknown")
        file_counts[name] += 1
        file_sizes[name].append(len(chunk.page_content))

    table = Table(title="Chunking Summary", show_header=True,
                  header_style="bold cyan")
    table.add_column("File",         style="white")
    table.add_column("Chunks",       justify="right", style="green")
    table.add_column("Avg size",     justify="right", style="yellow")
    table.add_column("Min / Max",    justify="right", style="dim")

    for filename, count in file_counts.items():
        sizes   = file_sizes[filename]
        avg     = sum(sizes) // len(sizes)
        mn, mx  = min(sizes), max(sizes)
        table.add_row(filename, str(count), f"{avg} chars",
                      f"{mn} / {mx}")

    console.print()
    console.print(table)
    console.print(f"\n[bold]Total chunks ready for embedding:[/bold] "
                  f"[green]{len(chunks)}[/green]\n")