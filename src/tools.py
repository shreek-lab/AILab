# =============================================================================
# src/tools.py
# CrewAI tools that wrap the existing DocSearch retrieval pipeline.
#
# A CrewAI Tool is just a Python function decorated with @tool.
# Agents decide WHEN to call these tools based on the task description.
# Each tool must:
#   - Have a clear docstring (agents read this to decide when to use it)
#   - Accept a single string input
#   - Return a string output
# =============================================================================
import os
os.environ["OPENAI_API_KEY"] = "sk-dummy-not-used"
from crewai.tools import tool
from rich.console import Console

console = Console()

# These are set once when the crew is initialised — see crew.py
_retriever = None
_chunks    = None


def init_tools(retriever, chunks):
    """
    Inject the retriever and chunks into the tools module.
    Called once from crew.py before the crew runs.
    """
    global _retriever, _chunks
    _retriever = retriever
    _chunks    = chunks


# =============================================================================
# Tool 1 — Hybrid document search (main retrieval tool)
# =============================================================================

@tool("Search Documents")
def search_docs_tool(query: str) -> str:
    """
    Search the indexed technical documents using hybrid BM25 + vector search.
    Use this tool to find relevant code snippets, function definitions,
    explanations, or any technical content related to the query.
    Input should be a clear search query string.
    Returns the top matching document chunks with their source files.
    """
    if _retriever is None:
        return "Error: retriever not initialised. Run init_tools() first."

    try:
        # Use the existing hybrid retriever from Step 4
        docs = _retriever.invoke(query)
        docs = docs[:5]

        if not docs:
            return f"No documents found for query: '{query}'"

        # Format results as readable string for the agent
        results = []
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "unknown")
            chunk_id = doc.metadata.get("chunk_id",  "unknown")
            content  = doc.page_content[:400]
            results.append(
                f"[Result {i}] Source: {filename} | ID: {chunk_id}\n"
                f"{content}\n"
            )

        return "\n---\n".join(results)

    except Exception as e:
        return f"Search failed: {e}"


# =============================================================================
# Tool 2 — Keyword-only BM25 search
# =============================================================================

@tool("Keyword Search")
def keyword_search_tool(keyword: str) -> str:
    """
    Search for exact keyword matches in the indexed documents using BM25.
    Use this tool when looking for specific function names, variable names,
    error codes, class names, or any exact technical term.
    Input should be the exact keyword or phrase to search for.
    Returns chunks containing that exact term.
    """
    if _chunks is None:
        return "Error: chunks not initialised. Run init_tools() first."

    try:
        from langchain_community.retrievers import BM25Retriever

        bm25 = BM25Retriever.from_documents(_chunks)
        bm25.k = 5
        docs = bm25.invoke(keyword)

        if not docs:
            return f"No exact matches found for keyword: '{keyword}'"

        results = []
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "unknown")
            content  = doc.page_content[:300]
            results.append(f"[Match {i}] {filename}:\n{content}")

        return "\n---\n".join(results)

    except Exception as e:
        return f"Keyword search failed: {e}"


# =============================================================================
# Tool 3 — List all indexed source files
# =============================================================================

@tool("List Indexed Sources")
def list_sources_tool(query: str = "") -> str:
    """
    List all documents and source files currently indexed in the system.
    Use this tool to understand what documents are available before searching,
    or when the user asks what files are indexed.
    Input can be empty or any string (it is ignored).
    Returns a list of all indexed filenames and chunk counts.
    """
    if _chunks is None:
        return "No documents indexed yet."

    try:
        from collections import Counter
        file_counts = Counter(
            c.metadata.get("filename", "unknown") for c in _chunks
        )

        lines = [f"Indexed documents ({len(file_counts)} files, "
                 f"{len(_chunks)} total chunks):"]
        for filename, count in sorted(file_counts.items()):
            lines.append(f"  - {filename}  ({count} chunks)")

        return "\n".join(lines)

    except Exception as e:
        return f"Failed to list sources: {e}"


# =============================================================================
# Tool 4 — Summarise a specific chunk by ID
# =============================================================================

@tool("Summarise Chunk")
def summarise_chunk_tool(chunk_id: str) -> str:
    """
    Retrieve and summarise the full content of a specific document chunk by ID.
    Use this tool when you need to get more context around a specific chunk
    that was returned by a previous search.
    Input should be the chunk_id string (e.g. 'sample.py::chunk_0').
    Returns the full content of that chunk.
    """
    if _chunks is None:
        return "No chunks available."

    try:
        # Find chunk by ID
        for chunk in _chunks:
            if chunk.metadata.get("chunk_id") == chunk_id:
                filename = chunk.metadata.get("filename", "?")
                index    = chunk.metadata.get("chunk_index", "?")
                total    = chunk.metadata.get("chunk_total", "?")
                return (
                    f"Chunk: {chunk_id}\n"
                    f"File: {filename} (chunk {index+1} of {total})\n"
                    f"Content:\n{chunk.page_content}"
                )

        return f"Chunk not found: '{chunk_id}'. Use Search Documents first."

    except Exception as e:
        return f"Failed to retrieve chunk: {e}"