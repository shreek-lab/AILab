import os
os.environ["OPENAI_API_KEY"] = "sk-dummy-not-used"

from crewai import Agent, Task, Crew, Process, LLM
from rich.console import Console
from rich.panel   import Panel
from rich.rule    import Rule

from src.tools import init_tools

console = Console()


def _format_chunks(docs) -> str:
    """Format retrieved docs into a readable string for the agent."""
    if not docs:
        return "No documents found."
    lines = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "unknown")
        content  = doc.page_content[:500]
        lines.append(f"[Source {i}: {filename}]\n{content}")
    return "\n\n---\n\n".join(lines)


def build_crew(query: str, retriever, chunks,
               model: str = "llama3.2") -> Crew:
    """
    Builds the crew with retrieval results pre-injected into task descriptions.

    Why pre-inject instead of using tool calls?
      Small local models (3B, 7B) don't reliably follow the ReAct
      tool-calling pattern. They hallucinate instead of calling tools.
      Pre-injecting the results bypasses this limitation entirely —
      the model only needs to READ and SYNTHESISE, not REASON and ACT.

      This is called 'context stuffing' and is a valid production
      pattern when you don't have access to a large capable model.
    """
    init_tools(retriever, chunks)

    llm = LLM(
        model=f"ollama/{model}",
        base_url="http://localhost:11434",
    )

    # ── Pre-retrieve results before building tasks ────────────────────────────
    # Run hybrid retrieval now so we can inject results into task description
    console.print(f"\n[dim]Pre-retrieving context for crew...[/dim]")

    # Semantic search
    semantic_docs = retriever.invoke(query)[:5]
    semantic_ctx  = _format_chunks(semantic_docs)

    # Keyword search
    try:
        from langchain_community.retrievers import BM25Retriever
        bm25      = BM25Retriever.from_documents(chunks)
        bm25.k    = 3
        kw_docs   = bm25.invoke(query)
        kw_ctx    = _format_chunks(kw_docs)
    except Exception:
        kw_ctx = "Keyword search unavailable."

    # List of indexed files
    from collections import Counter
    file_counts  = Counter(c.metadata.get("filename","?") for c in chunks)
    sources_list = "\n".join(f"  - {f} ({n} chunks)"
                              for f, n in file_counts.items())

    console.print(f"[green]Retrieved {len(semantic_docs)} chunks[/green]\n")

    # =========================================================================
    # AGENT 1 — Query Analyst
    # Has all info pre-injected — just needs to analyse
    # =========================================================================
    query_analyst = Agent(
        role="Query Analyst",
        goal="Analyse the query and identify key concepts.",
        backstory="You are an expert at understanding technical questions.",
        tools=[],           # no tools needed — data is pre-injected
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )

    # =========================================================================
    # AGENT 2 — Document Retriever
    # Results are pre-injected — just needs to select the best ones
    # =========================================================================
    doc_retriever = Agent(
        role="Document Retriever",
        goal="Select the most relevant chunks for the query.",
        backstory="You are a specialist in finding relevant technical content.",
        tools=[],           # no tools needed — results are pre-injected
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
    )

    # =========================================================================
    # AGENT 3 — Answer Writer
    # Writes the final grounded answer from the context
    # =========================================================================
    answer_writer = Agent(
        role="Answer Writer",
        goal="Write a clear grounded answer using only the provided context.",
        backstory=(
            "You are a senior technical writer. "
            "You only use facts from the provided context. "
            "You always cite sources like [filename]."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )

    # =========================================================================
    # TASK 1 — Analyse query
    # Pre-inject the file list so agent knows what's available
    # =========================================================================
    task_analyse = Task(
        description=(
            f"The user asked: '{query}'\n\n"
            f"Available indexed files:\n{sources_list}\n\n"
            "Identify the key technical concepts in this question. "
            "State which files are likely to contain the answer."
        ),
        expected_output=(
            "Key concepts identified and which files to focus on."
        ),
        agent=query_analyst,
    )

    # =========================================================================
    # TASK 2 — Select best chunks
    # Pre-inject ALL retrieved results so agent just needs to filter
    # =========================================================================
    task_retrieve = Task(
        description=(
            f"Query: '{query}'\n\n"
            f"SEMANTIC SEARCH RESULTS:\n{semantic_ctx}\n\n"
            f"KEYWORD SEARCH RESULTS:\n{kw_ctx}\n\n"
            "Review these results and select the 3 most relevant chunks "
            "that best answer the query. List them with their source filenames."
        ),
        expected_output=(
            "The 3 most relevant chunks with their source filenames."
        ),
        agent=doc_retriever,
        context=[task_analyse],
    )

    # =========================================================================
    # TASK 3 — Write final answer
    # =========================================================================
    task_answer = Task(
        description=(
            f"Question: '{query}'\n\n"
            "Using ONLY the content selected in the previous task, "
            "write a complete technical answer.\n\n"
            "Rules:\n"
            "1. Cite every fact as [filename]\n"
            "2. Include code snippets if present in the context\n"
            "3. If context is insufficient say: "
            "'I don't have enough information in the indexed documents'\n"
            "4. Do NOT use knowledge outside the provided context"
        ),
        expected_output=(
            "A complete technical answer with [filename] citations "
            "and code snippets from the context."
        ),
        agent=answer_writer,
        context=[task_retrieve],
    )

    crew = Crew(
        agents=[query_analyst, doc_retriever, answer_writer],
        tasks=[task_analyse, task_retrieve, task_answer],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def run_crew(query: str, retriever, chunks,
             model: str = "llama3.2") -> str:

    console.print(f"\n[bold cyan]Starting CrewAI pipeline...[/bold cyan]")
    console.print(f"[dim]Query: {query}[/dim]\n")
    console.print(Rule("Crew Execution", style="cyan"))

    crew         = build_crew(query, retriever, chunks, model)
    result       = crew.kickoff(inputs={"query": query})
    final_answer = str(result)

    console.print(Rule("Final Answer", style="green"))
    console.print(Panel(
        final_answer,
        title="[bold green]CrewAI Answer[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))

    return final_answer