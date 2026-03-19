# =============================================================================
# src/chain.py
# Builds the full RAG chain:
#   user query → hybrid retrieval → prompt construction → Ollama → answer
#
# Key design decisions:
#   - Grounding instruction prevents hallucination
#   - Citation forcing makes hallucination visible
#   - Temperature=0 for deterministic factual answers
#   - Conversation history for multi-turn chat
# =============================================================================
from langchain.retrievers import EnsembleRetriever
from typing import List, Tuple, Optional
from langchain_core.documents      import Document
from langchain_core.prompts        import PromptTemplate
from langchain_ollama              import ChatOllama
from langchain_core.runnables      import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rich.console                 import Console
from rich.panel                   import Panel
from rich.markdown                import Markdown
from rich.text                    import Text

console = Console()

# ── Ollama model settings ─────────────────────────────────────────────────────
OLLAMA_MODEL       = "llama3.2"   # change to "mistral" or "llama3.2" if needed
OLLAMA_TEMPERATURE = 0.0           # 0 = deterministic, no creativity
                                   # keeps answers grounded in context
OLLAMA_BASE_URL    = "http://localhost:11434"  # default Ollama port


# =============================================================================
# PROMPT TEMPLATE
# This is the most important part of the chain.
# Every block serves a specific anti-hallucination purpose.
# =============================================================================

RAG_PROMPT_TEMPLATE = """You are an expert technical assistant helping developers \
understand code and documentation.

INSTRUCTIONS:
- Answer ONLY using the context provided below.
- If the answer is not present in the context, respond with:
  "I don't have enough information in the provided documents to answer that."
- Do NOT use your training knowledge. Do NOT speculate.
- For every fact you state, cite the source file in square brackets like [filename].
- If the question is about code, include relevant code snippets from the context.
- Be concise and precise.

{context}

{chat_history}QUESTION:
{question}

ANSWER (cite sources using [filename]):"""


# =============================================================================
# Context formatter
# Takes retrieved Document chunks and formats them for the prompt
# =============================================================================

def format_context(docs: List[Document]) -> str:
    """
    Formats retrieved chunks into a structured context block.

    Each chunk gets:
      - A numbered header with source file and chunk position
      - The raw content
      - A separator line

    Why numbering matters:
      Helps the LLM reference specific chunks when citing.
      "As shown in [sample.py]..." maps back to CONTEXT [1].

    Example output:
      CONTEXT:
      ─────────────────────────────
      [1] Source: sample.py | Chunk 1 of 2
      def binary_search(arr, target):
          ...
      ─────────────────────────────
    """
    if not docs:
        return "CONTEXT:\nNo relevant documents found.\n"

    lines = ["CONTEXT:"]
    separator = "─" * 50

    for i, doc in enumerate(docs, 1):
        filename    = doc.metadata.get("filename",    "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        chunk_total = doc.metadata.get("chunk_total", "?")
        language    = doc.metadata.get("language",    "")

        lines.append(separator)
        lines.append(f"[{i}] Source: {filename} | "
                     f"Chunk {chunk_index + 1} of {chunk_total}")
        lines.append("")

        # Wrap code content in markdown code block
        # Helps the LLM understand it's reading code, not prose
        if language in ("PY", "JS", "TS", "JAVA", "CPP", "C", "GO", "RS"):
            lang_hint = language.lower()
            lines.append(f"```{lang_hint}")
            lines.append(doc.page_content)
            lines.append("```")
        else:
            lines.append(doc.page_content)

        lines.append("")

    lines.append(separator)
    return "\n".join(lines)


def format_chat_history(history: List[Tuple[str, str]]) -> str:
    """
    Formats conversation history for multi-turn chat.

    Args:
        history: list of (question, answer) tuples from previous turns

    Returns:
        Formatted string or empty string if no history

    Why we include history:
        Allows follow-up questions like "can you show me an example?"
        without repeating context from earlier in the conversation.

    Token budget warning:
        History grows with every turn. We keep last 3 turns only
        to avoid context window overflow.
    """
    if not history:
        return ""

    lines = ["CONVERSATION HISTORY:"]
    # Keep only last 3 turns to manage token budget
    for question, answer in history[-3:]:
        lines.append(f"Human: {question}")
        lines.append(f"Assistant: {answer}")
        lines.append("")

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# LLM setup
# =============================================================================

def get_llm(model: str = OLLAMA_MODEL,
            temperature: float = OLLAMA_TEMPERATURE) -> ChatOllama:
    """
    Creates a ChatOllama LLM instance.

    Why temperature=0?
      - We want factual, deterministic answers
      - Higher temperature = more creative = more likely to drift
        from the retrieved context = hallucination risk
      - For RAG: always use 0 or very low (max 0.1)

    Args:
        model      : Ollama model name (must be pulled already)
        temperature: 0.0 = deterministic, 1.0 = very creative

    Returns:
        ChatOllama instance ready to use
    """
    console.print(f"\n[bold]Connecting to Ollama...[/bold]")
    console.print(f"  Model      : {model}")
    console.print(f"  Temperature: {temperature}")
    console.print(f"  URL        : {OLLAMA_BASE_URL}")

    llm = ChatOllama(
        model       = f"{model}:latest" if ":" not in model else model,
        temperature = temperature,
        base_url = OLLAMA_BASE_URL,
        client_kwargs={"timeout": 120}  
    )

    console.print(f"[green]Ollama ready[/green]\n")
    return llm


# =============================================================================
# RAG Chain builder
# =============================================================================

def build_rag_chain(llm: ChatOllama):
    """
    Builds the RAG generation chain (without retriever).

    This is a simple chain:
        prompt → LLM → string output

    The retriever is kept separate so we can:
      1. Show retrieved chunks to the user before generating
      2. Log retrieval separately from generation
      3. Evaluate retrieval and generation independently

    Chain structure (LangChain LCEL):
        {context, chat_history, question}
              ↓
        PromptTemplate  (fills in the template)
              ↓
        ChatOllama      (generates answer)
              ↓
        StrOutputParser (extracts text from LLM response)
              ↓
        answer string

    Returns:
        Runnable chain that takes a dict and returns a string
    """
    prompt = PromptTemplate(
        template       = RAG_PROMPT_TEMPLATE,
        input_variables = ["context", "chat_history", "question"]
    )

    # LCEL pipe syntax: prompt | llm | parser
    # Each step's output feeds into the next step's input
    chain = prompt | llm | StrOutputParser()

    return chain


# =============================================================================
# Main ask() function — the entry point for all queries
# =============================================================================

def ask(query: str,
        retriever: EnsembleRetriever,
        chain,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        top_k: int = 5,
        show_sources: bool = True) -> Tuple[str, List[Document]]:
    """
    Full RAG pipeline: query → retrieve → prompt → generate → answer.

    Steps:
      1. Retrieve top_k relevant chunks using hybrid retriever
      2. Format chunks into structured context block
      3. Format conversation history (if any)
      4. Fill prompt template with context + history + query
      5. Send to Ollama → stream back the answer
      6. Return (answer, source_docs) tuple

    Args:
        query        : user's question
        retriever    : hybrid retriever from retriever.py
        chain        : RAG chain from build_rag_chain()
        chat_history : list of (q, a) tuples for multi-turn chat
        top_k        : number of chunks to retrieve
        show_sources : whether to print retrieved sources

    Returns:
        (answer_string, list_of_source_documents)
    """
    if chat_history is None:
        chat_history = []

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Retrieving context...[/bold cyan]")
    docs = retriever.invoke(query)
    docs = docs[:top_k]   # trim to top_k after RRF fusion

    if show_sources:
        _print_sources(docs)

    # ── Step 2: Format context ────────────────────────────────────────────────
    context      = format_context(docs)
    history_text = format_chat_history(chat_history)

    # ── Step 3: Generate ──────────────────────────────────────────────────────
    console.print(f"[bold cyan]Generating answer...[/bold cyan]\n")

    # Invoke the chain with all inputs
    answer = chain.invoke({
        "context":      context,
        "chat_history": history_text,
        "question":     query,
    })

    # ── Step 4: Display answer ────────────────────────────────────────────────
    _print_answer(answer, query)

    return answer, docs


# =============================================================================
# Display helpers
# =============================================================================

def _print_sources(docs: List[Document]):
    """Prints a compact list of retrieved source chunks."""
    if not docs:
        console.print("[yellow]No sources retrieved[/yellow]")
        return

    console.print(f"[dim]Retrieved {len(docs)} source(s):[/dim]")
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        preview  = doc.page_content.replace("\n", " ")[:80]
        console.print(f"  [dim]{i}. {filename} [{chunk_id}][/dim]")
        console.print(f"     [dim italic]{preview}...[/dim italic]")
    console.print()


def _print_answer(answer: str, query: str):
    """Prints the answer in a formatted panel."""
    console.print(
        Panel(
            Markdown(answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )