# =============================================================================
# src/evaluator.py
# Evaluates RAG pipeline quality
# Tries RAGAS first, falls back to heuristic scoring if unavailable
# =============================================================================

import json
from pathlib  import Path
from typing   import List, Dict, Tuple, Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_ollama  import ChatOllama
from rich.console      import Console
from rich.table        import Table
from rich.panel        import Panel
from rich.rule         import Rule

console = Console()


def generate_test_questions(chunks: List[Document],
                             llm: ChatOllama,
                             num_questions: int = 5) -> List[Dict]:
    """
    Auto-generates Q&A pairs from document chunks.
    Uses flexible parsing to handle small models like tinyllama.
    """
    console.print(f"\n[bold]Generating {num_questions} test questions "
                  f"from your documents...[/bold]")
    console.print("[dim]Uses tinyllama to create Q&A pairs[/dim]\n")

    if len(chunks) <= num_questions:
        selected = chunks
    else:
        step     = len(chunks) // num_questions
        selected = [chunks[i * step] for i in range(num_questions)]

    test_pairs = []

    for i, chunk in enumerate(selected, 1):
        filename = chunk.metadata.get("filename", "?")
        console.print(f"  [{i}/{len(selected)}] Generating from "
                      f"[cyan]{filename}[/cyan]...")

        # Simpler prompt — easier for small models to follow
        prompt = f"""Read this code and write one question and answer about it.

CODE:
{chunk.page_content[:400]}

Write your response like this:
Q: (write a question about the code)
A: (write the answer)"""

        try:
            response = llm.invoke(prompt).content.strip()
            console.print(f"    [dim]Raw response: {response[:120]}[/dim]")

            question     = ""
            ground_truth = ""

            # Try multiple parsing strategies
            lines = [l.strip() for l in response.split("\n") if l.strip()]

            for line in lines:
                # Strategy 1 — Q: / A: format
                if line.startswith("Q:") and not question:
                    question = line[2:].strip()
                elif line.startswith("A:") and not ground_truth:
                    ground_truth = line[2:].strip()

                # Strategy 2 — QUESTION: / ANSWER: format
                elif line.upper().startswith("QUESTION:") and not question:
                    question = line.split(":", 1)[1].strip()
                elif line.upper().startswith("ANSWER:") and not ground_truth:
                    ground_truth = line.split(":", 1)[1].strip()

                # Strategy 3 — numbered format like "1. What is..."
                elif line.startswith("1.") and not question:
                    question = line[2:].strip()
                elif line.startswith("2.") and not ground_truth:
                    ground_truth = line[2:].strip()

            # Strategy 4 — if still nothing, use first 2 non-empty lines
            if not question and len(lines) >= 2:
                question     = lines[0]
                ground_truth = lines[1]

            # Strategy 5 — if only one line, generate answer from chunk
            if not question and len(lines) == 1:
                question     = lines[0]
                ground_truth = chunk.page_content[:200]

            # Clean up — remove any leading Q:/A: that got included
            question     = question.lstrip("Q:").lstrip("q:").strip()
            ground_truth = ground_truth.lstrip("A:").lstrip("a:").strip()

            if question and ground_truth:
                test_pairs.append({
                    "question":     question,
                    "ground_truth": ground_truth,
                    "source":       filename,
                    "chunk_id":     chunk.metadata.get("chunk_id", "?"),
                })
                console.print(f"    [green]Q:[/green] {question[:70]}")
            else:
                # Last resort — create a simple question manually from filename
                console.print(f"    [yellow]Using fallback question[/yellow]")
                fallback_q = f"What does the code in {filename} do?"
                fallback_a = chunk.page_content[:300]
                test_pairs.append({
                    "question":     fallback_q,
                    "ground_truth": fallback_a,
                    "source":       filename,
                    "chunk_id":     chunk.metadata.get("chunk_id", "?"),
                })
                console.print(f"    [green]Q:[/green] {fallback_q}")

        except Exception as e:
            console.print(f"    [red]Failed:[/red] {e}")
            continue

    console.print(f"\n[green]Generated {len(test_pairs)} test pairs[/green]\n")
    return test_pairs


def run_pipeline_on_testset(test_pairs: List[Dict],
                             retriever,
                             chain,
                             top_k: int = 5) -> List[Dict]:
    """
    Runs each test question through the full RAG pipeline.
    Records question, generated answer, retrieved contexts, ground truth.
    """
    console.print(f"[bold]Running pipeline on {len(test_pairs)} "
                  f"test questions...[/bold]\n")

    results = []

    for i, pair in enumerate(test_pairs, 1):
        question     = pair["question"]
        ground_truth = pair["ground_truth"]

        console.print(f"  [{i}/{len(test_pairs)}] {question[:65]}...")

        try:
            # Retrieve chunks
            docs = retriever.invoke(question)
            docs = docs[:top_k]

            # Format context same way as chain.py
            from src.chain import format_context
            context = format_context(docs)

            # Generate answer
            answer = chain.invoke({
                "context":      context,
                "chat_history": "",
                "question":     question,
            })

            # Collect plain text contexts for RAGAS
            contexts = [doc.page_content for doc in docs]

            results.append({
                "question":     question,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": ground_truth,
            })

            console.print(f"    [green]Done[/green] — "
                          f"{answer[:60]}...")

        except Exception as e:
            console.print(f"    [red]Failed:[/red] {e}")
            continue

    console.print(f"\n[green]Completed {len(results)} / "
                  f"{len(test_pairs)} questions[/green]\n")
    return results


def evaluate_with_ragas(results: List[Dict]) -> Dict:
    """
    Scores results with RAGAS if available, heuristics if not.
    """
    try:
        from ragas         import evaluate
        from ragas.metrics import (faithfulness,
                                   answer_relevancy,
                                   context_precision,
                                   context_recall)
        from datasets      import Dataset

        console.print(Rule("[bold]Running RAGAS evaluation[/bold]",
                           style="cyan"))
        console.print("[dim]RAGAS uses an LLM judge internally — "
                      "may take a few minutes[/dim]\n")

        dataset = Dataset.from_list(results)
        score   = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
        )
        return dict(score)

    except Exception as e:
        console.print(f"[yellow]RAGAS unavailable:[/yellow] {e}")
        console.print("[dim]Falling back to heuristic scoring...[/dim]\n")
        return _heuristic_scores(results)


def _heuristic_scores(results: List[Dict]) -> Dict:
    """
    Fallback scoring without RAGAS.
    Uses word overlap as a proxy for each metric.
    """
    console.print("[bold]Running heuristic evaluation...[/bold]\n")

    faithfulness_scores = []
    relevancy_scores    = []
    precision_scores    = []
    recall_scores       = []

    for r in results:
        answer       = r["answer"].lower()
        question     = r["question"].lower()
        ground_truth = r["ground_truth"].lower()
        contexts     = " ".join(r["contexts"]).lower()

        # Faithfulness — answer words found in context?
        answer_words  = set(w for w in answer.split() if len(w) > 3)
        context_words = set(w for w in contexts.split() if len(w) > 3)
        faith = (len(answer_words & context_words) /
                 max(len(answer_words), 1))
        faithfulness_scores.append(min(faith * 1.5, 1.0))

        # Answer relevancy — question words found in answer?
        q_words = set(w for w in question.split() if len(w) > 3)
        rel     = len(q_words & answer_words) / max(len(q_words), 1)
        relevancy_scores.append(min(rel * 1.5, 1.0))

        # Context precision — contexts contain ground truth words?
        gt_words  = set(w for w in ground_truth.split() if len(w) > 3)
        prec      = (len(gt_words & context_words) /
                     max(len(gt_words), 1))
        precision_scores.append(min(prec * 1.5, 1.0))

        # Context recall — answer contains ground truth words?
        rec = (len(gt_words & answer_words) /
               max(len(gt_words), 1))
        recall_scores.append(min(rec * 1.5, 1.0))

    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    scores = {
        "faithfulness":      avg(faithfulness_scores),
        "answer_relevancy":  avg(relevancy_scores),
        "context_precision": avg(precision_scores),
        "context_recall":    avg(recall_scores),
    }

    console.print("[dim](Heuristic scores — install RAGAS for "
                  "LLM-judged scores)[/dim]\n")
    return scores


def print_scores(scores: Dict, num_questions: int):
    """Prints evaluation scores with interpretation and advice."""

    def color(s):
        if s >= 0.85: return "green"
        if s >= 0.70: return "yellow"
        return "red"

    def label(s):
        if s >= 0.85: return "Excellent"
        if s >= 0.70: return "Good"
        if s >= 0.50: return "Needs work"
        return "Critical"

    fixes = {
        "faithfulness":      "Strengthen grounding instruction in prompt",
        "answer_relevancy":  "Check prompt template clarity",
        "context_precision": "Lower top-k or add score threshold",
        "context_recall":    "Increase top-k or improve chunking",
    }

    descriptions = {
        "faithfulness":      "Answer grounded in context?",
        "answer_relevancy":  "Answer addresses the question?",
        "context_precision": "Retrieved chunks were useful?",
        "context_recall":    "Retrieved all needed info?",
    }

    table = Table(
        title=f"Evaluation Results ({num_questions} questions)",
        show_header=True,
        header_style="bold cyan",
        show_lines=True
    )
    table.add_column("Metric",       style="white",  width=22)
    table.add_column("Score",        justify="center", width=8)
    table.add_column("Rating",       justify="center", width=12)
    table.add_column("Meaning",      style="dim",    width=28)
    table.add_column("Fix if low",   style="dim",    width=36)

    for metric, score in scores.items():
        c = color(score)
        l = label(score)
        table.add_row(
            metric,
            f"[{c}]{score:.2f}[/{c}]",
            f"[{c}]{l}[/{c}]",
            descriptions.get(metric, ""),
            fixes.get(metric, "") if score < 0.85 else "[dim]—[/dim]"
        )

    console.print()
    console.print(table)

    values  = list(scores.values())
    overall = round(sum(values) / len(values), 3) if values else 0.0
    c       = color(overall)

    advice = {
        True:  "[green]Production ready.[/green] Pipeline performing well.",
        False: ("[yellow]Good foundation.[/yellow] Focus on lowest metric.\n"
                "Common fixes: tune top-k, improve chunking, stronger prompt.")
    }

    console.print(Panel(
        f"Overall RAG score: [{c}]{overall:.2f}[/{c}]\n\n"
        + advice[overall >= 0.85],
        title="Summary",
        border_style=c
    ))


def save_results(scores: Dict,
                 results: List[Dict],
                 output_path: str = "evaluation_results.json"):
    """Saves full evaluation results to JSON."""
    output = {
        "timestamp":  datetime.now().isoformat(),
        "scores":     scores,
        "num_tested": len(results),
        "details":    results,
    }
    Path(output_path).write_text(
        json.dumps(output, indent=2, default=str),
        encoding="utf-8"
    )
    console.print(f"\n[dim]Full results saved → {output_path}[/dim]")