"""Upwork Semantic Job Matcher — Main CLI."""

import re

import click
from rich.console import Console
from rich.table import Table
from rich.style import Style

from embedder import embed_text
from profile_parser import parse_profile
from qdrant_store import (
    get_client,
    upsert_profile,
    add_job as store_job,
    search_jobs,
    get_profile_vector,
    clear_jobs as clear_all_jobs,
)

console = Console()


@click.group()
def cli():
    """Upwork Semantic Job Matcher — find the best jobs for Ali's profile."""
    pass


@cli.command("index-profile")
def index_profile():
    """Embed and store Ali's profile in Qdrant."""
    console.print("[bold]Parsing profile...[/bold]")
    text = parse_profile()
    console.print(f"Profile text length: {len(text)} chars")

    console.print("[bold]Embedding profile with BGE-M3...[/bold]")
    vector = embed_text(text)

    client = get_client()
    upsert_profile(client, vector, text)
    console.print("[green]Profile indexed successfully.[/green]")


@cli.command("add-job")
@click.argument("job_text")
@click.option("--title", default="", help="Job title")
@click.option("--url", default="", help="Job URL")
@click.option("--budget", default="", help="Job budget")
def add_job_cmd(job_text: str, title: str, url: str, budget: str):
    """Add a single job description."""
    if not title:
        # Try to extract title from first line
        first_line = job_text.strip().splitlines()[0] if job_text.strip() else "Untitled"
        title = first_line[:120]

    # Extract skills: look for common patterns
    skills = _extract_skills(job_text)

    console.print(f"[bold]Embedding job:[/bold] {title[:80]}")
    embed_input = f"{title}\n{job_text}\n{skills}"
    vector = embed_text(embed_input)

    client = get_client()
    point_id = store_job(
        client,
        vector=vector,
        title=title,
        description=job_text,
        url=url,
        budget=budget,
        skills=skills,
    )
    console.print(f"[green]Job added (id={point_id[:8]}...)[/green]")


@cli.command("add-csv")
@click.argument("filepath")
def add_csv(filepath: str):
    """Add jobs from a CSV file.

    Expected columns (header row required):
      title, description, url (opt), budget (opt), skills (opt), posted_date (opt)

    Minimum viable CSV: just title + description columns.
    """
    import csv
    from pathlib import Path

    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        return

    client = get_client()
    count = 0

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalise header names to lowercase stripped
        reader.fieldnames = [h.strip().lower() for h in (reader.fieldnames or [])]

        for row in reader:
            title = row.get("title", "").strip() or "Untitled"
            description = row.get("description", "").strip()
            if not description:
                console.print(f"  [yellow]Skipping row with no description: {title[:60]}[/yellow]")
                continue

            url = row.get("url", "").strip()
            budget = row.get("budget", "").strip() or _extract_budget(description)
            posted_date = row.get("posted_date", row.get("date", "")).strip()
            skills = row.get("skills", "").strip() or _extract_skills(description)

            embed_input = f"{title}\n{description}\n{skills}"
            vector = embed_text(embed_input)

            store_job(
                client,
                vector=vector,
                title=title,
                description=description,
                url=url,
                budget=budget,
                posted_date=posted_date,
                skills=skills,
            )
            count += 1
            console.print(f"  [{count}] {title[:70]}")

    console.print(f"[green]Added {count} jobs from {path.name}.[/green]")


@cli.command("add-rss")
@click.argument("url")
def add_rss(url: str):
    """Fetch and add jobs from an RSS feed URL."""
    console.print(f"[bold]Fetching RSS feed:[/bold] {url}")
    response = httpx.get(url, timeout=30, follow_redirects=True)
    feed = feedparser.parse(response.text)

    if not feed.entries:
        console.print("[red]No entries found in feed.[/red]")
        return

    client = get_client()
    count = 0
    for entry in feed.entries:
        title = entry.get("title", "Untitled")
        description = entry.get("summary", entry.get("description", ""))
        # Strip HTML tags
        description = re.sub(r"<[^>]+>", " ", description)
        description = re.sub(r"\s+", " ", description).strip()
        link = entry.get("link", "")
        published = entry.get("published", "")
        skills = _extract_skills(description)

        embed_input = f"{title}\n{description}\n{skills}"
        vector = embed_text(embed_input)

        store_job(
            client,
            vector=vector,
            title=title,
            description=description,
            url=link,
            budget=_extract_budget(description),
            posted_date=published,
            skills=skills,
        )
        count += 1
        console.print(f"  [{count}] {title[:70]}")

    console.print(f"[green]Added {count} jobs from RSS feed.[/green]")


@cli.command("match")
@click.option("--top", default=10, help="Number of top matches to show")
def match(top: int):
    """Show top matching jobs ranked by semantic similarity to Ali's profile."""
    client = get_client()
    profile_vector = get_profile_vector(client)
    if profile_vector is None:
        console.print(
            "[red]Profile not indexed. Run 'index-profile' first.[/red]"
        )
        return

    results = search_jobs(client, profile_vector, top=top)

    if not results:
        console.print("[yellow]No jobs found. Add some jobs first.[/yellow]")
        return

    table = Table(title="Top Matching Jobs", show_lines=True)
    table.add_column("Rank", style="bold", width=5, justify="center")
    table.add_column("Score", width=7, justify="center")
    table.add_column("Title", width=40)
    table.add_column("Skills", width=30)
    table.add_column("Snippet", width=50)

    high_style = Style(color="green", bold=True)
    normal_style = Style()

    for i, result in enumerate(results, 1):
        score = result["score"]
        style = high_style if score > 0.75 else normal_style
        score_str = f"{score:.3f}"
        title = result.get("title", "Untitled")[:40]
        skills = result.get("skills", "")[:30]
        snippet = result.get("description", "")[:100]

        table.add_row(
            str(i),
            score_str,
            title,
            skills,
            snippet,
            style=style,
        )

    console.print(table)


@cli.command("clear-jobs")
def clear_jobs_cmd():
    """Clear all job embeddings (keeps profile)."""
    client = get_client()
    deleted = clear_all_jobs(client)
    console.print(f"[green]Cleared {deleted} job(s). Profile preserved.[/green]")


def _extract_skills(text: str) -> str:
    """Best-effort skill extraction from job text."""
    known_skills = [
        "Python", "JavaScript", "TypeScript", "React", "Next.js", "Node.js",
        "FastAPI", "Django", "Flask", "PostgreSQL", "MongoDB", "Redis",
        "Docker", "AWS", "GCP", "Azure", "Kubernetes",
        "LangChain", "LangGraph", "OpenAI", "GPT", "Claude", "LLM",
        "RAG", "Vector", "Qdrant", "Pinecone", "Weaviate", "ChromaDB",
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "TensorFlow", "PyTorch", "scikit-learn",
        "Selenium", "Scraping", "BeautifulSoup", "Scrapy",
        "REST API", "GraphQL", "WebSocket",
        "Stripe", "OAuth", "JWT", "Auth",
        "CI/CD", "GitHub Actions", "Terraform",
        "PHP", "Laravel", "WordPress", "Dolibarr",
        "D3", "Recharts", "Tailwind", "CSS",
        "Supabase", "Firebase", "Vercel",
        "Zustand", "Redux", "TanStack",
        "HDBSCAN", "UMAP", "Clustering",
        "sentence-transformers", "embeddings", "BGE",
        "multi-agent", "AI agent", "chatbot", "automation",
    ]
    found = []
    text_lower = text.lower()
    for skill in known_skills:
        if skill.lower() in text_lower:
            found.append(skill)
    return ", ".join(found)


def _extract_budget(text: str) -> str:
    """Try to find a budget/price mention in text."""
    match = re.search(r"\$[\d,]+(?:\s*[-–]\s*\$[\d,]+)?(?:/hr)?", text)
    return match.group(0) if match else ""


if __name__ == "__main__":
    cli()
