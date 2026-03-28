"""Parse upwork-profile.md into a single rich text block for embedding."""

from pathlib import Path
import re

PROFILE_PATH = Path(__file__).parent / "upwork-profile.md"


def parse_profile(path: Path = PROFILE_PATH) -> str:
    """Extract key sections from the profile and combine into one text block.

    Sections extracted: Title, Core Tech Stack, Competitive Advantages,
    Portfolio descriptions, Professional One-Liner.
    """
    text = path.read_text(encoding="utf-8")
    chunks: list[str] = []

    # Title
    title_match = re.search(r"\*\*Title:\*\*\s*(.+)", text)
    if title_match:
        chunks.append(title_match.group(1).strip())

    # Core Tech Stack table rows
    stack_section = re.search(
        r"## Core Tech Stack\n\|.*\n\|.*\n([\s\S]*?)(?=\n---|\n## )", text
    )
    if stack_section:
        rows = re.findall(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|", stack_section.group(1))
        for domain, techs in rows:
            chunks.append(f"{domain}: {techs}")

    # Competitive Advantages
    advantages_section = re.search(
        r"## Competitive Advantages.*?\n([\s\S]*?)(?=\n---|\n## )", text
    )
    if advantages_section:
        lines = [
            line.strip()
            for line in advantages_section.group(1).splitlines()
            if line.strip() and not line.strip().startswith(">")
        ]
        chunks.extend(lines)

    # Portfolio descriptions
    for desc_match in re.finditer(
        r"\*\*Description:\*\*\s*(.+)", text
    ):
        chunks.append(desc_match.group(1).strip())

    # Professional One-Liner
    oneliner = re.search(r'> "(.+)"', text)
    if oneliner:
        chunks.append(oneliner.group(1).strip())

    return "\n".join(chunks)


if __name__ == "__main__":
    print(parse_profile())
