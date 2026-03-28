# Semantic Job Matcher

Stop scrolling job boards manually. Embed your profile once, paste job listings, get ranked matches — no keyword rules needed.

Works with any freelance platform or job board (copy-paste job descriptions into a CSV).

## How It Works

1. **Index your profile** — your profile (title, description, tech stack, portfolio) gets embedded into Qdrant using BGE embeddings
2. **Add job listings** — paste job descriptions via CSV or one at a time
3. **Get ranked matches** — cosine similarity ranks jobs by how well they fit *your* profile semantically

## Stack

- **Embeddings:** `BAAI/bge-base-en-v1.5` (768-dim, cosine similarity)
- **Vector DB:** [Qdrant](https://qdrant.tech/) running locally on `localhost:6333`
- **CLI:** Python + Click + Rich

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# 3. Edit profile.md with your profile info

# 4. Index your profile
python3 matcher.py index-profile

# 5. Add jobs from a CSV
python3 matcher.py add-csv jobs.csv

# 6. See your top matches
python3 matcher.py match --top 10
```

## CSV Format

```csv
title,description,url,budget,skills,posted_date
"Senior AI Engineer","We need...","https://example.com/job/123","60-85/hr","Python,FastAPI","2026-03-27"
```

Only `title` or `description` is required. All other fields are optional.

## CLI Commands

| Command | Description |
|---------|-------------|
| `index-profile` | Embed your profile into Qdrant |
| `add-csv <file>` | Add jobs from a CSV file |
| `add-job <text>` | Add a single job by pasting description |
| `match [--top N]` | Show top N matching jobs (default: 10) |
| `clear-jobs` | Remove all job embeddings |

## Sample Output

```
 Rank  Score  Title                                    Skills
 ────  ─────  ───────────────────────────────────────  ──────────────────────
  1    0.87   Senior AI/ML Engineer – Multi-Agent      Python, LangChain, RAG
  2    0.83   Full-Stack Dev – Next.js + Python AI     Next.js, FastAPI, AI
  3    0.79   Backend Engineer – FastAPI + PostgreSQL  Python, FastAPI, Docker
  4    0.71   AWS Solutions Architect                  AWS, DynamoDB, CDK
  5    0.41   React Native Developer                   React Native, iOS ← correctly deprioritized
```

Semantic > keyword. It deprioritized React Native even though "React" appears in the profile.

## Profile Format

Edit `profile.md` with sections for:
- Title / headline
- Core tech stack
- Competitive advantages
- Portfolio items (these matter a lot for matching quality)

## Why This Works

Keyword filters miss context. A job asking for "AI systems" and a profile mentioning "multi-agent pipelines" won't match on keywords — but they score 0.87 cosine similarity. Semantic search finds the signal.

## License

MIT
