# Creative Writing Workshop v2.0

A standalone AI writing assistant with semantic search, grammar checking, and readability analysis. Browser-based. No configuration.

## Quick Start

```bash
python3 creative_workshop.py
```

The app automatically installs packages, finds and starts Ollama, pulls all models, and opens a web UI.

**Only prerequisite: [Ollama](https://ollama.com)** — install it once.

## Models (auto-downloaded on first run)

| Model | Size | Role |
|---|---|---|
| **mistral-small3.2** (24B) | ~14 GB | Creative writing — scenes, dialogue, worldbuilding |
| **qwen3.5:4b** | ~2.5 GB | Structural tasks — analysis, consistency, filing |
| **nomic-embed-text** | ~270 MB | Semantic search over world bible files |

Ollama swaps models in/out of VRAM automatically. Only one large model loaded at a time.

## Tools

### Create
- **Write** — free-form creative requests, world bible context auto-loaded via semantic search
- **Dialogue** — scene builder, character files found automatically
- **Worldbuild** — create lore entries, filed into world bible with duplicate detection
- **Feedback** — editorial critique on your writing
- **Next Steps** — story direction suggestions

### World Bible
- **Browse** — view, edit, create, delete files
- **Search** — grep across all files

### Analyze
- **Manuscript** — structural analysis + readability metrics (Flesch-Kincaid, Gunning Fog, word count, reading time)
- **Cross-Reference** — extract entities from a manuscript via spaCy NER, check against world bible for contradictions
- **Consistency** — scan world bible for internal conflicts
- **Proofread** — grammar/style checking via LanguageTool (rule-based, catches what LLMs miss)

## How Memory Works

World bible files ARE the long-term memory. On startup, every file is indexed two ways:

1. **Keyword index** — instant Python lookup for exact term matching
2. **Embedding index** — semantic search via nomic-embed-text for conceptual matching

You type "write a scene where the disgraced captain negotiates" and it finds `sera_voss.md` even though "disgraced captain" doesn't appear in the filename — the embedding captures the meaning.

## Backend Toggle

Bottom of the sidebar:
- **Local (Free)** — Mistral Small 3.2 via Ollama. Zero cost, unlimited.
- **Claude API** — Anthropic's Claude. Best quality, pay-per-token.

Set your key to enable Claude: `export ANTHROPIC_API_KEY=sk-ant-...`

## Optional Tools (auto-installed on first use)

| Tool | What it does | Requirement |
|---|---|---|
| **spaCy** | Named entity extraction for cross-referencing | Auto-installed |
| **LanguageTool** | Grammar/style checking (2000+ rules) | Requires Java |
| **textstat** | Readability metrics | Auto-installed |

## File Structure

```
creative_workshop.py       # the app
world_bible/               # your lore (auto-created, preserved)
  characters/  locations/  history/  magic_systems/
  cultures/  languages/  plot_outlines/  notes/
manuscripts/               # your writing
output/                    # saved outputs
```

## Requirements

- Python 3.10+
- Ollama (https://ollama.com)
- 16 GB VRAM recommended (runs on 8 GB with smaller models)
- Java (optional, for LanguageTool proofreading)
