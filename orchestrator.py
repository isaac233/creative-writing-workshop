#!/usr/bin/env python3
"""
Creative Writing Agentic Workflow Orchestrator
===============================================
Routes tasks between:
  - Claude API   → creative writing, feedback, dialogue, worldbuilding decisions
  - Qwen 3.5:4B  → file management, indexing, context retrieval, structural analysis
  - Gemma 4 E4B  → multimodal tasks (maps, art references, visual worldbuilding)

Memory model:
  - World bible files ARE the long-term memory
  - Fast local keyword index replaces LLM-based retrieval
  - Auto-context detection loads relevant files without manual keywords
  - Write-back keeps files current as the story evolves
  - Session log provides lightweight short-term continuity

Requirements:
  pip install anthropic requests rich prompt_toolkit --break-system-packages

Usage:
  1. Start Ollama:        ollama serve
  2. Pull models:         ollama pull qwen3.5:4b && ollama pull gemma4:e4b
  3. Set your API key:    export ANTHROPIC_API_KEY=sk-ant-...
  4. Run:                 python orchestrator.py
"""

import os
import re
import sys
import json
import shutil
import time
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

LOCAL_WRITER = "qwen3.5:4b"           # librarian, organizer, retriever
LOCAL_CREATIVE = "mistral-small3.2"   # local creative writing (24B, fits 16GB VRAM)

# ─── Backend toggle ──────────────────────────────────────────────────────────
# "claude"  = use Claude API (best quality, costs money per token)
# "local"   = use LOCAL_CREATIVE via Ollama (free, unlimited, lower quality)
creative_backend = "claude"

WORLD_BIBLE_DIR = Path("./world_bible")
MANUSCRIPTS_DIR = Path("./manuscripts")
OUTPUT_DIR = Path("./output")

# ─── Ensure directories exist (preserves any existing files) ─────────────────

for d in [WORLD_BIBLE_DIR, MANUSCRIPTS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

for sub in ["characters", "locations", "history", "magic_systems",
            "cultures", "languages", "plot_outlines", "notes"]:
    (WORLD_BIBLE_DIR / sub).mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TOKEN TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

class TokenTracker:
    """Approximate token usage and cost tracking for a session."""

    def __init__(self):
        self.input_chars = 0
        self.output_chars = 0
        self.api_calls = 0

    def record(self, input_text: str, output_text: str):
        self.input_chars += len(input_text)
        self.output_chars += len(output_text)
        self.api_calls += 1

    @property
    def input_tokens(self):
        return self.input_chars // 4  # rough approximation

    @property
    def output_tokens(self):
        return self.output_chars // 4

    @property
    def estimated_cost(self):
        # Sonnet pricing: ~$3/M input, ~$15/M output
        return (self.input_tokens * 3 / 1_000_000) + (self.output_tokens * 15 / 1_000_000)

    def summary(self):
        return (f"  API calls: {self.api_calls}\n"
                f"  ~{self.input_tokens:,} input tokens, ~{self.output_tokens:,} output tokens\n"
                f"  Estimated cost: ${self.estimated_cost:.4f}")


tokens = TokenTracker()


# ═══════════════════════════════════════════════════════════════════════════════
#  WORLD INDEX — fast keyword-based file retrieval
# ═══════════════════════════════════════════════════════════════════════════════

class WorldIndex:
    """
    In-memory keyword index over all world bible files.
    Rebuilt on startup and after any write. No LLM call needed for retrieval.
    Falls back to Qwen only for fuzzy/semantic queries.
    """

    # Words too common to be useful for matching
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "ought",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same", "than", "too",
        "very", "just", "about", "above", "after", "again", "against",
        "below", "between", "by", "down", "during", "for", "from", "in",
        "into", "of", "off", "on", "out", "over", "through", "to", "under",
        "until", "up", "with", "that", "this", "these", "those", "it", "its",
        "he", "she", "they", "them", "his", "her", "their", "him", "we",
        "you", "your", "my", "our", "who", "whom", "which", "what", "where",
        "when", "how", "why", "if", "then", "else", "also", "as", "at",
        "like", "one", "two", "three", "said", "says", "well", "back",
        "even", "still", "new", "now", "way", "many", "much", "make",
        "made", "know", "known", "see", "seen", "think", "come", "take",
        "get", "got", "go", "went", "here", "there",
    }

    def __init__(self, bible_dir: Path):
        self.bible_dir = bible_dir
        # term → set of relative file paths
        self.term_to_files: dict[str, set[str]] = defaultdict(set)
        # file path → set of terms (for reverse lookup)
        self.file_to_terms: dict[str, set[str]] = {}
        # file path → first line / title for display
        self.file_titles: dict[str, str] = {}
        self.rebuild()

    def _tokenize(self, text: str) -> set[str]:
        """Extract meaningful terms from text."""
        # Lowercase, split on non-alphanumeric, filter short/stop words
        words = re.findall(r"[a-zA-Z']+", text.lower())
        terms = set()
        for w in words:
            w = w.strip("'")
            if len(w) >= 3 and w not in self.STOP_WORDS:
                terms.add(w)
        # Also extract multi-word names from markdown headers
        headers = re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)
        for h in headers:
            # Keep full header as a matchable phrase (lowercased)
            normalized = h.strip().lower()
            if len(normalized) >= 3:
                terms.add(normalized)
        return terms

    def rebuild(self):
        """Rebuild the entire index from disk. Safe to call anytime."""
        self.term_to_files.clear()
        self.file_to_terms.clear()
        self.file_titles.clear()

        for path in sorted(self.bible_dir.rglob("*.md")):
            rel = str(path.relative_to(self.bible_dir))
            content = path.read_text(encoding="utf-8", errors="replace")

            # Extract title from first heading or filename
            title_match = re.search(r"^#+\s+(.+)$", content, re.MULTILINE)
            if title_match:
                self.file_titles[rel] = title_match.group(1).strip()
            else:
                self.file_titles[rel] = path.stem.replace("_", " ").title()

            # Index terms from filename + content
            filename_text = path.stem.replace("_", " ")
            category_text = path.parent.name.replace("_", " ")
            all_text = f"{filename_text} {category_text} {content}"
            terms = self._tokenize(all_text)

            self.file_to_terms[rel] = terms
            for t in terms:
                self.term_to_files[t].add(rel)

    def reindex_file(self, relative_path: str):
        """Reindex a single file after a write. Faster than full rebuild."""
        # Remove old entries for this file
        old_terms = self.file_to_terms.pop(relative_path, set())
        for t in old_terms:
            self.term_to_files[t].discard(relative_path)
            if not self.term_to_files[t]:
                del self.term_to_files[t]
        self.file_titles.pop(relative_path, None)

        # Re-add if file still exists
        full = self.bible_dir / relative_path
        if full.exists():
            content = full.read_text(encoding="utf-8", errors="replace")
            title_match = re.search(r"^#+\s+(.+)$", content, re.MULTILINE)
            if title_match:
                self.file_titles[relative_path] = title_match.group(1).strip()
            else:
                self.file_titles[relative_path] = full.stem.replace("_", " ").title()

            filename_text = full.stem.replace("_", " ")
            category_text = full.parent.name.replace("_", " ")
            all_text = f"{filename_text} {category_text} {content}"
            terms = self._tokenize(all_text)
            self.file_to_terms[relative_path] = terms
            for t in terms:
                self.term_to_files[t].add(relative_path)

    def find(self, query: str, max_results: int = 10) -> list[tuple[str, float]]:
        """
        Find files matching a query. Returns list of (filepath, score) sorted
        by relevance. Score = fraction of query terms found in file.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores: dict[str, float] = defaultdict(float)
        for term in query_terms:
            # Exact match
            for f in self.term_to_files.get(term, set()):
                scores[f] += 1.0
            # Substring match (catches partial names like "sera" matching "sera voss")
            for indexed_term, files in self.term_to_files.items():
                if term != indexed_term and (term in indexed_term or indexed_term in term):
                    for f in files:
                        scores[f] += 0.5

        if not scores:
            return []

        # Normalize by number of query terms
        results = [(f, s / len(query_terms)) for f, s in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def search_content(self, pattern: str, case_sensitive: bool = False) -> list[tuple[str, list[str]]]:
        """
        Grep-like search across all world bible files.
        Returns list of (filepath, [matching_lines]).
        """
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error:
            compiled = re.compile(re.escape(pattern), flags)

        results = []
        for path in sorted(self.bible_dir.rglob("*.md")):
            rel = str(path.relative_to(self.bible_dir))
            content = path.read_text(encoding="utf-8", errors="replace")
            matches = []
            for i, line in enumerate(content.splitlines(), 1):
                if compiled.search(line):
                    matches.append(f"  {i}: {line.strip()}")
            if matches:
                results.append((rel, matches))
        return results

    @property
    def file_count(self):
        return len(self.file_to_terms)

    @property
    def term_count(self):
        return len(self.term_to_files)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION LOG — lightweight short-term continuity
# ═══════════════════════════════════════════════════════════════════════════════

class SessionLog:
    """
    Append-only log of what happened this session.
    Persisted to notes/session_log.md so it survives restarts.
    Keeps a rolling in-memory buffer of recent exchanges for context.
    """

    def __init__(self, bible_dir: Path, buffer_size: int = 10):
        self.log_path = bible_dir / "notes" / "session_log.md"
        self.buffer: list[dict] = []
        self.buffer_size = buffer_size

        # Create log file if it doesn't exist, but don't overwrite
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(
                "# Session Log\n\nAutomatic record of workflow sessions.\n\n",
                encoding="utf-8",
            )

    def record(self, command: str, summary: str, files_touched: list[str] = None):
        """Record an exchange."""
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "command": command,
            "summary": summary[:200],  # keep summaries short
            "files": files_touched or [],
        }
        self.buffer.append(entry)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Append to persistent log
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n### {entry['time']} — `{command}`\n")
            f.write(f"{summary[:200]}\n")
            if files_touched:
                f.write(f"Files: {', '.join(files_touched)}\n")

    def recent_context(self) -> str:
        """Return recent session activity as context string."""
        if not self.buffer:
            return ""
        lines = ["RECENT SESSION ACTIVITY:"]
        for e in self.buffer[-5:]:  # last 5 exchanges
            files_note = f" (files: {', '.join(e['files'])})" if e['files'] else ""
            lines.append(f"  [{e['time']}] {e['command']}: {e['summary']}{files_note}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════

def call_ollama(model: str, prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """Call a local model via Ollama's API."""
    payload = {
        "model": model,
        "messages": [],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 32768},
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})

    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot reach Ollama. Is it running? (ollama serve)"
    except Exception as e:
        return f"[ERROR] Ollama call failed: {e}"


def call_claude(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """Call Claude via the Anthropic API. Tracks token usage."""
    if not CLAUDE_API_KEY:
        return "[ERROR] Set ANTHROPIC_API_KEY environment variable."

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system

    full_input = (system or "") + prompt

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        result = "".join(b["text"] for b in data["content"] if b["type"] == "text")
        tokens.record(full_input, result)
        return result
    except Exception as e:
        return f"[ERROR] Claude API call failed: {e}"


def is_error(text: str) -> bool:
    """Check if a model response is an error."""
    return text.strip().startswith("[ERROR]")


def call_creative(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    """
    Route a creative request to the active backend.
    - "claude" → Claude API (best quality, pay-per-token)
    - "local"  → LOCAL_CREATIVE via Ollama (free, unlimited)
    """
    if creative_backend == "local":
        return call_ollama(LOCAL_CREATIVE, prompt, system=system, temperature=temperature)
    else:
        return call_claude(prompt, system=system, temperature=temperature)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORLD BIBLE MANAGEMENT — file-backed long-term memory
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize the global index — picks up all existing files
world_index = WorldIndex(WORLD_BIBLE_DIR)
session_log = SessionLog(WORLD_BIBLE_DIR)


def list_world_bible() -> dict:
    """Scan the world bible and return a structured index."""
    index = {}
    for path in sorted(WORLD_BIBLE_DIR.rglob("*.md")):
        # Skip session log from display
        if path.name == "session_log.md":
            continue
        category = path.parent.name
        if category not in index:
            index[category] = []
        index[category].append({
            "file": str(path.relative_to(WORLD_BIBLE_DIR)),
            "name": path.stem.replace("_", " ").title(),
            "size": path.stat().st_size,
        })
    return index


def read_world_file(relative_path: str) -> str:
    """Read a file from the world bible."""
    full = WORLD_BIBLE_DIR / relative_path
    if full.exists():
        return full.read_text(encoding="utf-8")
    return f"[File not found: {relative_path}]"


def write_world_file(relative_path: str, content: str) -> str:
    """Write or update a file in the world bible. Updates the index."""
    full = WORLD_BIBLE_DIR / relative_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    world_index.reindex_file(relative_path)
    return f"Saved: {relative_path}"


def delete_world_file(relative_path: str) -> str:
    """Delete a file from the world bible. Updates the index."""
    full = WORLD_BIBLE_DIR / relative_path
    if not full.exists():
        return f"[File not found: {relative_path}]"
    full.unlink()
    world_index.reindex_file(relative_path)  # removes from index
    return f"Deleted: {relative_path}"


def move_world_file(from_path: str, to_path: str) -> str:
    """Move/rename a file within the world bible."""
    src = WORLD_BIBLE_DIR / from_path
    dst = WORLD_BIBLE_DIR / to_path
    if not src.exists():
        return f"[Source not found: {from_path}]"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    world_index.reindex_file(from_path)  # remove old
    world_index.reindex_file(to_path)    # add new
    return f"Moved: {from_path} → {to_path}"


def merge_world_files(paths: list[str], target_path: str, delete_sources: bool = False) -> str:
    """Merge multiple world bible files into one."""
    contents = []
    for p in paths:
        text = read_world_file(p)
        if text.startswith("[File not found"):
            return text
        contents.append(f"\n\n<!-- merged from: {p} -->\n\n{text}")

    merged = "\n".join(contents)
    write_world_file(target_path, merged)

    if delete_sources:
        for p in paths:
            if p != target_path:
                delete_world_file(p)

    return f"Merged {len(paths)} files into {target_path}"


def assemble_context(query: str, max_chars: int = 30000, extra_files: list[str] = None) -> str:
    """
    Fast context assembly using the keyword index.
    Falls back to Qwen for fuzzy matching if keyword search finds nothing.
    """
    # 1. Fast keyword lookup
    matches = world_index.find(query)

    # 2. If nothing found, try Qwen as semantic fallback
    if not matches:
        index = list_world_bible()
        if not index:
            return "(World bible is empty — start by creating some entries.)"

        index_text = json.dumps(index, indent=2)
        prompt = f"""Given this query: {query}

Here is the world bible index:
{index_text}

Return ONLY a JSON array of the file paths most relevant to this query.
Example: ["characters/arya.md", "locations/winterfell.md"]
Return [] if nothing is relevant. No explanation, just the JSON array."""

        result = call_ollama(LOCAL_WRITER, prompt, temperature=0.1)

        try:
            start = result.index("[")
            end = result.rindex("]") + 1
            files = json.loads(result[start:end])
        except (ValueError, json.JSONDecodeError):
            # Regex fallback for malformed JSON
            files = re.findall(r'"([^"]+\.md)"', result)

        matches = [(f, 1.0) for f in files]

    # 3. Add any explicitly requested files
    if extra_files:
        existing = {m[0] for m in matches}
        for ef in extra_files:
            if ef not in existing:
                matches.append((ef, 2.0))  # high priority

    if not matches:
        return "(No relevant world bible entries found.)"

    # 4. Read and concatenate, respecting size limit
    assembled = []
    total = 0
    for filepath, score in matches:
        content = read_world_file(filepath)
        if content.startswith("[File not found"):
            continue
        entry = f"\n--- {filepath} (relevance: {score:.1f}) ---\n{content}\n"
        if total + len(entry) > max_chars:
            break
        assembled.append(entry)
        total += len(entry)

    if not assembled:
        return "(No relevant world bible entries found.)"
    return "".join(assembled)


def auto_context(user_input: str, max_chars: int = 30000) -> str:
    """
    Automatically detect which world bible files are relevant to the user's
    input and load them. No manual keywords needed.
    """
    matches = world_index.find(user_input)
    if not matches:
        return ""

    # Only include files with meaningful relevance
    good_matches = [(f, s) for f, s in matches if s >= 0.3]
    if not good_matches:
        return ""

    assembled = []
    total = 0
    loaded_files = []
    for filepath, score in good_matches:
        content = read_world_file(filepath)
        if content.startswith("[File not found"):
            continue
        entry = f"\n--- {filepath} ---\n{content}\n"
        if total + len(entry) > max_chars:
            break
        assembled.append(entry)
        total += len(entry)
        loaded_files.append(filepath)

    if not assembled:
        return ""

    if loaded_files:
        print(f"  📂 Auto-loaded: {', '.join(loaded_files)}")

    return "".join(assembled)


# ═══════════════════════════════════════════════════════════════════════════════
#  FILE ORGANIZATION  (Qwen 3.5 — local)
# ═══════════════════════════════════════════════════════════════════════════════

def organize_files(directory: str, instruction: str) -> str:
    """Use Qwen 3.5 to analyze and propose file organization."""
    target = Path(directory)
    if not target.exists():
        return f"[Directory not found: {directory}]"

    file_list = []
    for p in sorted(target.rglob("*")):
        if p.is_file():
            file_list.append(f"  {p.relative_to(target)}  ({p.stat().st_size} bytes)")

    listing = "\n".join(file_list) if file_list else "(empty)"

    prompt = f"""You are a file organization assistant for a creative writer.

Directory: {directory}
Contents:
{listing}

Instruction: {instruction}

Provide a clear plan with specific rename/move commands. Format each action as:
ACTION: MOVE|RENAME|CREATE_DIR|DELETE
FROM: <path>
TO: <path>
REASON: <why>

Only list concrete actions. Be precise with paths."""

    return call_ollama(LOCAL_WRITER, prompt,
                       system="You organize files for creative writing projects. Be precise and systematic.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CREATIVE WRITING TASKS  (Claude — API)
# ═══════════════════════════════════════════════════════════════════════════════

CREATIVE_SYSTEM = """You are a skilled creative writing collaborator. You help with:
- Worldbuilding: designing rich, internally consistent worlds
- Character development: creating complex, believable characters
- Dialogue: writing natural speech that reflects character voice and subtext
- Plot structure: planning narrative arcs and scene sequences
- Prose craft: drafting and refining vivid, engaging writing
- Editorial feedback: honest, constructive critique with specific suggestions

You have access to the writer's world bible context below. Stay consistent with
established lore. When you notice potential contradictions, flag them.

When you establish NEW facts about existing elements (character traits, place
details, timeline events, rule clarifications), note them clearly at the end
under a section called "NEW LORE ESTABLISHED" so the system can offer to update
the relevant files.

Always write in a way that serves the story. Be bold with suggestions but
respect the writer's vision."""


def creative_task(task: str) -> str:
    """Route a creative writing task to Claude with auto-detected context."""
    context = auto_context(task)
    session_context = session_log.recent_context()

    full_prompt = task
    if context:
        full_prompt = f"""WORLD BIBLE CONTEXT:
{context}

{session_context}

TASK:
{task}"""
    elif session_context:
        full_prompt = f"""{session_context}

TASK:
{task}"""

    return call_creative(full_prompt, system=CREATIVE_SYSTEM)


def write_dialogue(characters: list[str], situation: str, tone: str = "") -> str:
    """Specialized dialogue writing with character context."""
    query = " ".join(characters) + " " + situation
    context = auto_context(query)
    session_context = session_log.recent_context()

    tone_note = f"\nTone/mood: {tone}" if tone else ""

    prompt = f"""WORLD BIBLE CONTEXT (characters involved):
{context if context else "(No character files found — writing from scratch.)"}

{session_context}

Write a dialogue scene with the following parameters:
- Characters: {', '.join(characters)}
- Situation: {situation}{tone_note}

Requirements:
- Each character must sound distinct and consistent with their established voice
- Include brief action beats and internal reactions between lines
- Subtext matters — what characters DON'T say is as important as what they do
- The scene should advance character relationships or plot

Write the scene."""

    return call_creative(prompt, system=CREATIVE_SYSTEM)


def get_feedback(text: str, focus: str = "general") -> str:
    """Get editorial feedback from Claude on a piece of writing."""
    prompt = f"""Please provide detailed editorial feedback on this writing.

FOCUS AREA: {focus}
(Options: general, dialogue, pacing, prose-style, character-voice, tension, worldbuilding)

TEXT TO REVIEW:
---
{text}
---

Structure your feedback as:
1. What's working well (be specific with examples from the text)
2. What needs attention (be specific and explain why)
3. Concrete suggestions for improvement (show, don't just tell)
4. Suggested next steps for revision"""

    return call_creative(prompt, system=CREATIVE_SYSTEM)


def suggest_next_steps(current_state: str) -> str:
    """Get suggestions for what to write or develop next."""
    context = auto_context(current_state + " plot outline current")
    session_context = session_log.recent_context()

    prompt = f"""WORLD BIBLE CONTEXT:
{context if context else "(No relevant context found.)"}

{session_context}

CURRENT STATE:
{current_state}

Based on the story's current state and the world bible:
1. What are the 3 most promising directions for the next scene/chapter?
2. Which characters are due for development or a POV shift?
3. Are there any planted seeds (foreshadowing, unresolved tension) ready to pay off?
4. What worldbuilding elements haven't been explored yet but could enrich the story?

Be specific and reference established lore."""

    return call_creative(prompt, system=CREATIVE_SYSTEM)


# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE-BACK — update files when new lore is established
# ═══════════════════════════════════════════════════════════════════════════════

def detect_new_lore(claude_output: str) -> str | None:
    """Check if Claude flagged new lore in its response."""
    marker = "NEW LORE ESTABLISHED"
    if marker in claude_output:
        idx = claude_output.index(marker)
        return claude_output[idx:]
    return None


def offer_writeback(claude_output: str, original_query: str) -> list[str]:
    """
    If Claude established new lore, offer to update relevant files.
    Returns list of files that were updated.
    """
    new_lore = detect_new_lore(claude_output)
    if not new_lore:
        return []

    print(f"\n  ✦ New lore was established in this response.")

    # Find which files the new lore relates to
    matches = world_index.find(original_query + " " + new_lore[:500])
    if not matches:
        print("  (No matching world bible files to update.)")
        save_new = input("  Save as a new note? (y/n): ").strip().lower()
        if save_new == "y":
            name = input("  Filename (e.g. unicorn_update): ").strip() or "lore_note"
            path = f"notes/{name}.md"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            content = f"# Lore Note — {timestamp}\n\n{new_lore}\n"
            write_world_file(path, content)
            print(f"  ✓ Saved to world_bible/{path}")
            return [path]
        return []

    updated = []
    for filepath, score in matches[:5]:  # offer top 5 at most
        title = world_index.file_titles.get(filepath, filepath)
        choice = input(f"  Update '{title}' ({filepath})? (y/n/view): ").strip().lower()
        if choice == "view":
            print(f"\n{read_world_file(filepath)}\n")
            choice = input(f"  Update this file? (y/n): ").strip().lower()
        if choice == "y":
            existing = read_world_file(filepath)
            timestamp = datetime.now().strftime("%Y-%m-%d")

            # Use Qwen to merge new lore into existing file
            merge_prompt = f"""Merge this new information into the existing document.
Keep all existing content. Add the new information in the appropriate sections.
If there are contradictions, keep the NEW information and note what changed.
Add a changelog entry at the bottom.

EXISTING DOCUMENT:
{existing}

NEW INFORMATION (established {timestamp}):
{new_lore}

Return the complete updated document in markdown format."""

            merged = call_ollama(LOCAL_WRITER, merge_prompt,
                                 system="You merge new information into existing documents. Preserve all content.",
                                 temperature=0.2)

            if not is_error(merged):
                write_world_file(filepath, merged)
                print(f"  ✓ Updated: {filepath}")
                updated.append(filepath)
            else:
                # Fallback: append new lore
                appended = existing + f"\n\n---\n\n## Update — {timestamp}\n\n{new_lore}\n"
                write_world_file(filepath, appended)
                print(f"  ✓ Appended to: {filepath}")
                updated.append(filepath)

    return updated


# ═══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL ANALYSIS  (Qwen 3.5 — local)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_manuscript(filepath: str) -> str:
    """Use Qwen 3.5 to produce a structural report on a manuscript file."""
    path = Path(filepath)
    if not path.exists():
        return f"[File not found: {filepath}]"

    text = path.read_text(encoding="utf-8")

    if len(text) > 50000:
        text = text[:50000] + "\n\n[...TRUNCATED...]"

    prompt = f"""Analyze this manuscript excerpt and produce a structural report.

TEXT:
{text}

Report should include:
- Word count (approximate)
- Number of scenes/sections
- POV character(s) identified
- Dialogue vs. prose ratio (estimate)
- Pacing notes (where it speeds up / slows down)
- Named characters mentioned (list them)
- Named locations mentioned (list them)
- Any terms, names, or concepts that seem like worldbuilding elements
- Timeline markers (any references to time passing)

Format as a clean structured report."""

    return call_ollama(LOCAL_WRITER, prompt,
                       system="You are a manuscript analysis tool. Be precise and thorough.")


def cross_reference(filepath: str) -> str:
    """
    Cross-reference a manuscript chapter against the world bible.
    Finds entities in the manuscript and checks them against known lore.
    """
    path = Path(filepath)
    if not path.exists():
        return f"[File not found: {filepath}]"

    text = path.read_text(encoding="utf-8")
    if len(text) > 40000:
        text = text[:40000] + "\n[...TRUNCATED...]"

    # Step 1: Extract entities with Qwen
    extract_prompt = f"""Extract all named entities from this text. Return a JSON object:
{{
  "characters": ["name1", "name2"],
  "locations": ["place1", "place2"],
  "concepts": ["magic term", "cultural reference"]
}}

TEXT:
{text}

Return ONLY the JSON. No explanation."""

    entity_result = call_ollama(LOCAL_WRITER, extract_prompt, temperature=0.1)

    try:
        start = entity_result.index("{")
        end = entity_result.rindex("}") + 1
        entities = json.loads(entity_result[start:end])
    except (ValueError, json.JSONDecodeError):
        entities = {"characters": [], "locations": [], "concepts": []}

    # Step 2: Look up each entity in the world bible
    all_terms = (entities.get("characters", []) +
                 entities.get("locations", []) +
                 entities.get("concepts", []))

    found_context = []
    missing = []
    for term in all_terms:
        matches = world_index.find(term, max_results=2)
        if matches:
            for f, score in matches:
                if score >= 0.5:
                    content = read_world_file(f)
                    found_context.append(f"--- {f} (matched: '{term}') ---\n{content[:2000]}\n")
                    break
            else:
                missing.append(term)
        else:
            missing.append(term)

    # Step 3: Ask Qwen to compare
    if not found_context:
        report = f"No world bible entries found for entities in {filepath}.\n"
        report += f"Entities detected: {', '.join(all_terms)}\n"
        report += "Consider creating world bible entries for these."
        return report

    bible_text = "\n".join(found_context[:10])  # limit context size

    check_prompt = f"""Compare this manuscript text against the world bible entries.
Flag any contradictions, inconsistencies, or missing details.

MANUSCRIPT ({filepath}):
{text[:15000]}

WORLD BIBLE ENTRIES:
{bible_text}

Check for:
- Character descriptions that don't match their world bible entry
- Locations described differently than established
- Timeline conflicts
- Magic/system rule violations
- Name spelling differences

Also note any entities in the manuscript that have NO world bible entry yet.

Missing from world bible: {', '.join(missing) if missing else 'None'}

Format as a clear report."""

    return call_ollama(LOCAL_WRITER, check_prompt,
                       system="You are a continuity checker. Be meticulous.", temperature=0.2)


def consistency_check(scope: str = "all") -> str:
    """
    Scan the world bible for contradictions.
    Uses category-based chunking to handle large bibles.
    """
    categories = defaultdict(list)
    for path in sorted(WORLD_BIBLE_DIR.rglob("*.md")):
        if path.name == "session_log.md":
            continue
        category = path.parent.name
        content = path.read_text(encoding="utf-8")
        categories[category].append(
            f"\n=== {path.relative_to(WORLD_BIBLE_DIR)} ===\n{content}"
        )

    if not categories:
        return "World bible is empty. Nothing to check."

    # Check within each category, then cross-category for characters/locations
    all_issues = []

    for cat, entries in categories.items():
        combined = "\n".join(entries)
        if len(combined) > 50000:
            combined = combined[:50000] + "\n[...TRUNCATED...]"

        if len(entries) < 2:
            continue  # can't have contradictions with one file

        prompt = f"""Review these {cat} entries for internal contradictions.

{combined}

Check for:
- Details described differently across entries
- Timeline contradictions
- Relationships described inconsistently
- Names spelled differently
- Rules that contradict themselves

List each issue with FILES, ISSUE, and SEVERITY (high/medium/low).
If no issues found, say "No issues in {cat}." """

        result = call_ollama(LOCAL_WRITER, prompt,
                             system="You are a continuity checker for fiction. Be meticulous.",
                             temperature=0.2)
        if not is_error(result):
            all_issues.append(f"\n── {cat.upper()} ──\n{result}")

    # Cross-category check: characters ↔ locations ↔ history
    cross_cats = ["characters", "locations", "history"]
    cross_entries = []
    for cat in cross_cats:
        for entry in categories.get(cat, []):
            cross_entries.append(entry)

    if len(cross_entries) >= 2:
        cross_combined = "\n".join(cross_entries)
        if len(cross_combined) > 50000:
            cross_combined = cross_combined[:50000] + "\n[...TRUNCATED...]"

        prompt = f"""Check for contradictions BETWEEN these categories (characters, locations, history).

{cross_combined}

Look for:
- A character's backstory conflicting with historical events
- Character locations conflicting with location descriptions
- Timeline mismatches between character and history entries

List each issue with FILES, ISSUE, and SEVERITY. Say "No cross-category issues" if clean."""

        result = call_ollama(LOCAL_WRITER, prompt,
                             system="You are a continuity checker for fiction.", temperature=0.2)
        if not is_error(result):
            all_issues.append(f"\n── CROSS-CATEGORY ──\n{result}")

    return "\n".join(all_issues) if all_issues else "No contradictions found."


# ═══════════════════════════════════════════════════════════════════════════════
#  WORLDBUILDING PIPELINE  (Claude creates → filed to world bible)
# ═══════════════════════════════════════════════════════════════════════════════

def worldbuild(topic: str, category: str, details: str = "") -> tuple[str, str]:
    """
    Full worldbuilding pipeline:
    1. Claude develops the concept creatively (with existing context)
    2. Files it into the world bible
    3. Returns (content, filepath) — guards against saving errors
    """
    context = auto_context(f"{topic} {category} {details}")

    prompt = f"""Develop this worldbuilding element:

CATEGORY: {category}
TOPIC: {topic}
{"ADDITIONAL DETAILS: " + details if details else ""}

EXISTING CONTEXT:
{context if context else "(Starting fresh — no related entries yet.)"}

Create a rich, detailed entry that is internally consistent with existing lore.
Include all relevant details a writer would need to reference this element.
Format with clear sections using markdown headers."""

    creative_result = call_creative(prompt, system=CREATIVE_SYSTEM)

    # Guard: don't save error messages as world bible entries
    if is_error(creative_result):
        return creative_result, ""

    filename = topic.lower().replace(" ", "_").replace("'", "") + ".md"
    filepath = f"{category}/{filename}"

    # Check if file already exists — offer to update or create new
    existing = read_world_file(filepath)
    if not existing.startswith("[File not found"):
        print(f"\n  ⚠ {filepath} already exists!")
        choice = input("  (o)verwrite / (m)erge / (n)ew name / (c)ancel: ").strip().lower()
        if choice == "c":
            return creative_result, ""
        elif choice == "n":
            new_name = input("  New filename (without .md): ").strip()
            if new_name:
                filepath = f"{category}/{new_name}.md"
            else:
                return creative_result, ""
        elif choice == "m":
            timestamp = datetime.now().strftime("%Y-%m-%d")
            merged = existing + f"\n\n---\n\n## Expanded — {timestamp}\n\n{creative_result}\n"
            creative_result = merged

    write_world_file(filepath, creative_result)

    output = f"""
{'='*60}
WORLDBUILDING: {topic}
{'='*60}

{creative_result}

{'='*60}
✓ Filed to: world_bible/{filepath}
{'='*60}"""

    return output, filepath


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORT — combine manuscripts into a single document
# ═══════════════════════════════════════════════════════════════════════════════

def export_manuscript(pattern: str = "*.md", output_name: str = None) -> str:
    """Concatenate manuscript files in sorted order into a single output file."""
    files = sorted(MANUSCRIPTS_DIR.glob(pattern))
    if not files:
        return f"No files matching '{pattern}' in {MANUSCRIPTS_DIR}"

    parts = []
    for f in files:
        content = f.read_text(encoding="utf-8")
        parts.append(f"\n\n{'='*60}\n{f.name}\n{'='*60}\n\n{content}")

    combined = "\n".join(parts)

    if not output_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_name = f"manuscript_export_{timestamp}.md"

    out_path = OUTPUT_DIR / output_name
    out_path.write_text(combined, encoding="utf-8")
    word_count = len(combined.split())
    return f"Exported {len(files)} files ({word_count:,} words) → {out_path}"


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════════════════════

HELP_TEXT = """
╔══════════════════════════════════════════════════════════════╗
║           CREATIVE WRITING WORKFLOW — COMMANDS              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  CREATIVE (uses Claude API — auto-loads context):            ║
║    write <prompt>         Free-form creative request         ║
║    dialogue               Guided dialogue scene builder      ║
║    feedback               Get critique on a piece of writing ║
║    nextsteps              Suggest where the story goes next  ║
║    worldbuild             Create a new world bible entry     ║
║                                                              ║
║  MEMORY & FILES:                                             ║
║    index                  Show world bible contents          ║
║    read <path>            Read a world bible file            ║
║    search <term>          Search across all world bible text ║
║    edit <path>            Open a file for manual editing     ║
║    move <from> <to>       Move/rename a world bible file     ║
║    merge <f1> <f2> <dst>  Merge files into one               ║
║    delete <path>          Delete a world bible file          ║
║    new <category/name>    Create a blank world bible file    ║
║    update <path>          Manually update a file's content   ║
║                                                              ║
║  ANALYSIS (uses Qwen 3.5 locally):                           ║
║    analyze <filepath>     Structural analysis of manuscript  ║
║    crossref <filepath>    Check manuscript vs world bible    ║
║    consistency            Check world bible for conflicts    ║
║    organize <dir>         Propose file organization plan     ║
║                                                              ║
║  OUTPUT:                                                     ║
║    export [pattern]       Combine manuscripts into one file  ║
║    save                   Save last Claude output to a file  ║
║                                                              ║
║  GENERAL:                                                    ║
║    backend [claude|local]   Switch creative AI backend       ║
║    status                 Check model connectivity + stats   ║
║    reindex                Rebuild the world bible index      ║
║    help                   Show this message                  ║
║    quit                   Exit                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def check_status():
    """Check connectivity to all models and show session stats."""
    print("\nChecking connections...\n")

    # Ollama
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  ✓ Ollama is running at {OLLAMA_URL}")
        print(f"    Available models: {', '.join(models) if models else '(none pulled)'}")
        if LOCAL_WRITER not in " ".join(models):
            print(f"    ⚠ {LOCAL_WRITER} not found — run: ollama pull {LOCAL_WRITER}")
        if LOCAL_VISION not in " ".join(models):
            print(f"    ⚠ {LOCAL_VISION} not found — run: ollama pull {LOCAL_VISION}")
    except Exception:
        print(f"  ✗ Ollama not reachable at {OLLAMA_URL}")
        print(f"    Start it with: ollama serve")

    # Claude
    if CLAUDE_API_KEY:
        print(f"  ✓ Claude API key is set")
    else:
        print(f"  ✗ ANTHROPIC_API_KEY not set")
        print(f"    Run: export ANTHROPIC_API_KEY=sk-ant-...")

    # Index stats
    print(f"\n  World Bible: {WORLD_BIBLE_DIR.resolve()}")
    print(f"  Manuscripts: {MANUSCRIPTS_DIR.resolve()}")
    print(f"  Output:      {OUTPUT_DIR.resolve()}")
    print(f"  Indexed files: {world_index.file_count}")
    print(f"  Indexed terms: {world_index.term_count}")

    # Backend
    if creative_backend == "local":
        print(f"\n  Creative backend: LOCAL ({LOCAL_CREATIVE}) — free, unlimited")
    else:
        print(f"\n  Creative backend: CLAUDE API ({CLAUDE_MODEL}) — pay-per-token")

    # Session stats
    if tokens.api_calls > 0:
        print(f"\n  Session usage:")
        print(tokens.summary())


def offer_save(content: str, default_dir: Path = OUTPUT_DIR) -> str | None:
    """Offer to save content to a file. Returns path if saved."""
    save = input("\nSave this? (y/n): ").strip().lower()
    if save == "y":
        name = input("Filename (e.g. ch3_scene): ").strip()
        if not name:
            name = f"output_{datetime.now().strftime('%H%M%S')}"
        if not name.endswith(".md"):
            name += ".md"
        path = default_dir / name
        path.write_text(content, encoding="utf-8")
        print(f"  ✓ Saved to {path}")
        return str(path)
    return None


def interactive_dialogue():
    """Guided dialogue scene builder."""
    print("\n── Dialogue Scene Builder ──")
    chars = input("Characters (comma-separated): ").strip().split(",")
    chars = [c.strip() for c in chars if c.strip()]
    situation = input("Situation: ").strip()
    tone = input("Tone/mood (or press Enter to skip): ").strip()

    if not chars or not situation:
        print("Need at least characters and a situation.")
        return

    print(f"\nBuilding scene with {', '.join(chars)}...")
    result = write_dialogue(chars, situation, tone)

    if is_error(result):
        print(result)
        return

    print(result)

    # Log the exchange
    session_log.record(
        f"dialogue: {', '.join(chars)}",
        f"Scene: {situation}",
        files_touched=[]
    )

    # Offer save
    saved_path = offer_save(result)

    # Offer write-back if new lore
    updated = offer_writeback(result, " ".join(chars) + " " + situation)
    if updated:
        session_log.record("writeback", f"Updated: {', '.join(updated)}", updated)


def interactive_feedback():
    """Guided feedback session."""
    print("\n── Editorial Feedback ──")
    source = input("Paste text (then type END), or enter a filepath: ").strip()

    if source and Path(source).exists():
        text = Path(source).read_text(encoding="utf-8")
        print(f"  Loaded {len(text.split())} words from {source}")
    else:
        if source:
            # They started typing — treat first line as text
            lines = [source]
        else:
            lines = []
        print("(Type END on a new line when done)")
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        text = "\n".join(lines)

    if not text.strip():
        print("No text provided.")
        return

    focus = input("Focus area (general/dialogue/pacing/prose-style/character-voice/tension/worldbuilding): ").strip()
    if not focus:
        focus = "general"

    print(f"\nAnalyzing ({focus})...\n")
    result = get_feedback(text, focus)
    print(result)

    session_log.record(f"feedback ({focus})", result[:200])


def interactive_worldbuild():
    """Guided worldbuilding session."""
    print("\n── Worldbuilding ──")
    categories = ["characters", "locations", "history", "magic_systems",
                   "cultures", "languages", "plot_outlines", "notes"]
    print("Categories:", ", ".join(categories))
    category = input("Category: ").strip()
    if category not in categories:
        # Allow creating new categories
        confirm = input(f"  '{category}' is new. Create it? (y/n): ").strip().lower()
        if confirm != "y":
            print(f"Choose from: {', '.join(categories)}")
            return
        (WORLD_BIBLE_DIR / category).mkdir(exist_ok=True)

    topic = input("Topic/Name: ").strip()
    if not topic:
        print("Need a topic.")
        return

    details = input("Additional details (or Enter to skip): ").strip()

    print(f"\nBuilding {topic} in {category}...")
    result, filepath = worldbuild(topic, category, details)

    if is_error(result):
        print(result)
        return

    print(result)
    session_log.record(
        f"worldbuild: {topic}",
        f"Created {category} entry",
        [filepath] if filepath else [],
    )


def interactive_file_command(cmd: str, arg: str):
    """Handle flexible file management commands."""

    if cmd == "move":
        parts = arg.split()
        if len(parts) < 2:
            print("Usage: move <from_path> <to_path>")
            print("Example: move characters/old_name.md characters/new_name.md")
            return
        from_path, to_path = parts[0], parts[1]
        print(move_world_file(from_path, to_path))
        session_log.record(f"move {from_path}", f"→ {to_path}", [to_path])

    elif cmd == "merge":
        parts = arg.split()
        if len(parts) < 3:
            print("Usage: merge <file1> <file2> <destination>")
            print("Example: merge characters/draft1.md characters/draft2.md characters/combined.md")
            return
        sources = parts[:-1]
        dest = parts[-1]
        delete = input("Delete source files after merge? (y/n): ").strip().lower() == "y"
        print(merge_world_files(sources, dest, delete_sources=delete))
        session_log.record(f"merge → {dest}", f"Merged {len(sources)} files", [dest])

    elif cmd == "delete":
        if not arg:
            print("Usage: delete <path>")
            return
        confirm = input(f"  Delete world_bible/{arg}? This cannot be undone. (y/n): ").strip().lower()
        if confirm == "y":
            print(delete_world_file(arg))
            session_log.record(f"delete {arg}", "File deleted")

    elif cmd == "new":
        if not arg:
            print("Usage: new <category/filename>")
            print("Example: new characters/new_ally")
            return
        if not arg.endswith(".md"):
            arg += ".md"
        existing = read_world_file(arg)
        if not existing.startswith("[File not found"):
            print(f"  {arg} already exists. Use 'edit' or 'update' to modify it.")
            return
        # Create with a template
        name = Path(arg).stem.replace("_", " ").title()
        category = Path(arg).parent.name.replace("_", " ").title()
        template = f"# {name}\n\n*{category} entry — created {datetime.now().strftime('%Y-%m-%d')}*\n\n## Overview\n\n(Write your content here.)\n"
        write_world_file(arg, template)
        print(f"  ✓ Created world_bible/{arg}")
        print(f"  Edit it directly or use 'update {arg}' to add content.")
        session_log.record(f"new {arg}", f"Created blank entry", [arg])

    elif cmd == "update":
        if not arg:
            print("Usage: update <path>")
            return
        existing = read_world_file(arg)
        if existing.startswith("[File not found"):
            print(existing)
            return

        print(f"\nCurrent content of {arg}:")
        print("─" * 40)
        # Show first 30 lines
        lines = existing.splitlines()
        for line in lines[:30]:
            print(f"  {line}")
        if len(lines) > 30:
            print(f"  ... ({len(lines) - 30} more lines)")
        print("─" * 40)

        print("\nEnter new content to APPEND (type END when done):")
        new_lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            new_lines.append(line)

        if new_lines:
            new_content = "\n".join(new_lines)
            timestamp = datetime.now().strftime("%Y-%m-%d")
            updated = existing + f"\n\n---\n\n## Update — {timestamp}\n\n{new_content}\n"
            write_world_file(arg, updated)
            print(f"  ✓ Updated {arg}")
            session_log.record(f"update {arg}", new_content[:100], [arg])
        else:
            print("  No changes made.")

    elif cmd == "edit":
        if not arg:
            print("Usage: edit <path>")
            return
        full_path = WORLD_BIBLE_DIR / arg
        if not full_path.exists():
            print(f"[File not found: {arg}]")
            return

        print(f"\nFull content of {arg}:")
        print("─" * 40)
        print(read_world_file(arg))
        print("─" * 40)

        choice = input("\n(r)eplace all / (a)ppend / (q)uit: ").strip().lower()
        if choice == "r":
            print("Enter new content (type END when done):")
            new_lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                new_lines.append(line)
            if new_lines:
                write_world_file(arg, "\n".join(new_lines))
                print(f"  ✓ Replaced content of {arg}")
                session_log.record(f"edit {arg}", "Full replacement", [arg])
        elif choice == "a":
            print("Enter content to append (type END when done):")
            new_lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                new_lines.append(line)
            if new_lines:
                existing = read_world_file(arg)
                write_world_file(arg, existing + "\n\n" + "\n".join(new_lines))
                print(f"  ✓ Appended to {arg}")
                session_log.record(f"edit {arg}", "Appended content", [arg])


def main():
    global creative_backend
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         CREATIVE WRITING AGENTIC WORKFLOW                    ║
║         Claude + Qwen 3.5:4B + Gemma 4 E4B                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print(f"  World bible: {world_index.file_count} files indexed, {world_index.term_count} terms")
    backend_label = f"LOCAL ({LOCAL_CREATIVE})" if creative_backend == "local" else f"CLAUDE API"
    print(f"  Creative backend: {backend_label}  (change with 'backend' command)")
    print(f"  Type 'help' for commands, 'status' to check connections.\n")

    last_output = ""  # track last Claude output for 'save' command

    while True:
        try:
            raw = input("\n🖊  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            if tokens.api_calls > 0:
                print(tokens.summary())
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit"):
            print("Goodbye.")
            if tokens.api_calls > 0:
                print("\nSession stats:")
                print(tokens.summary())
            break

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "status":
            check_status()

        elif cmd == "reindex":
            world_index.rebuild()
            print(f"  ✓ Rebuilt index: {world_index.file_count} files, {world_index.term_count} terms")

        elif cmd == "backend":
            if arg in ("claude", "api"):
                creative_backend = "claude"
                print(f"  ✓ Creative backend: CLAUDE API ({CLAUDE_MODEL})")
                print(f"    Best quality. Costs per token.")
            elif arg in ("local", "free", "ollama"):
                creative_backend = "local"
                print(f"  ✓ Creative backend: LOCAL ({LOCAL_CREATIVE})")
                print(f"    Free and unlimited. Quality depends on model size.")
                print(f"    Current model: {LOCAL_CREATIVE}")
                print(f"    To use a larger model: edit LOCAL_CREATIVE in orchestrator.py")
            else:
                print(f"  Current backend: {creative_backend.upper()}")
                if creative_backend == "claude":
                    print(f"    Model: {CLAUDE_MODEL} (pay-per-token)")
                else:
                    print(f"    Model: {LOCAL_CREATIVE} (free, unlimited)")
                print(f"\n  Switch with:")
                print(f"    backend claude   — best quality, API costs")
                print(f"    backend local    — free, unlimited, runs on your machine")

        elif cmd == "index":
            idx = list_world_bible()
            if not idx:
                print("World bible is empty. Use 'worldbuild' to create entries.")
            else:
                for cat, files in idx.items():
                    print(f"\n  {cat}/")
                    for f in files:
                        print(f"    {f['name']}  ({f['size']} bytes)")

        elif cmd == "read":
            if not arg:
                print("Usage: read <path>  (e.g., read characters/arya.md)")
            else:
                print(read_world_file(arg))

        elif cmd == "search":
            if not arg:
                print("Usage: search <term or pattern>")
                print("Example: search unicorn")
                print("Example: search \"golden horn\"")
            else:
                results = world_index.search_content(arg)
                if not results:
                    print(f"  No matches for '{arg}' in world bible.")
                else:
                    total_matches = sum(len(lines) for _, lines in results)
                    print(f"  Found {total_matches} matches in {len(results)} files:\n")
                    for filepath, lines in results:
                        title = world_index.file_titles.get(filepath, filepath)
                        print(f"  ── {title} ({filepath}) ──")
                        for line in lines[:5]:  # show max 5 per file
                            print(f"  {line}")
                        if len(lines) > 5:
                            print(f"    ... and {len(lines) - 5} more matches")
                        print()

        elif cmd == "write":
            if not arg:
                print("Usage: write <your creative request>")
                print("Example: write a tense scene where the rebel leader confronts the king")
            else:
                print()
                result = creative_task(arg)
                if is_error(result):
                    print(result)
                else:
                    print(result)
                    last_output = result
                    session_log.record(f"write", arg[:100])
                    offer_save(result)
                    offer_writeback(result, arg)

        elif cmd == "dialogue":
            interactive_dialogue()

        elif cmd == "feedback":
            interactive_feedback()

        elif cmd == "nextsteps":
            state = arg if arg else input("Describe where your story is right now: ").strip()
            if state:
                print("\nAnalyzing story state...\n")
                result = suggest_next_steps(state)
                if not is_error(result):
                    print(result)
                    last_output = result
                    session_log.record("nextsteps", result[:200])
                else:
                    print(result)

        elif cmd == "worldbuild":
            interactive_worldbuild()

        elif cmd == "analyze":
            if not arg:
                print("Usage: analyze <filepath>")
                print("Example: analyze manuscripts/chapter1.md")
            else:
                print(f"\nAnalyzing {arg} with Qwen 3.5...\n")
                result = analyze_manuscript(arg)
                print(result)
                session_log.record(f"analyze {arg}", result[:200])

        elif cmd == "crossref":
            if not arg:
                print("Usage: crossref <filepath>")
                print("Example: crossref manuscripts/chapter1.md")
                print("Checks the manuscript against world bible for contradictions.")
            else:
                print(f"\nCross-referencing {arg} against world bible...\n")
                result = cross_reference(arg)
                print(result)
                session_log.record(f"crossref {arg}", result[:200])

        elif cmd == "consistency":
            print("\nRunning consistency check across world bible...\n")
            result = consistency_check()
            print(result)
            session_log.record("consistency", result[:200])

        elif cmd == "organize":
            if not arg:
                print("Usage: organize <directory>")
                print("Example: organize ./manuscripts")
            else:
                instruction = input("What should be done? ").strip()
                if instruction:
                    print(f"\nAnalyzing {arg}...\n")
                    result = organize_files(arg, instruction)
                    print(result)

        elif cmd == "export":
            pattern = arg if arg else "*.md"
            output_name = None
            if " " in arg:
                parts = arg.split(maxsplit=1)
                pattern = parts[0]
                output_name = parts[1]
            print(export_manuscript(pattern, output_name))

        elif cmd == "save":
            if last_output:
                offer_save(last_output)
            else:
                print("Nothing to save yet.")

        elif cmd in ("move", "merge", "delete", "new", "update", "edit"):
            interactive_file_command(cmd, arg)

        else:
            # Default: treat as a creative request
            print("(Treating as creative request — use 'help' for commands)\n")
            result = creative_task(raw)
            if not is_error(result):
                print(result)
                last_output = result
                session_log.record("creative", raw[:100])
                offer_save(result)
                offer_writeback(result, raw)
            else:
                print(result)


if __name__ == "__main__":
    main()
