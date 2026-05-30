#!/usr/bin/env python3
"""
Creative Writing Workshop — Standalone Application v2.0
=========================================================
Double-click or run this file. Everything else is automatic.

Models used:
  - mistral-small3.2 (24B) → creative writing (free, local)
  - qwen3.5:4b             → structural tasks (analysis, filing, consistency)
  - nomic-embed-text        → semantic search over world bible
  - Claude API (optional)   → highest quality creative output

Optional tools (auto-installed on first use):
  - spaCy          → named entity extraction for cross-referencing
  - LanguageTool   → grammar/style checking (requires Java)
  - textstat       → readability metrics
  - Kokoro TTS     → hear your writing read aloud (requires espeak-ng)
"""

# ─── Bootstrap ────────────────────────────────────────────────────────────────
import subprocess, sys, os

def _ensure_package(name, pip_name=None):
    try:
        __import__(name)
        return True
    except ImportError:
        pip_name = pip_name or name
        print(f"  Installing {pip_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", pip_name,
                "--break-system-packages", "--quiet"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            print(f"  ⚠ Could not install {pip_name} (optional)")
            return False

print("\n  Checking dependencies...")
_ensure_package("requests")
_ensure_package("textstat")
print("  ✓ Core ready\n")

# ─── Imports ──────────────────────────────────────────────────────────────────
import re
import json
import math
import shutil
import time
import signal
import socket
import hashlib
import threading
import webbrowser
import requests
import mimetypes
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

APP_VERSION = "2.0"

APP_DIR = Path(__file__).resolve().parent
WORLD_BIBLE_DIR = APP_DIR / "world_bible"
MANUSCRIPTS_DIR = APP_DIR / "manuscripts"
OUTPUT_DIR = APP_DIR / "output"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Model roles
CREATIVE_MODEL = "mistral-small3.2"   # 24B — creative writing
STRUCTURAL_MODEL = "qwen3.5:4b"       # 4B — analysis, filing, consistency
EMBED_MODEL = "nomic-embed-text"       # embedding model for semantic search

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"

WEB_HOST = "127.0.0.1"

class AppState:
    creative_backend = "local"
    ollama_ready = False
    creative_model_ready = False
    structural_model_ready = False
    embed_model_ready = False
    models_pulling = False
    pull_progress = ""
    startup_errors = []
    # Optional tools
    has_spacy = False
    has_languagetool = False
    has_kokoro = False

state = AppState()
if CLAUDE_API_KEY:
    state.creative_backend = "claude"

# ─── Directories ──────────────────────────────────────────────────────────────
for d in [WORLD_BIBLE_DIR, MANUSCRIPTS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for sub in ["characters", "locations", "history", "magic_systems",
            "cultures", "languages", "plot_outlines", "notes"]:
    (WORLD_BIBLE_DIR / sub).mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  OLLAMA MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def find_ollama():
    result = shutil.which("ollama")
    if result:
        return result
    candidates = []
    if sys.platform == "darwin":
        candidates = ["/usr/local/bin/ollama",
                      "/Applications/Ollama.app/Contents/Resources/ollama"]
    elif sys.platform == "win32":
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "ollama.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama.exe"]
    else:
        candidates = ["/usr/local/bin/ollama", "/usr/bin/ollama",
                      Path.home() / ".local" / "bin" / "ollama"]
    for p in candidates:
        if Path(p).exists():
            return str(p)
    return None

def is_ollama_running():
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False

def start_ollama(binary_path):
    try:
        kw = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kw["start_new_session"] = True
        subprocess.Popen([binary_path, "serve"], **kw)
        for _ in range(30):
            time.sleep(1)
            if is_ollama_running():
                return True
        return False
    except Exception as e:
        state.startup_errors.append(f"Failed to start Ollama: {e}")
        return False

def has_model(model_name):
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(model_name in m for m in models)
    except Exception:
        return False

def pull_models_background():
    """Pull all needed models in background."""
    state.models_pulling = True
    models_to_pull = []

    if not has_model(STRUCTURAL_MODEL):
        models_to_pull.append(STRUCTURAL_MODEL)
    else:
        state.structural_model_ready = True

    if not has_model(EMBED_MODEL):
        models_to_pull.append(EMBED_MODEL)
    else:
        state.embed_model_ready = True

    if not has_model(CREATIVE_MODEL):
        models_to_pull.append(CREATIVE_MODEL)
    else:
        state.creative_model_ready = True

    for model in models_to_pull:
        state.pull_progress = f"Pulling {model}..."
        try:
            r = requests.post(f"{OLLAMA_URL}/api/pull",
                              json={"name": model, "stream": True},
                              stream=True, timeout=1800)
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if "total" in data and "completed" in data and data["total"] > 0:
                        pct = int(data["completed"] / data["total"] * 100)
                        state.pull_progress = f"{model}: {status} — {pct}%"
                    else:
                        state.pull_progress = f"{model}: {status}"
            if model == STRUCTURAL_MODEL:
                state.structural_model_ready = True
            elif model == EMBED_MODEL:
                state.embed_model_ready = True
            elif model == CREATIVE_MODEL:
                state.creative_model_ready = True
        except Exception as e:
            state.startup_errors.append(f"Pull {model} failed: {e}")

    state.pull_progress = "All models ready"
    state.models_pulling = False

def ensure_ollama_and_model():
    print("  Looking for Ollama...")
    if is_ollama_running():
        print("  ✓ Ollama is already running")
        state.ollama_ready = True
    else:
        binary = find_ollama()
        if binary:
            print(f"  Found Ollama at {binary}, starting...")
            if start_ollama(binary):
                print("  ✓ Ollama started")
                state.ollama_ready = True
            else:
                state.startup_errors.append("Could not start Ollama.")
                return
        else:
            state.startup_errors.append("Ollama not found. Install from https://ollama.com")
            return

    # Check models and pull missing ones in background
    all_ready = (has_model(STRUCTURAL_MODEL) and
                 has_model(EMBED_MODEL) and
                 has_model(CREATIVE_MODEL))
    if all_ready:
        state.structural_model_ready = True
        state.embed_model_ready = True
        state.creative_model_ready = True
        print(f"  ✓ All models available")
    else:
        missing = []
        if not has_model(STRUCTURAL_MODEL): missing.append(STRUCTURAL_MODEL)
        if not has_model(EMBED_MODEL): missing.append(EMBED_MODEL)
        if not has_model(CREATIVE_MODEL): missing.append(CREATIVE_MODEL)
        # Mark available ones
        if has_model(STRUCTURAL_MODEL): state.structural_model_ready = True
        if has_model(EMBED_MODEL): state.embed_model_ready = True
        if has_model(CREATIVE_MODEL): state.creative_model_ready = True
        print(f"  ↓ Pulling in background: {', '.join(missing)}")
        threading.Thread(target=pull_models_background, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  TOKEN TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class TokenTracker:
    def __init__(self):
        self.input_chars = 0; self.output_chars = 0; self.api_calls = 0
    def record(self, inp, out):
        self.input_chars += len(inp); self.output_chars += len(out); self.api_calls += 1
    @property
    def input_tokens(self): return self.input_chars // 4
    @property
    def output_tokens(self): return self.output_chars // 4
    @property
    def cost(self):
        return (self.input_tokens * 3 / 1_000_000) + (self.output_tokens * 15 / 1_000_000)
    def to_dict(self):
        return {"api_calls": self.api_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "cost": f"${self.cost:.4f}"}

tokens = TokenTracker()


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING INDEX — semantic search over world bible
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingIndex:
    """
    Semantic search using nomic-embed-text via Ollama.
    Stores embeddings in a cache file next to the world bible.
    Falls back gracefully if Ollama or model unavailable.
    """
    def __init__(self, bible_dir: Path):
        self.bible_dir = bible_dir
        self.cache_path = bible_dir / ".embeddings_cache.json"
        self.embeddings: dict[str, dict] = {}  # path -> {hash, vector}
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                self.embeddings = json.loads(
                    self.cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, Exception):
                self.embeddings = {}

    def _save_cache(self):
        try:
            self.cache_path.write_text(
                json.dumps(self.embeddings), encoding="utf-8")
        except Exception:
            pass

    def _file_hash(self, path: Path) -> str:
        content = path.read_text(encoding="utf-8", errors="replace")
        return hashlib.md5(content.encode()).hexdigest()

    def _embed(self, text: str) -> list[float] | None:
        try:
            r = requests.post(f"{OLLAMA_URL}/api/embed",
                              json={"model": EMBED_MODEL, "input": text},
                              timeout=30)
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings", [])
            return embs[0] if embs else None
        except Exception:
            return None

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def rebuild(self):
        """Re-embed all files. Called on startup or manually."""
        if not state.embed_model_ready:
            return
        updated = 0
        for path in sorted(self.bible_dir.rglob("*.md")):
            if path.name == "session_log.md" or path.name.startswith("."):
                continue
            rel = str(path.relative_to(self.bible_dir))
            file_hash = self._file_hash(path)
            # Skip if unchanged
            if rel in self.embeddings and self.embeddings[rel].get("hash") == file_hash:
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            # Embed title + first 2000 chars (enough for semantic matching)
            snippet = f"{path.stem.replace('_',' ')} {path.parent.name} {content[:2000]}"
            vec = self._embed(snippet)
            if vec:
                self.embeddings[rel] = {"hash": file_hash, "vector": vec}
                updated += 1
        # Remove deleted files
        existing = {str(p.relative_to(self.bible_dir))
                    for p in self.bible_dir.rglob("*.md")}
        for key in list(self.embeddings.keys()):
            if key not in existing:
                del self.embeddings[key]
        if updated > 0:
            self._save_cache()
        return updated

    def reindex_file(self, relative_path: str):
        """Re-embed a single file after a write."""
        full = self.bible_dir / relative_path
        if not full.exists():
            self.embeddings.pop(relative_path, None)
            self._save_cache()
            return
        if not state.embed_model_ready:
            return
        content = full.read_text(encoding="utf-8", errors="replace")
        snippet = f"{full.stem.replace('_',' ')} {full.parent.name} {content[:2000]}"
        vec = self._embed(snippet)
        if vec:
            self.embeddings[relative_path] = {
                "hash": self._file_hash(full), "vector": vec}
            self._save_cache()

    def find(self, query: str, max_results: int = 8) -> list[tuple[str, float]]:
        """Semantic search. Returns [(filepath, score)]."""
        if not self.embeddings:
            return []
        q_vec = self._embed(query)
        if not q_vec:
            return []
        scores = []
        for path, data in self.embeddings.items():
            vec = data.get("vector")
            if vec:
                sim = self._cosine_sim(q_vec, vec)
                scores.append((path, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(p, s) for p, s in scores[:max_results] if s > 0.3]

    @property
    def count(self):
        return len(self.embeddings)


# ═══════════════════════════════════════════════════════════════════════════════
#  KEYWORD INDEX — fast fallback when embeddings unavailable
# ═══════════════════════════════════════════════════════════════════════════════

class WorldIndex:
    STOP_WORDS = {
        "the","a","an","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "shall","can","need","must","and","but","or","nor","not","so","yet",
        "both","either","neither","each","every","all","any","few","more","most",
        "other","some","such","no","only","own","same","than","too","very",
        "just","about","above","after","again","against","below","between","by",
        "down","during","for","from","in","into","of","off","on","out","over",
        "through","to","under","until","up","with","that","this","these","those",
        "it","its","he","she","they","them","his","her","their","him","we",
        "you","your","my","our","who","whom","which","what","where","when",
        "how","why","if","then","else","also","as","at","like","one","two",
        "three","said","says","well","back","even","still","new","now","way",
        "many","much","make","made","know","known","see","seen","think","come",
        "take","get","got","go","went","here","there","write","scene",
    }

    def __init__(self, bible_dir: Path):
        self.bible_dir = bible_dir
        self.term_to_files: dict[str, set[str]] = defaultdict(set)
        self.file_to_terms: dict[str, set[str]] = {}
        self.file_titles: dict[str, str] = {}
        self.rebuild()

    def _tokenize(self, text):
        words = re.findall(r"[a-zA-Z']+", text.lower())
        terms = set()
        for w in words:
            w = w.strip("'")
            if len(w) >= 3 and w not in self.STOP_WORDS:
                terms.add(w)
        for h in re.findall(r"^#+\s+(.+)$", text, re.MULTILINE):
            n = h.strip().lower()
            if len(n) >= 3: terms.add(n)
        return terms

    def rebuild(self):
        self.term_to_files.clear(); self.file_to_terms.clear(); self.file_titles.clear()
        for path in sorted(self.bible_dir.rglob("*.md")):
            self._index_file(path)

    def _index_file(self, path):
        rel = str(path.relative_to(self.bible_dir))
        content = path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^#+\s+(.+)$", content, re.MULTILINE)
        self.file_titles[rel] = m.group(1).strip() if m else path.stem.replace("_"," ").title()
        terms = self._tokenize(f"{path.stem.replace('_',' ')} {path.parent.name} {content}")
        self.file_to_terms[rel] = terms
        for t in terms: self.term_to_files[t].add(rel)

    def reindex_file(self, relative_path):
        old = self.file_to_terms.pop(relative_path, set())
        for t in old:
            self.term_to_files[t].discard(relative_path)
            if not self.term_to_files[t]: del self.term_to_files[t]
        self.file_titles.pop(relative_path, None)
        full = self.bible_dir / relative_path
        if full.exists(): self._index_file(full)

    def find(self, query, max_results=10):
        query_terms = self._tokenize(query)
        if not query_terms: return []
        scores = defaultdict(float)
        for term in query_terms:
            for f in self.term_to_files.get(term, set()): scores[f] += 1.0
            for it, files in self.term_to_files.items():
                if term != it and (term in it or it in term):
                    for f in files: scores[f] += 0.5
        if not scores: return []
        results = [(f, s / len(query_terms)) for f, s in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def search_content(self, pattern):
        flags = re.IGNORECASE
        try: compiled = re.compile(pattern, flags)
        except re.error: compiled = re.compile(re.escape(pattern), flags)
        results = []
        for path in sorted(self.bible_dir.rglob("*.md")):
            rel = str(path.relative_to(self.bible_dir))
            content = path.read_text(encoding="utf-8", errors="replace")
            matches = []
            for i, line in enumerate(content.splitlines(), 1):
                if compiled.search(line):
                    matches.append({"line": i, "text": line.strip()})
            if matches:
                results.append({"file": rel, "title": self.file_titles.get(rel, rel),
                                "matches": matches[:8]})
        return results

    @property
    def file_count(self): return len(self.file_to_terms)
    @property
    def term_count(self): return len(self.term_to_files)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION LOG
# ═══════════════════════════════════════════════════════════════════════════════

class SessionLog:
    def __init__(self, bible_dir, buffer_size=10):
        self.log_path = bible_dir / "notes" / "session_log.md"
        self.buffer = []; self.buffer_size = buffer_size
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("# Session Log\n\n", encoding="utf-8")
    def record(self, command, summary, files=None):
        entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "command": command, "summary": summary[:200], "files": files or []}
        self.buffer.append(entry)
        if len(self.buffer) > self.buffer_size: self.buffer.pop(0)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n### {entry['time']} — `{command}`\n{summary[:200]}\n")
    def recent_context(self):
        if not self.buffer: return ""
        lines = ["RECENT SESSION ACTIVITY:"]
        for e in self.buffer[-5:]:
            lines.append(f"  [{e['time']}] {e['command']}: {e['summary']}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CLIENTS
# ═══════════════════════════════════════════════════════════════════════════════

def call_ollama(model, prompt, system="", temperature=0.7):
    payload = {"model": model, "messages": [], "stream": False,
               "options": {"temperature": temperature, "num_ctx": 32768}}
    if system: payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot reach Ollama."
    except Exception as e:
        return f"[ERROR] Ollama call failed: {e}"

def call_claude(prompt, system="", temperature=0.7):
    if not CLAUDE_API_KEY:
        return "[ERROR] No Claude API key set."
    headers = {"x-api-key": CLAUDE_API_KEY, "content-type": "application/json",
               "anthropic-version": "2023-06-01"}
    payload = {"model": CLAUDE_MODEL, "max_tokens": 4096, "temperature": temperature,
               "messages": [{"role": "user", "content": prompt}]}
    if system: payload["system"] = system
    full_input = (system or "") + prompt
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
                          headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        result = "".join(b["text"] for b in r.json()["content"] if b["type"] == "text")
        tokens.record(full_input, result)
        return result
    except Exception as e:
        return f"[ERROR] Claude API call failed: {e}"

def call_creative(prompt, system="", temperature=0.7):
    if state.creative_backend == "local":
        return call_ollama(CREATIVE_MODEL, prompt, system=system, temperature=temperature)
    return call_claude(prompt, system=system, temperature=temperature)

def is_error(text): return text.strip().startswith("[ERROR]")


# ═══════════════════════════════════════════════════════════════════════════════
#  INITIALIZE INDEXES
# ═══════════════════════════════════════════════════════════════════════════════

world_index = WorldIndex(WORLD_BIBLE_DIR)
embed_index = EmbeddingIndex(WORLD_BIBLE_DIR)
session_log = SessionLog(WORLD_BIBLE_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORLD BIBLE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def read_world_file(rel):
    full = WORLD_BIBLE_DIR / rel
    return full.read_text(encoding="utf-8") if full.exists() else f"[File not found: {rel}]"

def write_world_file(rel, content):
    full = WORLD_BIBLE_DIR / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    world_index.reindex_file(rel)
    embed_index.reindex_file(rel)
    return f"Saved: {rel}"

def delete_world_file(rel):
    full = WORLD_BIBLE_DIR / rel
    if not full.exists(): return f"[File not found: {rel}]"
    full.unlink()
    world_index.reindex_file(rel)
    embed_index.reindex_file(rel)
    return f"Deleted: {rel}"

def auto_context(user_input, max_chars=30000):
    """Find relevant files using embeddings first, keyword fallback."""
    # Try semantic search first
    matches = embed_index.find(user_input)
    # Fallback to keywords
    if not matches:
        kw_matches = world_index.find(user_input)
        matches = [(f, s) for f, s in kw_matches if s >= 0.3]
    if not matches:
        return "", []
    assembled, loaded, total = [], [], 0
    for filepath, score in matches:
        content = read_world_file(filepath)
        if content.startswith("[File not found"): continue
        entry = f"\n--- {filepath} ---\n{content}\n"
        if total + len(entry) > max_chars: break
        assembled.append(entry)
        loaded.append(filepath)
        total += len(entry)
    return ("".join(assembled), loaded) if assembled else ("", [])

def list_world_bible():
    index = {}
    for path in sorted(WORLD_BIBLE_DIR.rglob("*.md")):
        if path.name == "session_log.md": continue
        cat = path.parent.name
        if cat not in index: index[cat] = []
        index[cat].append({"file": str(path.relative_to(WORLD_BIBLE_DIR)),
                           "name": path.stem.replace("_"," ").title(),
                           "size": path.stat().st_size})
    return index


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL TOOLS — loaded lazily on first use
# ═══════════════════════════════════════════════════════════════════════════════

_spacy_nlp = None
def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("  Downloading spaCy model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download",
                                   "en_core_web_sm", "--quiet"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _spacy_nlp = spacy.load("en_core_web_sm")
        state.has_spacy = True
        return _spacy_nlp
    except Exception:
        # Try installing spacy
        if _ensure_package("spacy"):
            try:
                import spacy
                subprocess.check_call([sys.executable, "-m", "spacy", "download",
                                       "en_core_web_sm", "--quiet"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _spacy_nlp = spacy.load("en_core_web_sm")
                state.has_spacy = True
                return _spacy_nlp
            except Exception:
                pass
    return None

_lang_tool = None
def get_languagetool():
    global _lang_tool
    if _lang_tool is not None:
        return _lang_tool
    try:
        import language_tool_python
        _lang_tool = language_tool_python.LanguageTool("en-US")
        state.has_languagetool = True
        return _lang_tool
    except ImportError:
        if _ensure_package("language_tool_python"):
            try:
                import language_tool_python
                _lang_tool = language_tool_python.LanguageTool("en-US")
                state.has_languagetool = True
                return _lang_tool
            except Exception:
                pass
    except Exception:
        pass
    return None

def readability_stats(text):
    """Get readability metrics using textstat."""
    if not HAS_TEXTSTAT or not text.strip():
        return None
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    result = {
        "word_count": len(words),
        "sentence_count": max(sentences, 1),
        "avg_sentence_length": round(len(words) / max(sentences, 1), 1),
        "reading_time_min": round(len(words) / 250, 1),
    }
    try:
        # textstat may need nltk data — download silently if missing
        try:
            import nltk
            nltk.download('cmudict', quiet=True)
        except Exception:
            pass
        result["flesch_reading_ease"] = round(textstat.flesch_reading_ease(text), 1)
        result["flesch_kincaid_grade"] = round(textstat.flesch_kincaid_grade(text), 1)
        result["gunning_fog"] = round(textstat.gunning_fog(text), 1)
    except Exception:
        # If textstat metrics fail, still return basic stats
        result["flesch_reading_ease"] = None
        result["flesch_kincaid_grade"] = None
        result["gunning_fog"] = None
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CREATIVE SYSTEM PROMPT
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
details, timeline events), note them clearly at the end under a section called
"NEW LORE ESTABLISHED" so the writer can update their files.

Always write in a way that serves the story. Be bold with suggestions but
respect the writer's vision."""


# ═══════════════════════════════════════════════════════════════════════════════
#  API HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def handle_status():
    return {
        "ollama_ready": state.ollama_ready,
        "creative_model_ready": state.creative_model_ready,
        "structural_model_ready": state.structural_model_ready,
        "embed_model_ready": state.embed_model_ready,
        "models_pulling": state.models_pulling,
        "pull_progress": state.pull_progress,
        "backend": state.creative_backend,
        "has_claude_key": bool(CLAUDE_API_KEY),
        "creative_model": CREATIVE_MODEL,
        "structural_model": STRUCTURAL_MODEL,
        "claude_model": CLAUDE_MODEL,
        "files_indexed": world_index.file_count,
        "terms_indexed": world_index.term_count,
        "embeddings_indexed": embed_index.count,
        "token_usage": tokens.to_dict(),
        "startup_errors": state.startup_errors,
        "has_spacy": state.has_spacy,
        "has_languagetool": state.has_languagetool,
    }

def handle_write(data):
    prompt = data.get("prompt", "").strip()
    if not prompt: return {"error": "No prompt provided."}
    context, loaded = auto_context(prompt)
    session_ctx = session_log.recent_context()
    full = prompt
    if context: full = f"WORLD BIBLE CONTEXT:\n{context}\n\n{session_ctx}\n\nTASK:\n{prompt}"
    elif session_ctx: full = f"{session_ctx}\n\nTASK:\n{prompt}"
    result = call_creative(full, system=CREATIVE_SYSTEM)
    session_log.record("write", prompt[:100], loaded)
    return {"result": result, "loaded_files": loaded, "is_error": is_error(result)}

def handle_dialogue(data):
    characters = [c.strip() for c in data.get("characters","").split(",") if c.strip()]
    situation = data.get("situation","").strip()
    tone = data.get("tone","").strip()
    if not characters or not situation: return {"error": "Need characters and a situation."}
    query = " ".join(characters) + " " + situation
    context, loaded = auto_context(query)
    session_ctx = session_log.recent_context()
    tone_note = f"\nTone/mood: {tone}" if tone else ""
    prompt = f"""WORLD BIBLE CONTEXT:\n{context or "(No files found.)"}\n\n{session_ctx}\n
Write a dialogue scene:
- Characters: {', '.join(characters)}
- Situation: {situation}{tone_note}

Requirements:
- Each character must sound distinct and consistent with their established voice
- Include brief action beats and internal reactions between lines
- Subtext matters — what characters DON'T say is as important as what they do
- The scene should advance character relationships or plot

Write the scene."""
    result = call_creative(prompt, system=CREATIVE_SYSTEM)
    session_log.record(f"dialogue: {', '.join(characters)}", situation[:100], loaded)
    return {"result": result, "loaded_files": loaded, "is_error": is_error(result)}

def handle_feedback(data):
    text = data.get("text","").strip()
    focus = data.get("focus","general").strip()
    if not text: return {"error": "No text provided."}
    prompt = f"""Please provide detailed editorial feedback on this writing.

FOCUS AREA: {focus}

TEXT TO REVIEW:
---
{text}
---

Structure your feedback as:
1. What's working well (be specific)
2. What needs attention (be specific and explain why)
3. Concrete suggestions for improvement
4. Suggested next steps for revision"""
    result = call_creative(prompt, system=CREATIVE_SYSTEM)
    session_log.record(f"feedback ({focus})", result[:100])
    return {"result": result, "is_error": is_error(result)}

def handle_worldbuild(data):
    category = data.get("category","").strip()
    topic = data.get("topic","").strip()
    details = data.get("details","").strip()
    if not category or not topic: return {"error": "Need category and topic."}
    (WORLD_BIBLE_DIR / category).mkdir(exist_ok=True)
    context, loaded = auto_context(f"{topic} {category} {details}")
    prompt = f"""Develop this worldbuilding element:

CATEGORY: {category}
TOPIC: {topic}
{"ADDITIONAL DETAILS: " + details if details else ""}

EXISTING CONTEXT:
{context or "(Starting fresh.)"}

Create a rich, detailed entry. Include all relevant details a writer would need.
Format with clear sections using markdown headers."""
    result = call_creative(prompt, system=CREATIVE_SYSTEM)
    if is_error(result): return {"result": result, "is_error": True}
    filename = topic.lower().replace(" ","_").replace("'","") + ".md"
    filepath = f"{category}/{filename}"
    existing = read_world_file(filepath)
    already_exists = not existing.startswith("[File not found")
    if already_exists:
        return {"result": result, "filepath": filepath, "already_exists": True,
                "loaded_files": loaded, "is_error": False}
    write_world_file(filepath, result)
    session_log.record(f"worldbuild: {topic}", f"Created {category} entry", [filepath])
    return {"result": result, "filepath": filepath, "already_exists": False,
            "loaded_files": loaded, "is_error": False}

def handle_worldbuild_save(data):
    filepath = data.get("filepath","")
    content = data.get("content","")
    mode = data.get("mode","overwrite")
    if mode == "new":
        new_name = data.get("new_name","")
        if new_name:
            cat = str(Path(filepath).parent)
            filepath = f"{cat}/{new_name}.md"
    if mode == "merge":
        existing = read_world_file(filepath)
        ts = datetime.now().strftime("%Y-%m-%d")
        content = existing + f"\n\n---\n\n## Expanded — {ts}\n\n{content}\n"
    write_world_file(filepath, content)
    return {"saved": filepath}

def handle_nextsteps(data):
    current_state = data.get("state","").strip()
    if not current_state: return {"error": "Describe where your story is right now."}
    context, loaded = auto_context(current_state + " plot outline current")
    session_ctx = session_log.recent_context()
    prompt = f"""WORLD BIBLE CONTEXT:\n{context or "(No context.)"}\n\n{session_ctx}\n
CURRENT STATE:\n{current_state}\n
Based on the story's current state and the world bible:
1. What are the 3 most promising directions for the next scene/chapter?
2. Which characters are due for development or a POV shift?
3. Are there any planted seeds ready to pay off?
4. What worldbuilding elements haven't been explored yet?

Be specific and reference established lore."""
    result = call_creative(prompt, system=CREATIVE_SYSTEM)
    session_log.record("nextsteps", result[:100], loaded)
    return {"result": result, "loaded_files": loaded, "is_error": is_error(result)}

def handle_index():
    return {"index": list_world_bible()}

def handle_read(data):
    path = data.get("path","")
    if not path: return {"error": "No path."}
    content = read_world_file(path)
    return {"content": content, "title": world_index.file_titles.get(path, path), "path": path}

def handle_save_file(data):
    return {"result": write_world_file(data.get("path",""), data.get("content",""))}

def handle_delete_file(data):
    return {"result": delete_world_file(data.get("path",""))}

def handle_search(data):
    query = data.get("query","").strip()
    if not query: return {"error": "No query."}
    return {"results": world_index.search_content(query), "query": query}

def handle_save_output(data):
    content = data.get("content","")
    name = data.get("name", f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not name.endswith(".md"): name += ".md"
    (OUTPUT_DIR / name).write_text(content, encoding="utf-8")
    return {"saved": str(OUTPUT_DIR / name)}

def handle_set_backend(data):
    b = data.get("backend","").strip()
    if b in ("claude","local"):
        state.creative_backend = b
        return {"backend": b}
    return {"error": "Invalid backend."}

def handle_get_categories():
    defaults = {"characters","locations","history","magic_systems",
                "cultures","languages","plot_outlines","notes"}
    existing = {p.name for p in WORLD_BIBLE_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")}
    return {"categories": sorted(defaults | existing)}

def handle_rebuild_embeddings():
    if not state.embed_model_ready:
        return {"error": "Embedding model not ready yet.", "updated": 0}
    updated = embed_index.rebuild()
    return {"updated": updated, "total": embed_index.count}

# ─── Analysis handlers ────────────────────────────────────────────────────────

def handle_analyze(data):
    filepath = data.get("filepath","").strip()
    if not filepath: return {"error": "No filepath."}
    path = Path(filepath)
    if not path.exists(): path = APP_DIR / filepath
    if not path.exists(): return {"error": f"File not found: {filepath}"}
    text = path.read_text(encoding="utf-8")
    if len(text) > 50000: text = text[:50000] + "\n[...TRUNCATED...]"
    prompt = f"""Analyze this manuscript and produce a structural report.

TEXT:
{text}

Report should include:
- Word count (approximate)
- Number of scenes/sections
- POV character(s) identified
- Dialogue vs. prose ratio (estimate)
- Pacing notes
- Named characters mentioned
- Named locations mentioned
- Worldbuilding terms
- Timeline markers

Format as a clean structured report."""
    result = call_ollama(STRUCTURAL_MODEL, prompt,
                         system="You are a manuscript analysis tool. Be precise.")
    # Add readability stats
    stats = readability_stats(text)
    session_log.record(f"analyze {filepath}", result[:200])
    return {"result": result, "readability": stats, "is_error": is_error(result)}

def handle_consistency():
    categories = defaultdict(list)
    for path in sorted(WORLD_BIBLE_DIR.rglob("*.md")):
        if path.name == "session_log.md": continue
        cat = path.parent.name
        categories[cat].append(f"\n=== {path.relative_to(WORLD_BIBLE_DIR)} ===\n"
                               + path.read_text(encoding="utf-8"))
    if not categories: return {"result": "World bible is empty."}
    issues = []
    for cat, entries in categories.items():
        if len(entries) < 2: continue
        combined = "\n".join(entries)[:50000]
        prompt = f"""Review these {cat} entries for contradictions.

{combined}

Check for conflicting details, timeline issues, name spelling differences,
inconsistent relationships, and rule contradictions.

List each issue with FILES, ISSUE, and SEVERITY (high/medium/low).
If clean, say "No issues in {cat}." """
        r = call_ollama(STRUCTURAL_MODEL, prompt,
                        system="You are a continuity checker. Be meticulous.", temperature=0.2)
        if not is_error(r): issues.append(f"\n── {cat.upper()} ──\n{r}")
    final = "\n".join(issues) if issues else "No contradictions found."
    session_log.record("consistency", final[:200])
    return {"result": final}

def handle_crossref(data):
    """Cross-reference a manuscript against the world bible using spaCy NER."""
    filepath = data.get("filepath","").strip()
    if not filepath: return {"error": "No filepath."}
    path = Path(filepath)
    if not path.exists(): path = APP_DIR / filepath
    if not path.exists(): return {"error": f"File not found: {filepath}"}
    text = path.read_text(encoding="utf-8")

    # Extract entities
    nlp = get_spacy()
    entities = {"characters": [], "locations": [], "other": []}

    if nlp:
        doc = nlp(text[:100000])  # spaCy limit
        seen = set()
        for ent in doc.ents:
            key = ent.text.lower().strip()
            if len(key) < 2 or key in seen: continue
            seen.add(key)
            if ent.label_ in ("PERSON",):
                entities["characters"].append(ent.text)
            elif ent.label_ in ("GPE","LOC","FAC"):
                entities["locations"].append(ent.text)
            elif ent.label_ in ("ORG","EVENT","WORK_OF_ART","NORP"):
                entities["other"].append(ent.text)
        method = "spaCy NER"
    else:
        # Fallback: ask structural model
        extract = call_ollama(STRUCTURAL_MODEL,
            f"Extract named entities from this text. Return JSON with keys: characters, locations, other.\n\nTEXT:\n{text[:15000]}\n\nReturn ONLY JSON.",
            temperature=0.1)
        try:
            s, e = extract.index("{"), extract.rindex("}") + 1
            entities = json.loads(extract[s:e])
        except Exception:
            entities = {"characters": [], "locations": [], "other": []}
        method = "LLM extraction"

    # Look up each entity in world bible
    all_terms = entities.get("characters",[]) + entities.get("locations",[]) + entities.get("other",[])
    found_context, missing = [], []
    for term in all_terms[:30]:  # cap to avoid huge lookups
        matches = world_index.find(term, max_results=1)
        if matches and matches[0][1] >= 0.5:
            f = matches[0][0]
            content = read_world_file(f)[:1500]
            found_context.append(f"--- {f} (matched: '{term}') ---\n{content}\n")
        else:
            missing.append(term)

    if not found_context:
        return {"result": f"No world bible entries found for entities in {filepath}.\n\nEntities detected ({method}): {', '.join(all_terms)}\n\nMissing from world bible: {', '.join(missing)}",
                "entities": entities, "missing": missing, "method": method}

    check = call_ollama(STRUCTURAL_MODEL,
        f"""Compare this manuscript against world bible entries. Flag contradictions.

MANUSCRIPT ({filepath}):
{text[:12000]}

WORLD BIBLE:
{"".join(found_context[:8])}

Missing from world bible: {', '.join(missing) or 'None'}

Check for description mismatches, timeline conflicts, name spelling differences.
Format as a clear report.""",
        system="You are a continuity checker. Be meticulous.", temperature=0.2)
    session_log.record(f"crossref {filepath}", check[:200])
    return {"result": check, "entities": entities, "missing": missing, "method": method}

def handle_proofread(data):
    """Grammar and style checking via LanguageTool."""
    text = data.get("text","").strip()
    if not text: return {"error": "No text provided."}
    tool = get_languagetool()
    if not tool:
        return {"error": "LanguageTool not available. It requires Java installed on your system. Install Java, then restart the app.",
                "install_hint": "Install Java from https://adoptium.net then restart."}
    try:
        matches = tool.check(text)
        issues = []
        for m in matches[:50]:  # cap results
            issues.append({
                "message": m.message,
                "context": m.context,
                "offset": m.offset,
                "length": m.errorLength,
                "replacements": m.replacements[:3] if m.replacements else [],
                "rule": m.ruleId,
                "category": m.category,
            })
        return {"issues": issues, "count": len(matches),
                "summary": f"Found {len(matches)} issue(s) in {len(text.split())} words."}
    except Exception as e:
        return {"error": f"LanguageTool error: {e}"}

def handle_readability(data):
    """Standalone readability analysis."""
    text = data.get("text","").strip()
    if not text: return {"error": "No text provided."}
    stats = readability_stats(text)
    if not stats: return {"error": "textstat not available."}
    return {"stats": stats}


# ═══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

API_ROUTES = {
    "status": (handle_status, False),
    "write": (handle_write, True),
    "dialogue": (handle_dialogue, True),
    "feedback": (handle_feedback, True),
    "worldbuild": (handle_worldbuild, True),
    "worldbuild_save": (handle_worldbuild_save, True),
    "nextsteps": (handle_nextsteps, True),
    "index": (handle_index, False),
    "read": (handle_read, True),
    "save_file": (handle_save_file, True),
    "delete_file": (handle_delete_file, True),
    "search": (handle_search, True),
    "save_output": (handle_save_output, True),
    "analyze": (handle_analyze, True),
    "consistency": (handle_consistency, False),
    "crossref": (handle_crossref, True),
    "proofread": (handle_proofread, True),
    "readability": (handle_readability, True),
    "set_backend": (handle_set_backend, True),
    "categories": (handle_get_categories, False),
    "rebuild_embeddings": (handle_rebuild_embeddings, False),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  WEB SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",len(body)); self.end_headers(); self.wfile.write(body)
    def _html(self, html):
        body = html.encode()
        self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length",len(body)); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/",""):
            self._html(get_frontend_html())
        elif parsed.path.startswith("/api/"):
            route = parsed.path[5:]
            if route in API_ROUTES:
                handler, needs = API_ROUTES[route]
                if not needs: self._json(handler())
                else:
                    params = parse_qs(parsed.query)
                    self._json(handler({k:v[0] for k,v in params.items()}))
            else: self._json({"error":"Unknown route"},404)
        else: self._json({"error":"Not found"},404)
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            route = parsed.path[5:]
            if route in API_ROUTES:
                handler, needs = API_ROUTES[route]
                if needs:
                    length = int(self.headers.get("Content-Length",0))
                    body = self.rfile.read(length)
                    try: data = json.loads(body) if body else {}
                    except: data = {}
                    self._json(handler(data))
                else: self._json(handler())
            else: self._json({"error":"Unknown route"},404)
        else: self._json({"error":"Not found"},404)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("",0)); return s.getsockname()[1]


# ═══════════════════════════════════════════════════════════════════════════════
#  FRONTEND HTML
# ═══════════════════════════════════════════════════════════════════════════════

def get_frontend_html():
    return '''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Creative Writing Workshop</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');
:root{--bg:#1a1914;--bg2:#22211b;--bg3:#2a2923;--bgi:#1e1d18;--bdr:#3a382f;--bdrf:#8b7a5e;--tx:#d4cfc4;--txd:#8a8478;--txb:#f0ebe0;--ac:#c9a96e;--acd:#9a7d4e;--acg:rgba(201,169,110,0.15);--err:#c07070;--ok:#70a070;--tag:#3d3a30;--r:6px}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--tx);min-height:100vh;display:flex}
.sb{width:260px;min-width:260px;background:var(--bg2);border-right:1px solid var(--bdr);display:flex;flex-direction:column;height:100vh;position:sticky;top:0}
.sb-h{padding:24px 20px 16px;border-bottom:1px solid var(--bdr)}
.sb-h h1{font-family:'Crimson Pro',serif;font-size:20px;font-weight:600;color:var(--ac)}
.sb-h .sub{font-size:11px;color:var(--txd);margin-top:4px}
.sb-n{flex:1;overflow-y:auto;padding:12px 0}
.ns{padding:0 12px;margin-bottom:8px}
.ns-t{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--txd);padding:8px 8px 6px}
.nb{display:flex;align-items:center;gap:10px;width:100%;padding:9px 12px;background:none;border:none;border-radius:var(--r);color:var(--tx);font-family:inherit;font-size:13px;cursor:pointer;text-align:left}
.nb:hover{background:var(--bg3)}.nb.active{background:var(--acg);color:var(--ac)}
.nb .i{font-size:16px;width:20px;text-align:center}
.sb-f{padding:12px 16px;border-top:1px solid var(--bdr)}
.bt{display:flex;background:var(--bgi);border-radius:var(--r);overflow:hidden;border:1px solid var(--bdr)}
.bo{flex:1;padding:7px 0;text-align:center;font-size:11px;font-weight:500;background:none;border:none;color:var(--txd);cursor:pointer;font-family:inherit}
.bo.active{background:var(--ac);color:var(--bg)}
.sl{font-size:10px;color:var(--txd);margin-top:8px;text-align:center}
.main{flex:1;display:flex;flex-direction:column;height:100vh;overflow:hidden}
.vc{flex:1;overflow-y:auto;padding:32px 40px;max-width:900px;width:100%;margin:0 auto}
.vw{display:none}.vw.active{display:block}
h2{font-family:'Crimson Pro',serif;font-size:28px;font-weight:300;color:var(--txb);margin-bottom:8px}
h2+p.d{color:var(--txd);font-size:14px;margin-bottom:28px;line-height:1.5}
.f{margin-bottom:20px}
.f label{display:block;font-size:12px;font-weight:500;color:var(--txd);margin-bottom:6px;text-transform:uppercase;letter-spacing:.8px}
.f input,.f textarea,.f select{width:100%;padding:10px 14px;background:var(--bgi);border:1px solid var(--bdr);border-radius:var(--r);color:var(--tx);font-family:inherit;font-size:14px}
.f input:focus,.f textarea:focus,.f select:focus{outline:none;border-color:var(--bdrf)}
.f textarea{min-height:120px;resize:vertical;line-height:1.6}
.f select{cursor:pointer;appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%238a8478'%3E%3Cpath d='M6 8L1 3h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:32px}
.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;border:1px solid var(--bdr);border-radius:var(--r);background:var(--bg3);color:var(--tx);font-family:inherit;font-size:13px;font-weight:500;cursor:pointer}
.btn:hover{border-color:var(--bdrf)}.btn.p{background:var(--ac);color:var(--bg);border-color:var(--ac)}.btn.p:hover{background:var(--acd)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.br{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap}
.oa{margin-top:28px;padding:24px;background:var(--bg3);border:1px solid var(--bdr);border-radius:var(--r);display:none}
.oa.vis{display:block}
.lf{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.lf .tg{font-size:11px;padding:3px 8px;background:var(--tag);border-radius:3px;color:var(--ac);font-family:'JetBrains Mono',monospace}
.oc{font-family:'Crimson Pro',serif;font-size:17px;line-height:1.75;white-space:pre-wrap;word-wrap:break-word}
.oact{margin-top:16px;padding-top:14px;border-top:1px solid var(--bdr);display:flex;gap:10px}
.sp{display:none;align-items:center;gap:12px;margin-top:20px;color:var(--txd);font-size:14px}
.sp.vis{display:flex}
.sp::before{content:'';width:18px;height:18px;border:2px solid var(--bdr);border-top-color:var(--ac);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.stb{padding:12px 20px;font-size:13px;display:flex;align-items:center;gap:10px}
.stb.w{background:#3a2f1a;color:#ddb050}.stb.e{background:#3a1a1a;color:var(--err)}.stb.ok{background:#1a2a1a;color:var(--ok)}
.cp{margin-top:16px;padding:16px;background:#2a2518;border:1px solid var(--acd);border-radius:var(--r)}
.cp p{margin-bottom:10px;font-size:14px}
.mo{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;align-items:center;justify-content:center}
.mo.vis{display:flex}
.md{background:var(--bg2);border:1px solid var(--bdr);border-radius:8px;width:90%;max-width:700px;max-height:85vh;display:flex;flex-direction:column}
.md-h{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--bdr)}
.md-h h3{font-family:'JetBrains Mono',monospace;font-size:14px;color:var(--ac);font-weight:400}
.md-h button{background:none;border:none;color:var(--txd);font-size:20px;cursor:pointer}
.md-b{flex:1;overflow:auto;padding:16px 20px}
.md-b textarea{width:100%;min-height:400px;padding:14px;background:var(--bgi);border:1px solid var(--bdr);border-radius:var(--r);color:var(--tx);font-family:'JetBrains Mono',monospace;font-size:13px;line-height:1.6;resize:vertical}
.md-b textarea:focus{outline:none;border-color:var(--bdrf)}
.md-f{padding:12px 20px;border-top:1px solid var(--bdr);display:flex;gap:10px;justify-content:flex-end}
.sr{margin-bottom:16px;padding:12px 16px;background:var(--bg3);border-radius:var(--r);border:1px solid var(--bdr)}
.sr h4{font-size:14px;color:var(--ac);margin-bottom:6px;cursor:pointer}
.sr h4:hover{text-decoration:underline}
.sr .ml{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--txd);padding:2px 0}
.fc{margin-bottom:6px}
.fc summary{font-size:13px;font-weight:600;color:var(--ac);cursor:pointer;padding:6px 0;list-style:none}
.fc summary::before{content:'▸ ';font-size:11px}
.fc[open] summary::before{content:'▾ '}
.fe{display:flex;align-items:center;justify-content:space-between;padding:5px 8px 5px 20px;border-radius:3px;cursor:pointer;font-size:13px}
.fe:hover{background:var(--bgi)}
.fe .sz{font-size:11px;color:var(--txd)}
.rd{margin-top:16px;padding:16px;background:var(--bgi);border:1px solid var(--bdr);border-radius:var(--r)}
.rd h4{font-size:13px;color:var(--ac);margin-bottom:8px}
.rd .rg{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.rd .ri{font-size:13px}.rd .ri span{color:var(--txd);font-size:11px;display:block}
.pi{margin-bottom:12px;padding:12px;background:var(--bg3);border-left:3px solid var(--ac);border-radius:0 var(--r) var(--r) 0}
.pi .pm{font-size:14px;margin-bottom:4px}.pi .pc{font-size:12px;color:var(--txd)}
.pi .pr{font-size:12px;color:var(--ok);margin-top:4px}
@media(max-width:768px){.sb{width:60px;min-width:60px}.sb-h h1,.sb-h .sub,.ns-t,.nb span,.sb-f .sl{display:none}.nb{justify-content:center}.vc{padding:20px 16px}}
</style></head><body>
<div class="sb">
<div class="sb-h"><h1>Writing Workshop</h1><div class="sub" id="stats"></div></div>
<nav class="sb-n">
<div class="ns"><div class="ns-t">Create</div>
<button class="nb active" onclick="sv('write')" data-v="write"><span class="i">✦</span><span>Write</span></button>
<button class="nb" onclick="sv('dialogue')" data-v="dialogue"><span class="i">💬</span><span>Dialogue</span></button>
<button class="nb" onclick="sv('worldbuild')" data-v="worldbuild"><span class="i">🌍</span><span>Worldbuild</span></button>
<button class="nb" onclick="sv('feedback')" data-v="feedback"><span class="i">✎</span><span>Feedback</span></button>
<button class="nb" onclick="sv('nextsteps')" data-v="nextsteps"><span class="i">→</span><span>Next Steps</span></button>
</div>
<div class="ns"><div class="ns-t">World Bible</div>
<button class="nb" onclick="sv('files')" data-v="files"><span class="i">📁</span><span>Browse</span></button>
<button class="nb" onclick="sv('search')" data-v="search"><span class="i">🔍</span><span>Search</span></button>
</div>
<div class="ns"><div class="ns-t">Analyze</div>
<button class="nb" onclick="sv('analyze')" data-v="analyze"><span class="i">📊</span><span>Manuscript</span></button>
<button class="nb" onclick="sv('crossref')" data-v="crossref"><span class="i">⚖</span><span>Cross-Ref</span></button>
<button class="nb" onclick="sv('consistency')" data-v="consistency"><span class="i">🔗</span><span>Consistency</span></button>
<button class="nb" onclick="sv('proofread')" data-v="proofread"><span class="i">📝</span><span>Proofread</span></button>
</div>
</nav>
<div class="sb-f">
<div class="bt"><button class="bo" id="bL" onclick="setB('local')">Local (Free)</button><button class="bo" id="bC" onclick="setB('claude')">Claude API</button></div>
<div class="sl" id="ts"></div>
</div></div>
<div class="main">
<div id="banner" class="stb" style="display:none"></div>
<div class="vc">

<!-- WRITE -->
<div class="vw active" id="v-write"><h2>Write</h2><p class="d">Describe what you want. World bible context loads automatically.</p>
<div class="f"><label>Prompt</label><textarea id="wP" rows="5" placeholder="Write a tense scene where..."></textarea></div>
<div class="br"><button class="btn p" onclick="doWrite()">Generate</button></div>
<div class="sp" id="wS">Thinking...</div>
<div class="oa" id="wO"><div class="lf" id="wL"></div><div class="oc" id="wC"></div>
<div class="oact"><button class="btn" onclick="save('wC')">💾 Save</button><button class="btn" onclick="copy('wC')">📋 Copy</button></div></div></div>

<!-- DIALOGUE -->
<div class="vw" id="v-dialogue"><h2>Dialogue</h2><p class="d">Build a dialogue scene with auto-loaded character context.</p>
<div class="f"><label>Characters</label><input id="dCh" placeholder="Sera Voss, The Harbormaster"></div>
<div class="f"><label>Situation</label><input id="dSi" placeholder="Sera needs passage through a blockaded port"></div>
<div class="f"><label>Tone (optional)</label><input id="dTo" placeholder="Tense, undercurrent of old respect"></div>
<div class="br"><button class="btn p" onclick="doDialogue()">Write Scene</button></div>
<div class="sp" id="dS">Writing...</div>
<div class="oa" id="dO"><div class="lf" id="dL"></div><div class="oc" id="dC"></div>
<div class="oact"><button class="btn" onclick="save('dC')">💾 Save</button><button class="btn" onclick="copy('dC')">📋 Copy</button></div></div></div>

<!-- WORLDBUILD -->
<div class="vw" id="v-worldbuild"><h2>Worldbuild</h2><p class="d">Create and file a new world bible entry.</p>
<div class="f"><label>Category</label><select id="wbCat"></select></div>
<div class="f"><label>Topic</label><input id="wbTop" placeholder="Sera Voss"></div>
<div class="f"><label>Details (optional)</label><textarea id="wbDet" rows="3" placeholder="A disgraced naval commander..."></textarea></div>
<div class="br"><button class="btn p" onclick="doWB()">Create</button></div>
<div class="sp" id="wbS">Building lore...</div>
<div class="oa" id="wbO"><div class="lf" id="wbL"></div><div class="oc" id="wbC"></div>
<div id="wbConf" class="cp" style="display:none"><p>⚠ File exists. How to handle?</p>
<div class="br"><button class="btn" onclick="wbR('overwrite')">Overwrite</button><button class="btn" onclick="wbR('merge')">Merge</button><button class="btn" onclick="wbR('new')">New Name</button></div></div>
<div class="oact"><span id="wbSv" style="font-size:13px;color:var(--ok)"></span></div></div></div>

<!-- FEEDBACK -->
<div class="vw" id="v-feedback"><h2>Feedback</h2><p class="d">Paste writing for editorial critique.</p>
<div class="f"><label>Focus</label><select id="fbF"><option>general</option><option>dialogue</option><option>pacing</option><option>prose-style</option><option>character-voice</option><option>tension</option><option>worldbuilding</option></select></div>
<div class="f"><label>Text</label><textarea id="fbT" rows="10" placeholder="Paste your text..."></textarea></div>
<div class="br"><button class="btn p" onclick="doFB()">Get Feedback</button></div>
<div class="sp" id="fbS">Reading...</div>
<div class="oa" id="fbO"><div class="oc" id="fbC"></div></div></div>

<!-- NEXTSTEPS -->
<div class="vw" id="v-nextsteps"><h2>Next Steps</h2><p class="d">Describe your story state. Get direction suggestions.</p>
<div class="f"><label>Current state</label><textarea id="nsT" rows="5" placeholder="Sera just escaped the harbor..."></textarea></div>
<div class="br"><button class="btn p" onclick="doNS()">Suggest</button></div>
<div class="sp" id="nsS">Analyzing...</div>
<div class="oa" id="nsO"><div class="lf" id="nsL"></div><div class="oc" id="nsC"></div></div></div>

<!-- FILES -->
<div class="vw" id="v-files"><h2>World Bible</h2><p class="d">Browse and manage your files.</p>
<div class="br" style="margin-bottom:16px"><button class="btn" onclick="rIdx()">↻ Refresh</button><button class="btn" onclick="newFile()">+ New</button><button class="btn" onclick="rebuildEmb()">🔄 Rebuild Embeddings</button></div>
<div id="fTree"></div></div>

<!-- SEARCH -->
<div class="vw" id="v-search"><h2>Search</h2><p class="d">Search across all world bible files.</p>
<div class="f"><input id="sQ" placeholder="Search..." onkeydown="if(event.key==='Enter')doSearch()"></div>
<div class="br"><button class="btn p" onclick="doSearch()">Search</button></div>
<div id="sR" style="margin-top:20px"></div></div>

<!-- ANALYZE -->
<div class="vw" id="v-analyze"><h2>Manuscript Analysis</h2><p class="d">Structural analysis + readability metrics (local model).</p>
<div class="f"><label>Filepath</label><input id="anF" placeholder="manuscripts/chapter1.md"></div>
<div class="br"><button class="btn p" onclick="doAnalyze()">Analyze</button></div>
<div class="sp" id="anS">Analyzing...</div>
<div class="oa" id="anO"><div class="oc" id="anC"></div><div id="anR"></div></div></div>

<!-- CROSSREF -->
<div class="vw" id="v-crossref"><h2>Cross-Reference</h2><p class="d">Check manuscript against world bible for contradictions. Uses spaCy NER + local model.</p>
<div class="f"><label>Filepath</label><input id="crF" placeholder="manuscripts/chapter1.md"></div>
<div class="br"><button class="btn p" onclick="doCR()">Cross-Reference</button></div>
<div class="sp" id="crS">Extracting entities...</div>
<div class="oa" id="crO"><div id="crE"></div><div class="oc" id="crC"></div></div></div>

<!-- CONSISTENCY -->
<div class="vw" id="v-consistency"><h2>Consistency Check</h2><p class="d">Scan world bible for contradictions (local model).</p>
<div class="br"><button class="btn p" onclick="doCons()">Run Check</button></div>
<div class="sp" id="coS">Checking...</div>
<div class="oa" id="coO"><div class="oc" id="coC"></div></div></div>

<!-- PROOFREAD -->
<div class="vw" id="v-proofread"><h2>Proofread</h2><p class="d">Grammar and style checking via LanguageTool (rule-based, not AI). Catches mechanical errors LLMs miss.</p>
<div class="f"><label>Text</label><textarea id="prT" rows="10" placeholder="Paste text to proofread..."></textarea></div>
<div class="br"><button class="btn p" onclick="doPR()">Proofread</button></div>
<div class="sp" id="prS">Checking grammar...</div>
<div class="oa" id="prO"><div id="prSum" style="margin-bottom:16px;font-size:14px;color:var(--ac)"></div><div id="prI"></div></div></div>

</div></div>

<!-- EDITOR MODAL -->
<div class="mo" id="edM"><div class="md"><div class="md-h"><h3 id="edT">file.md</h3><button onclick="clEd()">×</button></div>
<div class="md-b"><textarea id="edC"></textarea></div>
<div class="md-f"><button class="btn" onclick="clEd()">Cancel</button><button class="btn" onclick="delEd()">🗑 Delete</button><button class="btn p" onclick="svEd()">Save</button></div></div></div>

<!-- SAVE MODAL -->
<div class="mo" id="svM"><div class="md" style="max-width:400px"><div class="md-h"><h3>Save to File</h3><button onclick="clSv()">×</button></div>
<div class="md-b"><div class="f"><label>Filename</label><input id="svN" placeholder="my_scene"></div></div>
<div class="md-f"><button class="btn" onclick="clSv()">Cancel</button><button class="btn p" onclick="cfSv()">Save</button></div></div></div>

<script>
let curView='write',edPath='',svCId='',wbRes=null,wbFp='';

async function api(r,d=null){
const o=d?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}:{};
return(await fetch('/api/'+r,o)).json()}

function sv(n){document.querySelectorAll('.vw').forEach(v=>v.classList.remove('active'));
document.querySelectorAll('.nb').forEach(b=>b.classList.remove('active'));
const el=document.getElementById('v-'+n);if(el)el.classList.add('active');
const btn=document.querySelector('[data-v="'+n+'"]');if(btn)btn.classList.add('active');
curView=n;if(n==='files')rIdx();if(n==='worldbuild')ldCat()}

async function setB(b){await api('set_backend',{backend:b});uB(b)}
function uB(b){document.getElementById('bL').classList.toggle('active',b==='local');
document.getElementById('bC').classList.toggle('active',b==='claude')}

async function poll(){try{const s=await api('status');
document.getElementById('stats').textContent=s.files_indexed+' files · '+s.embeddings_indexed+' embeddings';
uB(s.backend);
if(s.token_usage.api_calls>0)document.getElementById('ts').textContent=s.token_usage.api_calls+' calls · '+s.token_usage.cost;
const bn=document.getElementById('banner');
if(s.models_pulling){bn.style.display='flex';bn.className='stb w';bn.textContent='↓ '+s.pull_progress}
else if(s.startup_errors.length>0){bn.style.display='flex';bn.className='stb e';bn.textContent=s.startup_errors[0]}
else{bn.style.display='none'}}catch(e){}}
setInterval(poll,3000);poll();

function ss(id){document.getElementById(id).classList.add('vis')}
function hs(id){document.getElementById(id).classList.remove('vis')}
function so(id,data,cid,lid){const a=document.getElementById(id);a.classList.add('vis');
document.getElementById(cid).textContent=data.result||'';
if(lid&&data.loaded_files&&data.loaded_files.length)
document.getElementById(lid).innerHTML=data.loaded_files.map(f=>'<span class="tg">📂 '+f+'</span>').join('');
else if(lid)document.getElementById(lid).innerHTML=''}

async function doWrite(){const p=document.getElementById('wP').value.trim();if(!p)return;
ss('wS');document.getElementById('wO').classList.remove('vis');
const d=await api('write',{prompt:p});hs('wS');so('wO',d,'wC','wL')}

async function doDialogue(){const ch=document.getElementById('dCh').value.trim(),
si=document.getElementById('dSi').value.trim(),to=document.getElementById('dTo').value.trim();
if(!ch||!si)return;ss('dS');document.getElementById('dO').classList.remove('vis');
const d=await api('dialogue',{characters:ch,situation:si,tone:to});hs('dS');so('dO',d,'dC','dL')}

async function ldCat(){const d=await api('categories');
document.getElementById('wbCat').innerHTML=d.categories.map(c=>'<option value="'+c+'">'+c.replace(/_/g,' ')+'</option>').join('')}

async function doWB(){const cat=document.getElementById('wbCat').value,
top=document.getElementById('wbTop').value.trim(),det=document.getElementById('wbDet').value.trim();
if(!top)return;ss('wbS');document.getElementById('wbO').classList.remove('vis');
document.getElementById('wbConf').style.display='none';
const d=await api('worldbuild',{category:cat,topic:top,details:det});hs('wbS');so('wbO',d,'wbC','wbL');
if(d.already_exists){document.getElementById('wbConf').style.display='block';wbRes=d.result;wbFp=d.filepath;
document.getElementById('wbSv').textContent=''}
else if(d.filepath)document.getElementById('wbSv').textContent='✓ Saved to world_bible/'+d.filepath}

async function wbR(m){let nn='';if(m==='new'){nn=prompt('New filename:');if(!nn)return}
await api('worldbuild_save',{filepath:wbFp,content:wbRes,mode:m,new_name:nn});
document.getElementById('wbConf').style.display='none';document.getElementById('wbSv').textContent='✓ Saved';poll()}

async function doFB(){const t=document.getElementById('fbT').value.trim(),f=document.getElementById('fbF').value;
if(!t)return;ss('fbS');document.getElementById('fbO').classList.remove('vis');
const d=await api('feedback',{text:t,focus:f});hs('fbS');so('fbO',d,'fbC')}

async function doNS(){const st=document.getElementById('nsT').value.trim();if(!st)return;
ss('nsS');document.getElementById('nsO').classList.remove('vis');
const d=await api('nextsteps',{state:st});hs('nsS');so('nsO',d,'nsC','nsL')}

async function rIdx(){const d=await api('index');const t=document.getElementById('fTree');
if(!d.index||!Object.keys(d.index).length){t.innerHTML='<p style="color:var(--txd)">Empty. Use Worldbuild to create entries.</p>';return}
let h='';for(const[c,fs]of Object.entries(d.index)){
h+='<details class="fc" open><summary>'+c.replace(/_/g,' ')+' ('+fs.length+')</summary>';
for(const f of fs)h+='<div class="fe" onclick="opF(\\''+f.file+'\\')"><span>'+f.name+'</span><span class="sz">'+(f.size/1024).toFixed(1)+' KB</span></div>';
h+='</details>'}t.innerHTML=h;poll()}

async function opF(p){const d=await api('read',{path:p});edPath=p;
document.getElementById('edT').textContent=p;document.getElementById('edC').value=d.content||'';
document.getElementById('edM').classList.add('vis')}
function clEd(){document.getElementById('edM').classList.remove('vis')}
async function svEd(){await api('save_file',{path:edPath,content:document.getElementById('edC').value});clEd();if(curView==='files')rIdx();poll()}
async function delEd(){if(!confirm('Delete '+edPath+'?'))return;await api('delete_file',{path:edPath});clEd();if(curView==='files')rIdx();poll()}
function newFile(){const p=prompt('Path (e.g. characters/new_char):');if(!p)return;
const fp=p.endsWith('.md')?p:p+'.md';const nm=fp.split('/').pop().replace('.md','').replace(/_/g,' ');
edPath=fp;document.getElementById('edT').textContent=fp;
document.getElementById('edC').value='# '+nm.charAt(0).toUpperCase()+nm.slice(1)+'\\n\\n## Overview\\n\\n';
document.getElementById('edM').classList.add('vis')}
async function rebuildEmb(){const d=await api('rebuild_embeddings');alert('Rebuilt: '+d.total+' embeddings ('+d.updated+' updated)')}

async function doSearch(){const q=document.getElementById('sQ').value.trim();if(!q)return;
const d=await api('search',{query:q});const c=document.getElementById('sR');
if(!d.results||!d.results.length){c.innerHTML='<p style="color:var(--txd)">No matches.</p>';return}
c.innerHTML=d.results.map(r=>'<div class="sr"><h4 onclick="opF(\\''+r.file+'\\')">'+r.title+' <span style="color:var(--txd);font-weight:400;font-size:12px">'+r.file+'</span></h4>'+r.matches.map(m=>'<div class="ml">'+m.line+': '+m.text+'</div>').join('')+'</div>').join('')}

async function doAnalyze(){const f=document.getElementById('anF').value.trim();if(!f)return;
ss('anS');document.getElementById('anO').classList.remove('vis');
const d=await api('analyze',{filepath:f});hs('anS');so('anO',d,'anC');
const rd=document.getElementById('anR');
if(d.readability){const r=d.readability;let grid='<div class="ri">'+r.word_count.toLocaleString()+'<span>Words</span></div><div class="ri">'+r.sentence_count+'<span>Sentences</span></div><div class="ri">'+r.avg_sentence_length+'<span>Avg words/sentence</span></div>';
if(r.flesch_reading_ease!=null)grid+='<div class="ri">'+r.flesch_reading_ease+'<span>Flesch Reading Ease</span></div>';
if(r.flesch_kincaid_grade!=null)grid+='<div class="ri">'+r.flesch_kincaid_grade+'<span>Flesch-Kincaid Grade</span></div>';
if(r.gunning_fog!=null)grid+='<div class="ri">'+r.gunning_fog+'<span>Gunning Fog Index</span></div>';
grid+='<div class="ri">'+r.reading_time_min+' min<span>Reading Time</span></div>';
rd.innerHTML='<div class="rd"><h4>Readability Metrics</h4><div class="rg">'+grid+'</div></div>'}
else{rd.innerHTML=''}}

async function doCR(){const f=document.getElementById('crF').value.trim();if(!f)return;
ss('crS');document.getElementById('crO').classList.remove('vis');
const d=await api('crossref',{filepath:f});hs('crS');
document.getElementById('crO').classList.add('vis');document.getElementById('crC').textContent=d.result||'';
const e=document.getElementById('crE');let h='<div style="margin-bottom:16px">';
if(d.method)h+='<span class="tg">Method: '+d.method+'</span> ';
if(d.entities){const ent=d.entities;if(ent.characters&&ent.characters.length)h+='<span class="tg">Characters: '+ent.characters.join(', ')+'</span> ';
if(ent.locations&&ent.locations.length)h+='<span class="tg">Locations: '+ent.locations.join(', ')+'</span> '}
if(d.missing&&d.missing.length)h+='<div style="margin-top:8px;font-size:12px;color:var(--txd)">Missing from world bible: '+d.missing.join(', ')+'</div>';
e.innerHTML=h+'</div>'}

async function doCons(){ss('coS');document.getElementById('coO').classList.remove('vis');
const d=await api('consistency');hs('coS');so('coO',d,'coC')}

async function doPR(){const t=document.getElementById('prT').value.trim();if(!t)return;
ss('prS');document.getElementById('prO').classList.remove('vis');
const d=await api('proofread',{text:t});hs('prS');document.getElementById('prO').classList.add('vis');
if(d.error){document.getElementById('prSum').textContent=d.error;document.getElementById('prI').innerHTML='';return}
document.getElementById('prSum').textContent=d.summary||'';
const c=document.getElementById('prI');
if(!d.issues||!d.issues.length){c.innerHTML='<p style="color:var(--ok)">No issues found!</p>';return}
c.innerHTML=d.issues.map(i=>'<div class="pi"><div class="pm">'+i.message+'</div>'+(i.context?'<div class="pc">...'+i.context+'...</div>':'')+(i.replacements&&i.replacements.length?'<div class="pr">Suggestion: '+i.replacements.join(' / ')+'</div>':'')+'</div>').join('')}

function save(cid){svCId=cid;document.getElementById('svN').value='';document.getElementById('svM').classList.add('vis')}
function clSv(){document.getElementById('svM').classList.remove('vis')}
async function cfSv(){const n=document.getElementById('svN').value.trim()||'output';
await api('save_output',{content:document.getElementById(svCId).textContent,name:n});clSv();alert('Saved to output/'+n+'.md')}
function copy(cid){navigator.clipboard.writeText(document.getElementById(cid).textContent)}

ldCat()
</script></body></html>'''


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         Creative Writing Workshop v{APP_VERSION}                      ║
║         Mistral 24B + Qwen 4B + Semantic Search              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    ensure_ollama_and_model()
    print(f"\n  World bible: {world_index.file_count} files keyword-indexed")

    # Build embeddings in background after model is ready
    def build_embeddings_when_ready():
        for _ in range(120):  # wait up to 2 min
            if state.embed_model_ready:
                n = embed_index.rebuild()
                if n: print(f"  ✓ Embedded {n} files for semantic search")
                return
            time.sleep(1)
    threading.Thread(target=build_embeddings_when_ready, daemon=True).start()

    port = find_free_port()
    server = HTTPServer((WEB_HOST, port), RequestHandler)
    url = f"http://{WEB_HOST}:{port}"

    print(f"\n  ✓ Server at {url}")
    print(f"  Opening browser...\n  Press Ctrl+C to stop.\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    def shutdown(sig, frame):
        print("\n  Shutting down...")
        server.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
