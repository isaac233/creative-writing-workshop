#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Creative Writing Workshop v2.0 — Setup
# ═══════════════════════════════════════════════════════════════
#  Optional: run this to pre-download everything before first use.
#  The app handles setup automatically, but this lets you do it
#  ahead of time (useful for slow connections).
# ═══════════════════════════════════════════════════════════════

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     Creative Writing Workshop v2.0 — Setup                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Python ───────────────────────────────────────────────────
echo "▸ Checking Python..."
if command -v python3 &>/dev/null; then
    echo "  ✓ $(python3 --version)"
else
    echo "  ✗ Python 3 not found. Install it first."
    exit 1
fi

# ── Python packages ──────────────────────────────────────────
echo ""
echo "▸ Installing Python packages..."
pip3 install requests textstat --quiet --break-system-packages 2>/dev/null || \
pip3 install requests textstat --quiet
echo "  ✓ Core packages installed"

# Optional packages
echo "  Installing optional packages (spaCy, LanguageTool)..."
pip3 install spacy language_tool_python --quiet --break-system-packages 2>/dev/null || \
pip3 install spacy language_tool_python --quiet 2>/dev/null || true
python3 -m spacy download en_core_web_sm --quiet 2>/dev/null || true
echo "  ✓ Optional packages installed (LanguageTool requires Java)"

# ── Ollama ───────────────────────────────────────────────────
echo ""
echo "▸ Checking Ollama..."
if command -v ollama &>/dev/null; then
    echo "  ✓ Ollama is installed"
else
    echo "  ✗ Ollama not found."
    echo ""
    echo "  Install Ollama:"
    echo "    macOS/Linux:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "    Windows:      Download from https://ollama.com/download"
    echo ""
    echo "  After installing, re-run this script."
    exit 1
fi

# ── Pull models ──────────────────────────────────────────────
echo ""
echo "▸ Pulling models (this may take a while on first run)..."
echo ""

echo "  Pulling qwen3.5:4b (structural tasks — ~2.5 GB)..."
ollama pull qwen3.5:4b

echo ""
echo "  Pulling nomic-embed-text (semantic search — ~270 MB)..."
ollama pull nomic-embed-text

echo ""
echo "  Pulling mistral-small3.2 (creative writing — ~14 GB)..."
ollama pull mistral-small3.2

echo ""
echo "  ✓ All models ready"

# ── Directories ──────────────────────────────────────────────
echo ""
echo "▸ Creating project directories..."
mkdir -p world_bible/{characters,locations,history,magic_systems,cultures,languages,plot_outlines,notes}
mkdir -p manuscripts output
echo "  ✓ Directories ready"

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "  Optional: set a Claude API key for highest quality output:"
    echo "    export ANTHROPIC_API_KEY=sk-ant-your-key-here"
    echo ""
fi

echo "  To start:  python3 creative_workshop.py"
echo ""
echo "═══════════════════════════════════════════════════════════════"
