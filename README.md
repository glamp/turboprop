# Turboprop

A lightning-fast, aviation-inspired code search & indexing MCP server and CLI.


- 🔍 **Cosine-HNSW search** over file embeddings  
- 🐆 **Fast** local search via DuckDB + HNSWLib  
- 🔄 **Watch mode** with **incremental**, **debounced** updates  
- 🚀 **HTTP API** (FastAPI) — run with `uvicorn`/`uvx`  
- 🤖 **Slash-command shortcuts** for Claude Code

---

## 📦 Prerequisites

- **macOS/Linux/Windows** with Python 3.8+  
- Git (repo must have a `.git` folder)  
- (Optional) virtualenv: `python -m venv .venv`

## 🍰 Foolproof Install

```bash
git clone https://github.com/you/your-code-index.git
cd your-code-index
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install duckdb sentence-transformers hnswlib watchdog fastapi uvicorn
```

## ⚙️ CLI Usage

```bash
python code_index.py index /path/to/repo --max-mb 1.0
python code_index.py search "terms" --k 5
python code_index.py watch /path/to/repo --max-mb 1.0 --debounce-sec 5.0
```

## 🚀 HTTP MCP Server

Run with:

```bash
uvicorn server:app --reload  # or uvx server:app --reload
```

## 🧠 Optimized for Claude Code

Add `.claude/code-index.commands.md` for slash commands.

---

That’s it—**fucking easy as pie**. 🍰🚀
