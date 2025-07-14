#!/usr/bin/env python3
"""
server.py: FastAPI MCP server wrapper around code_index functions.
Dependencies:
  pip install fastapi uvicorn watchdog
"""

import threading
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

from code_index import init_db, search_index, reindex_all, watch_mode, TABLE_NAME
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Code Index MCP")
con = init_db()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class IndexRequest(BaseModel):
    repo: str
    max_mb: float = 1.0

class SearchResponse(BaseModel):
    path: str
    snippet: str
    distance: float

@app.post("/index")
def http_index(req: IndexRequest):
    reindex_all(Path(req.repo), int(req.max_mb * 1024**2), con, embedder)
    count = con.execute(f"SELECT count(*) FROM {TABLE_NAME}").fetchone()[0]
    return {"status": "indexed", "files": count}

@app.get("/search", response_model=list[SearchResponse])
def http_search(query: str, k: int = 5):
    results = search_index(con, embedder, query, k)
    return [{"path": p, "snippet": s, "distance": d} for p,s,d in results]

@app.on_event("startup")
def _startup_watch():
    t = threading.Thread(target=lambda: watch_mode(".", 1.0, 5.0), daemon=True)
    t.start()
