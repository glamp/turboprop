#!/usr/bin/env python3
"""
code_index.py: Scan a Git repo, build a DuckDB-backed code index with embeddings and HNSW (cosine similarity),
with CLI commands: index, search, watch (incremental, debounced).
Dependencies:
  pip install duckdb sentence-transformers hnswlib watchdog
"""

import os, subprocess, argparse, hashlib, time, threading
from pathlib import Path

import duckdb
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
DB_PATH         = "code_index.duckdb"
TABLE_NAME      = "code_files"
INDEX_PATH      = "hnsw_code.idx"
DIMENSIONS      = 384
EMBED_MODEL     = "all-MiniLM-L6-v2"
CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
                   ".h", ".cs", ".go", ".rs", ".swift", ".kt", ".m", ".rb",
                   ".php", ".sh", ".html", ".css", ".json", ".yaml", ".yml",
                   ".xml"}

def compute_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def init_db():
    con = duckdb.connect(DB_PATH)
    con.execute(f"""
      CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id VARCHAR PRIMARY KEY,
        path VARCHAR,
        content TEXT,
        embedding BLOB
      )
    """)
    return con

def scan_repo(repo_path: Path, max_bytes: int):
    try:
        result = subprocess.run(
            ["git", "ls-files", "--exclude-standard", "-oi", "--directory"],
            cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True
        )
        ignored = {str(repo_path / line.strip()) for line in result.stdout.splitlines()}
    except subprocess.CalledProcessError:
        ignored = set()
    files = []
    for root, _, names in os.walk(repo_path):
        for name in names:
            p = Path(root) / name
            if str(p) in ignored: continue
            if p.suffix.lower() not in CODE_EXTENSIONS: continue
            try:
                if p.stat().st_size > max_bytes: continue
            except OSError:
                continue
            files.append(p)
    return files

def embed_and_store(con, embedder, files):
    rows = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        uid  = compute_id(str(path) + text)
        emb  = embedder.encode(text).astype(np.float32)
        blob = emb.tobytes()
        rows.append((uid, str(path), text, blob))
    if rows:
        con.executemany(
            f"INSERT OR REPLACE INTO {TABLE_NAME} VALUES (?, ?, ?, ?)",
            rows
        )

def build_full_index(con):
    rows = con.execute(f"SELECT id, embedding FROM {TABLE_NAME}").fetchall()
    ids, vecs = [], []
    for uid, blob in rows:
        v = np.frombuffer(blob, dtype=np.float32)
        if v.size == DIMENSIONS:
            ids.append(uid)
            vecs.append(v)
    if not vecs:
        return None
    idx = hnswlib.Index(space='cosine', dim=DIMENSIONS)
    idx.init_index(max_elements=len(vecs), ef_construction=200, M=16)
    idx.add_items(np.vstack(vecs), ids)
    idx.save_index(INDEX_PATH)
    return idx

def search_index(con, embedder, query: str, k: int):
    idx = hnswlib.Index(space='cosine', dim=DIMENSIONS)
    idx.load_index(INDEX_PATH)
    q_emb = embedder.encode(query).astype(np.float32)
    labels, distances = idx.knn_query(q_emb, k=k)
    results = []
    for uid, dist in zip(labels[0], distances[0]):
        row = con.execute(
            f"SELECT path, substr(content,1,300) FROM {TABLE_NAME} WHERE id = ?",
            (uid,)
        ).fetchone()
        if row:
            results.append((row[0], row[1], dist))
    return results

def embed_and_store_single(con, embedder, path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    uid  = compute_id(str(path) + text)
    emb  = embedder.encode(text).astype(np.float32)
    blob = emb.tobytes()
    con.execute(
      f"INSERT OR REPLACE INTO {TABLE_NAME} VALUES (?, ?, ?, ?)",
      (uid, str(path), text, blob)
    )
    return uid, emb

class DebouncedHandler(FileSystemEventHandler):
    def __init__(self, repo: Path, max_bytes: int, con, embedder, index, debounce_sec: float):
        self.repo      = repo
        self.max_bytes = max_bytes
        self.con       = con
        self.embedder  = embedder
        self.index     = index
        self.debounce  = debounce_sec
        self.timers    = {}

    def on_any_event(self, event):
        p = Path(event.src_path)
        if event.is_directory or p.suffix.lower() not in CODE_EXTENSIONS:
            return
        if p in self.timers:
            self.timers[p].cancel()
        timer = threading.Timer(self.debounce, self._process, args=(event.event_type, p))
        self.timers[p] = timer
        timer.start()

    def _process(self, ev_type: str, p: Path):
        self.timers.pop(p, None)
        if ev_type in ("created", "modified") and p.exists():
            try:
                if p.stat().st_size <= self.max_bytes:
                    res = embed_and_store_single(self.con, self.embedder, p)
                    if res:
                        uid, emb = res
                        self.index.add_items(emb.reshape(1, -1), [uid])
                        self.index.save_index(INDEX_PATH)
                        print(f"[debounce] updated {p}")
            except OSError:
                pass
        elif ev_type == "deleted":
            print(f"[debounce] {p} deleted → full rebuild")
            self.index = build_full_index(self.con)

def watch_mode(repo_path: str, max_mb: float, debounce_sec: float):
    repo      = Path(repo_path).resolve()
    max_bytes = int(max_mb * 1024**2)
    con       = init_db()
    embedder  = SentenceTransformer(EMBED_MODEL)
    idx       = build_full_index(con)
    handler   = DebouncedHandler(repo, max_bytes, con, embedder, idx, debounce_sec)
    observer  = Observer()
    observer.schedule(handler, str(repo), recursive=True)
    print(f"[watch] watching {repo} (max {max_mb} MB, debounce {debounce_sec}s)")
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def main():
    parser = argparse.ArgumentParser()
    sub    = parser.add_subparsers(dest="cmd", required=True)

    p_i = sub.add_parser("index")
    p_i.add_argument("repo")
    p_i.add_argument("--max-mb", type=float, default=1.0)

    p_s = sub.add_parser("search")
    p_s.add_argument("query")
    p_s.add_argument("--k", type=int, default=5)

    p_w = sub.add_parser("watch")
    p_w.add_argument("repo")
    p_w.add_argument("--max-mb", type=float, default=1.0)
    p_w.add_argument("--debounce-sec", type=float, default=5.0)

    args = parser.parse_args()
    con      = init_db()
    embedder = SentenceTransformer(EMBED_MODEL)

    if args.cmd == "index":
        files = scan_repo(Path(args.repo), int(args.max_mb * 1024**2))
        embed_and_store(con, embedder, files)
        build_full_index(con)
        print("Index built.")
    elif args.cmd == "search":
        results = search_index(con, embedder, args.query, args.k)
        for path, snippet, dist in results:
            print(f"{path} (dist={dist:.4f})\n{snippet}\n{'-'*40}")
    else:
        watch_mode(args.repo, args.max_mb, args.debounce_sec)

if __name__ == "__main__":
    main()
