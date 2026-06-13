"""
fcrag/ingest/chunker.py — FCRAG 2.0 Document Chunker
=====================================================
Reads all normalized data sources and produces a unified stream of
FCRAGDocument chunks ready for embedding and Qdrant indexing.

Strategy:
  - 3GPP .txt files  → sentence-aware 125-token sliding window, 25-token overlap
  - JSONL narratives → passed through as-is (already ≤125 tokens each)
  - Heading markers  → extracted as clause_id metadata per chunk

Output schema (FCRAGDocument):
  {
    doc_id      : str   # unique chunk ID
    source_type : str   # "3gpp_spec" | "simu5g_narrative" | "alarm_history"
    source_file : str   # original filename
    clause_id   : str   # nearest section heading (e.g. "TS 38.331 §5.5.4")
    text        : str   # chunk text (≤125 tokens)
    metadata    : dict  # {spec_name, chunk_index, token_count, ...}
  }
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator, Iterator

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data" / "processed"

TXT_DIR    = PROCESSED / "3gpp_text"
SIMU5G_DIR = PROCESSED / "simu5g_docs"
KPI_DIR    = PROCESSED / "kpi_narratives"

CHUNK_OUTPUT = PROCESSED / "chunks"   # final unified chunks go here

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MAX_TOKENS   = 125    # PRD §FR-6.3
OVERLAP_TOKS = 25     # PRD §FR-6.3

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FCRAGDocument:
    doc_id:      str
    source_type: str
    source_file: str
    clause_id:   str
    text:        str
    metadata:    dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Token counter (whitespace approximation — no model load needed here)
# ─────────────────────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    """Approximate token count: words (≈ BPE tokens) - Fast version."""
    return len(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitter
# ─────────────────────────────────────────────────────────────────────────────
_SENT_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'          # period/exclaim/question → Capital
    r'|(?<=\d\.)\s+(?=\d)'             # e.g. "5.5.4 foo" style section refs
    r'|(?<=:)\s*\n'                     # colon + newline
    r'|\n{2,}'                          # paragraph break
)

def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units for sentence-aware chunking."""
    parts = _SENT_BOUNDARY.split(text)
    # Filter empty / whitespace-only
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Heading / clause_id extractor
# ─────────────────────────────────────────────────────────────────────────────
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Maps source filename prefix → spec name used in clause_id
_SPEC_NAME_MAP = {
    "TS_38_331": "TS 38.331",
    "TS_38_300": "TS 38.300",
    "TS_23_501": "TS 23.501",
    "TS_23_502": "TS 23.502",
    "TR_21_916": "TR 21.916",
    "TR_21_918": "TR 21.918",
}

def get_spec_name(filename: str) -> str:
    stem = Path(filename).stem          # e.g. "TS_38_331"
    return _SPEC_NAME_MAP.get(stem, stem.replace("_", " "))


def build_clause_id(spec_name: str, heading_text: str) -> str:
    """
    Build a clause_id like 'TS 38.331 §5.5.4' from heading text.
    Falls back to spec_name if no clause number found.
    """
    # Look for numeric section pattern at start: "5.5.4", "A.3.2", etc.
    m = re.match(r"^([A-Z]?\d[\d.]*)\s+", heading_text.strip())
    if m:
        return f"{spec_name} §{m.group(1)}"
    return spec_name


# ─────────────────────────────────────────────────────────────────────────────
# Core chunker: sliding window over sentence list
# ─────────────────────────────────────────────────────────────────────────────
def sliding_window_chunks(
    sentences: list[str],
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKS,
) -> Generator[str, None, None]:
    """
    Yield text chunks of ≤ max_tokens using a sliding window over sentences.
    When a chunk is full, back up by overlap_tokens worth of sentences
    before starting the next chunk.
    """
    if not sentences:
        return

    current_sents: list[str] = []
    current_tokens = 0

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = count_tokens(sent)

        # Single sentence exceeds max — force-split by words
        if sent_tokens > max_tokens:
            # Yield whatever we have collected so far
            if current_sents:
                yield " ".join(current_sents)
                current_sents = []
                current_tokens = 0
            
            words = sent.split()
            sub_chunk: list[str] = []
            sub_tok = 0
            for w in words:
                if sub_tok + 1 > max_tokens:
                    yield " ".join(sub_chunk)
                    sub_chunk = sub_chunk[-(overlap_tokens // 2):]
                    sub_tok = len(sub_chunk)
                sub_chunk.append(w)
                sub_tok += 1
            if sub_chunk:
                # Keep the remainder as overlap for the next sentences
                current_sents = [" ".join(sub_chunk)]
                current_tokens = count_tokens(current_sents[0])
            i += 1
            continue

        # Adding this sentence would exceed limit → emit current chunk
        if current_tokens + sent_tokens > max_tokens and current_sents:
            yield " ".join(current_sents)

            # Back up: keep sentences whose tokens sum ≤ overlap_tokens
            overlap_sents: list[str] = []
            overlap_tok = 0
            for s in reversed(current_sents):
                t = count_tokens(s)
                if overlap_tok + t > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += t
                
            # PREVENT INFINITE LOOP: If overlap + new sentence is STILL too big,
            # we must clear the overlap to make room for the new sentence.
            while overlap_tok + sent_tokens > max_tokens and overlap_sents:
                t = count_tokens(overlap_sents.pop(0))
                overlap_tok -= t

            current_sents = overlap_sents
            current_tokens = overlap_tok
            # Do NOT increment `i`, let it loop and add the sentence now
        else:
            current_sents.append(sent)
            current_tokens += sent_tokens
            i += 1

    if current_sents:
        yield " ".join(current_sents)


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — Chunk 3GPP .txt files
# ─────────────────────────────────────────────────────────────────────────────
def chunk_3gpp_txt(txt_path: Path) -> Generator[FCRAGDocument, None, None]:
    """
    Parse a 3GPP plain-text file (with ## heading markers from normalizer)
    and yield FCRAGDocument chunks.
    """
    spec_name = get_spec_name(txt_path.stem)
    raw = txt_path.read_text(encoding="utf-8")

    chunk_idx = 0
    current_clause = spec_name          # tracks nearest heading
    current_heading_level = 0

    # Split file into sections by heading markers
    # Each section: optional heading line + body text
    sections: list[tuple[str, str]] = []   # (clause_id, body_text)

    current_body_lines: list[str] = []
    current_clause_id = spec_name

    for line in raw.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save previous section body
            if current_body_lines:
                sections.append((current_clause_id, "\n".join(current_body_lines)))
                current_body_lines = []

            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            current_clause_id = build_clause_id(spec_name, heading_text)
        else:
            stripped = line.strip()
            if stripped:
                current_body_lines.append(stripped)

    # Don't forget the last section
    if current_body_lines:
        sections.append((current_clause_id, "\n".join(current_body_lines)))

    # Now chunk each section independently
    for clause_id, body in sections:
        if not body.strip():
            continue

        sentences = split_sentences(body)
        for chunk_text in sliding_window_chunks(sentences):
            chunk_text = chunk_text.strip()
            if not chunk_text or count_tokens(chunk_text) < 5:
                continue

            yield FCRAGDocument(
                doc_id=f"{txt_path.stem}_chunk_{chunk_idx:05d}",
                source_type="3gpp_spec",
                source_file=txt_path.name,
                clause_id=clause_id,
                text=chunk_text,
                metadata={
                    "spec_name": spec_name,
                    "chunk_index": chunk_idx,
                    "token_count": count_tokens(chunk_text),
                },
            )
            chunk_idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 & 3 — Pass-through JSONL (narratives already ≤125 tokens)
# ─────────────────────────────────────────────────────────────────────────────
def passthrough_jsonl(jsonl_path: Path) -> Generator[FCRAGDocument, None, None]:
    """
    Read a normalized JSONL file and yield FCRAGDocument objects.
    If any record's text exceeds MAX_TOKENS, it gets chunked too.
    """
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            text = raw.get("text", "").strip()
            if not text:
                continue

            tok_count = count_tokens(text)

            # Short enough → pass through directly
            if tok_count <= MAX_TOKENS:
                yield FCRAGDocument(
                    doc_id=raw.get("doc_id", f"{jsonl_path.stem}_{i:05d}"),
                    source_type=raw.get("source_type", "unknown"),
                    source_file=raw.get("source_file", jsonl_path.name),
                    clause_id=raw.get("clause_id", ""),
                    text=text,
                    metadata=raw.get("metadata", {}),
                )
            else:
                # Rare — chunk the long text
                sentences = split_sentences(text)
                base_id = raw.get("doc_id", f"{jsonl_path.stem}_{i:05d}")
                for j, chunk_text in enumerate(sliding_window_chunks(sentences)):
                    chunk_text = chunk_text.strip()
                    if not chunk_text:
                        continue
                    yield FCRAGDocument(
                        doc_id=f"{base_id}_chunk_{j:03d}",
                        source_type=raw.get("source_type", "unknown"),
                        source_file=raw.get("source_file", jsonl_path.name),
                        clause_id=raw.get("clause_id", ""),
                        text=chunk_text,
                        metadata={
                            **raw.get("metadata", {}),
                            "chunk_index": j,
                            "token_count": count_tokens(chunk_text),
                        },
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline: chunk ALL sources → unified output JSONL files
# ─────────────────────────────────────────────────────────────────────────────
def run_chunker(verbose: bool = True) -> dict[str, int]:
    """
    Chunk all normalized data sources and write to data/processed/chunks/.
    Returns a dict of {collection_name: chunk_count}.
    """
    CHUNK_OUTPUT.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    # ── Collection 1: 3gpp_specs ────────────────────────────────────────────
    if verbose:
        print("\n[1/3] Chunking 3GPP spec .txt files → 3gpp_specs")

    txt_files = sorted(TXT_DIR.glob("*.txt"))
    count_3gpp = 0
    with open(CHUNK_OUTPUT / "3gpp_specs.jsonl", "w", encoding="utf-8") as f_out:
        for txt_path in tqdm(txt_files, desc="  3GPP specs", disable=not verbose):
            file_chunks = 0
            for chunk in chunk_3gpp_txt(txt_path):
                f_out.write(json.dumps(chunk.to_dict()) + "\n")
                file_chunks += 1
                count_3gpp += 1
            if verbose:
                tqdm.write(f"    {txt_path.name}: {file_chunks} chunks")

    stats["3gpp_specs"] = count_3gpp
    if verbose:
        print(f"  ✅ 3gpp_specs: {count_3gpp:,} total chunks")

    # ── Collection 2: simu5g_narratives ─────────────────────────────────────
    if verbose:
        print("\n[2/3] Passing through Simu5G narratives → simu5g_narratives")

    narrative_file = SIMU5G_DIR / "fault_narratives_normalized.jsonl"
    count_simu5g = 0
    if narrative_file.exists():
        with open(CHUNK_OUTPUT / "simu5g_narratives.jsonl", "w", encoding="utf-8") as f_out:
            for doc in passthrough_jsonl(narrative_file):
                f_out.write(json.dumps(doc.to_dict()) + "\n")
                count_simu5g += 1
        stats["simu5g_narratives"] = count_simu5g
        if verbose:
            print(f"  ✅ simu5g_narratives: {count_simu5g:,} documents")
    else:
        print(f"  [WARNING] {narrative_file} not found. Skipping.")
        stats["simu5g_narratives"] = 0

    # ── Collection 3: alarm_history ─────────────────────────────────────────
    if verbose:
        print("\n[3/3] Passing through KPI narratives → alarm_history")

    kpi_file = KPI_DIR / "kpi_narratives.jsonl"
    count_kpi = 0
    if kpi_file.exists():
        with open(CHUNK_OUTPUT / "alarm_history.jsonl", "w", encoding="utf-8") as f_out:
            for doc in passthrough_jsonl(kpi_file):
                f_out.write(json.dumps(doc.to_dict()) + "\n")
                count_kpi += 1
        stats["alarm_history"] = count_kpi
        if verbose:
            print(f"  ✅ alarm_history: {count_kpi:,} documents")
    else:
        print(f"  [WARNING] {kpi_file} not found. Skipping.")
        stats["alarm_history"] = 0

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: iterate all chunks from disk (used by embedder)
# ─────────────────────────────────────────────────────────────────────────────
def iter_chunks(collection: str) -> Iterator[FCRAGDocument]:
    """
    Iterate over chunks for a given collection name from disk.
    collection: "3gpp_specs" | "simu5g_narratives" | "alarm_history"
    """
    jsonl_path = CHUNK_OUTPUT / f"{collection}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {jsonl_path}\n"
            f"Run chunker first: python scripts/ingest_all.py --step chunk"
        )
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                yield FCRAGDocument(**raw)


def get_chunk_stats() -> dict[str, int]:
    """Return chunk counts per collection from disk."""
    stats = {}
    for collection in ("3gpp_specs", "simu5g_narratives", "alarm_history"):
        path = CHUNK_OUTPUT / f"{collection}.jsonl"
        if path.exists():
            with open(path) as f:
                stats[collection] = sum(1 for line in f if line.strip())
        else:
            stats[collection] = 0
    return stats
