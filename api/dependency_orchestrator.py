"""dependency_orchestrator.py

Agentic, dependency-aware processing pipeline for the /document endpoint.

Responsibilities:
1. Chunk a file (already delegated via CHUNKING_STRATEGIES).
2. For each chunk run: analysis -> documentation -> refinement (JSON enforced).
3. Extract and normalize dependency names from analysis + lightweight regex heuristics.
4. Aggregate per-file enriched chunk metadata and create a single embedding per file
   (keeps compatibility with previous retrieval granularity while storing richer metadata).
5. Return a document record ready to be added to the vector DB.

If desired later, we can switch to per-chunk embeddings (granularity upgrade) with minimal changes.

Selective Reprocessing Strategy (conceptual, implemented incrementally):
1. Compute stable hash per chunk (already have _hash_text of file_path::index + code hash if needed).
2. Before processing, check if an existing document (file-level or chunk-level) exists in vector DB with same hash marker.
3. If unchanged, skip LLM calls and reuse stored metadata (avoids cost and latency).
4. If only subset of chunks changed, merge old unchanged chunk metadata with new results, recompute embeddings only for changed scope (or file aggregate if single-embedding mode).
5. Maintain a lightweight manifest mapping file_path -> list[chunk_hash].
6. Future: store this manifest externally (json) to detect deletes/renames.

Current code prepares the primitives (hashing, per-chunk IDs). Next step: integrate a cache lookup layer.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set, Optional
import zlib
import base64
import time

DEFAULT_TRUNCATION_LIMIT = 4000  # Hard cap to prevent oversized metadata blobs
DEFAULT_COMPRESSION_MIN_RATIO = 0.85
FALLBACK_VALUE = "None found"

from api.prompt_templates import (
    get_chunk_analysis_prompt,
    get_chunk_documentation_prompt,
    get_chunk_refinement_prompt,
)
from api.llm_providers import LlmClient, LlmError


# ----------------------------- Utility Structures -----------------------------

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_code_chunk(code: str) -> str:
    """Stable hash for a chunk's raw code (used for selective reprocessing)."""
    return _hash_text(code)


def decompress_if_needed(value: str) -> str:
    """If a metadata field was compressed (::zlib64::<base64>) return the decompressed text.

    Silently returns original value if pattern not matched or decompression fails.
    """
    if not isinstance(value, str):
        return value
    prefix = "::zlib64::"
    if not value.startswith(prefix):
        return value
    b64 = value[len(prefix):]
    try:
        raw = base64.b64decode(b64)
        return zlib.decompress(raw).decode('utf-8', errors='replace')
    except Exception:
        return value


@dataclass
class ChunkEnriched:
    index: int
    chunk_type: str
    original_code: str
    analysis: Dict[str, Any] = field(default_factory=dict)
    documentation: Dict[str, Any] = field(default_factory=dict)
    refined: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_metadata_dict(self, truncation_limit: int) -> Dict[str, Any]:
        original = self.original_code[:truncation_limit]
        if len(self.original_code) > truncation_limit:
            self.warnings.append(f"original_code_truncated_{truncation_limit}")
        return {
            "chunk_index": self.index,
            "chunk_type": self.chunk_type,
            "original_code": original,
            "analysis": json.dumps(self.analysis, ensure_ascii=False),
            "documentation": json.dumps(self.documentation, ensure_ascii=False),
            "refined_documentation": json.dumps(self.refined, ensure_ascii=False),
            "dependencies": json.dumps(self.dependencies, ensure_ascii=False),
            "risks": json.dumps(self.risks, ensure_ascii=False),
            "relationships": json.dumps(self.relationships, ensure_ascii=False),
            "warnings": json.dumps(self.warnings, ensure_ascii=False),
            "summary": self.refined.get("summary") or self.documentation.get("summary") or self.analysis.get("summary", ""),
        }


class DependencyExtractor:
    """Lightweight regex-based dependency extraction fallback.

    We still primarily rely on the model's own dependency list in analysis JSON,
    but this heuristic ensures we capture obvious import/include patterns even
    if the model omits them.
    """

    # Basic patterns across multiple languages
    PATTERNS = [
        r"from\s+([\w\.]+)\s+import",          # Python from import
        r"import\s+([\w\.]+)",                  # Generic import
        r"require\([\"']([^\"']+)[\"']\)",  # CommonJS require
        r"include\s+[\"']([^\"']+)[\"']",    # C/C++ like include
        r"using\s+([\w\.]+)",                   # Scala/C# using
        r"package\s+([\w\.]+)",                 # Package decl
        r"class\s+\w+\s+extends\s+([\w\.]+)", # Inheritance
        r"@import\s+[\"']([^\"']+)[\"']",    # CSS import
    ]

    @classmethod
    def extract(cls, code: str) -> List[str]:
        import re
        found = []
        for pat in cls.PATTERNS:
            for m in re.findall(pat, code):
                if m and m not in found:
                    found.append(m.strip())
        return found


class AgentOrchestrator:
    """Encapsulates the agentic processing for a single file.

    Current strategy (per file):
    1. Receive pre-chunked (content, type) list.
    2. For each chunk run analysis -> documentation -> refinement (with JSON parsing & retries handled by LLM provider wrapper for JSON robustness where possible).
    3. Merge model-declared dependencies with heuristic extraction.
    4. Produce aggregated embedding text (join refined summaries + risks for semantic retrieval).
    5. Return vector DB ready record structure.
    """

    def __init__(
        self,
        llm_client: LlmClient,
        llm_client_embedding: LlmClient,
        max_attempts: int = 2,
        per_chunk_embeddings: bool = True,
        compress_large_fields: bool = True,
        truncation_limit: int = DEFAULT_TRUNCATION_LIMIT,
        compression_min_ratio: float = DEFAULT_COMPRESSION_MIN_RATIO,
    ):
        self.llm_client = llm_client
        self.embedding_client = llm_client_embedding
        self.max_attempts = max_attempts
        self.per_chunk_embeddings = per_chunk_embeddings
        self.compress_large_fields = compress_large_fields
        self.truncation_limit = truncation_limit
        self.compression_min_ratio = compression_min_ratio
        logging.info(
            f"[ORCH] Init AgentOrchestrator per_chunk_embeddings={per_chunk_embeddings} compress={compress_large_fields} "
            f"trunc_limit={truncation_limit} min_comp_ratio={compression_min_ratio}"
        )

    # ---- Core LLM Call Helpers ----
    def _call_json(self, prompt: str, stage: str, file_path: str, chunk_index: int, expected: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        warnings_local: List[str] = []
        result: Dict[str, Any] = {}
        try:
            raw = self.llm_client.generate_json(prompt, expected_keys=expected, attempts=self.max_attempts)
            if isinstance(raw, dict):
                result = raw
                missing = [k for k in expected if k not in result]
                if missing:
                    msg = f"missing_keys_{stage.lower()}={missing}"
                    warnings_local.append(msg)
                    logging.warning(f"[{stage}] Chunk {chunk_index} {file_path}: {msg}")
                # Sanitize: ensure expected keys exist
                for k in expected:
                    if k not in result or result[k] in (None, ""):
                        result.setdefault(k, FALLBACK_VALUE)
            else:
                warnings_local.append("non_dict_response")
                logging.warning(f"[{stage}] Chunk {chunk_index} {file_path}: resultado no dict final")
        except Exception as e:
            logging.error(f"[{stage}] Error definitivo chunk {chunk_index} {file_path}: {e}")
            warnings_local.append(f"exception:{type(e).__name__}")
        return result, warnings_local

    # ---- Public API ----
    def _maybe_compress(self, text: str) -> str:
        if not self.compress_large_fields:
            return text
        if not text:
            return text
        raw = text.encode('utf-8')
        comp = zlib.compress(raw, level=6)
        if len(comp) < len(raw) * self.compression_min_ratio:  # only keep compression if worthwhile
            return "::zlib64::" + base64.b64encode(comp).decode('ascii')
        return text

    def process_file(self, file_path: str, chunks: List[Tuple[str, str]]) -> Dict[str, Any]:
        enriched_chunks: List[ChunkEnriched] = []
        logging.info(f"[ORCH] PROCESS_FILE start file={file_path} chunk_count={len(chunks)}")

        def process_single_chunk(idx: int, code: str, chunk_type: str) -> ChunkEnriched:
            ce = ChunkEnriched(index=idx, chunk_type=chunk_type, original_code=code)

            # Analysis
            logging.info(f"[ORCH] ANALYSIS start file={file_path} chunk={idx} type={chunk_type}")
            analysis_prompt = get_chunk_analysis_prompt(chunk_type, file_path, code)
            ce.analysis, warn_analysis = self._call_json(
                analysis_prompt, "ANALYSIS", file_path, idx,
                ["summary", "dependencies", "key_elements", "context", "risks", "relationships"]
            )
            ce.warnings.extend(warn_analysis)

            # Dependencias modelo + heurísticas
            model_dependencies: List[str] = []
            if isinstance(ce.analysis, dict):
                raw_deps = ce.analysis.get("dependencies")
                if isinstance(raw_deps, list):
                    model_dependencies = [str(d).strip() for d in raw_deps if d]
                elif isinstance(raw_deps, dict):
                    vals = [str(v).strip() for v in raw_deps.values() if v]
                    model_dependencies = vals or [str(k).strip() for k in raw_deps.keys() if k]
                elif isinstance(raw_deps, (str, int, float)):
                    s = str(raw_deps).strip()
                    if s:
                        model_dependencies = [s]

            heuristic_dependencies = DependencyExtractor.extract(code)
            combined_deps = []
            for d in model_dependencies + heuristic_dependencies:
                if d and d not in combined_deps:
                    combined_deps.append(d)
            ce.dependencies = combined_deps

            if isinstance(ce.analysis, dict):
                # Normaliza riesgos
                risks = ce.analysis.get("risks") or []
                if isinstance(risks, (str, int, float)):
                    risks = [risks]
                norm_risks = []
                for r in risks:
                    try:
                        norm_risks.append(str(r).strip())
                    except Exception:
                        pass
                ce.risks = norm_risks
                relationships = ce.analysis.get("relationships") or []
                if isinstance(relationships, str):
                    relationships = [relationships]
                ce.relationships = relationships

            # Documentation
            logging.info(f"[ORCH] DOCUMENTATION start file={file_path} chunk={idx}")
            logging.info(f"[ORCH] DOCUMENTATION chunk_info={code.replace(chr(10), ' ')[:50]}")
            doc_prompt = get_chunk_documentation_prompt(chunk_type, file_path, code)
            ce.documentation, warn_doc = self._call_json(
                doc_prompt, "DOCUMENTATION", file_path, idx,
                ["summary", "details", "examples", "notes", "cross_references", "context"]
            )
            ce.warnings.extend(warn_doc)

            # Refinement
            try:
                documentation_json_str = json.dumps(ce.documentation, ensure_ascii=False)
            except Exception:
                documentation_json_str = "{}"
            logging.info(f"[ORCH] REFINEMENT start file={file_path} chunk={idx}")
            refine_prompt = get_chunk_refinement_prompt(documentation_json_str)
            ce.refined, warn_ref = self._call_json(
                refine_prompt, "REFINEMENT", file_path, idx,
                ["summary", "details", "examples", "notes", "cross_references", "context"]
            )
            ce.warnings.extend(warn_ref)
            logging.info(f"[ORCH] FINISHED chunk file={file_path} chunk={idx} warnings={len(ce.warnings)} deps={len(ce.dependencies)}")
            return ce

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_single_chunk, idx, code, chunk_type): idx
                for idx, (code, chunk_type) in enumerate(chunks)
            }
            for future in as_completed(futures):
                ce = future.result()
                enriched_chunks.append(ce)

        # Ordenar chunks en base a su índice original
        enriched_chunks.sort(key=lambda c: c.index)

        # Aggregate embedding text
        aggregate_text_parts = []
        for c in enriched_chunks:
            summary = c.refined.get("summary") if c.refined else None
            if not summary:
                summary = c.documentation.get("summary") if c.documentation else None
            if not summary:
                summary = c.analysis.get("summary") if c.analysis else None
            if summary:
                aggregate_text_parts.append(summary)
            if c.risks:
                aggregate_text_parts.append("RISKS: " + "; ".join(c.risks))
        aggregate_text = "\n".join(aggregate_text_parts) or f"File {file_path}"
        logging.info(f"[ORCH] AGGREGATE file={file_path} chunks={len(enriched_chunks)} text_len={len(aggregate_text)}")

        documents: List[Dict[str, Any]] = []
        dependency_union_set: Set[str] = {d for c in enriched_chunks for d in c.dependencies}

        if self.per_chunk_embeddings:
            # One vector per chunk
            for c in enriched_chunks:
                # Build chunk text for embedding
                chunk_text = c.refined.get("summary") or c.documentation.get("summary") or c.analysis.get("summary") or c.original_code[:500]
                try:
                    emb = self.embedding_client.generate_embedding(chunk_text)
                except LlmError as e:
                    logging.error(f"Embedding error chunk {c.index} {file_path}: {e}")
                    emb = []
                meta = c.to_metadata_dict(self.truncation_limit)
                # Optionally compress large json fields
                for key in ("analysis", "documentation", "refined_documentation"):
                    meta[key] = self._maybe_compress(meta[key])
                documents.append({
                    "id": _hash_text(f"{file_path}::{c.index}"),
                    "embedding": emb,
                    "metadata": {
                        "file_path": file_path,
                        "chunk_index": c.index,
                        "is_chunk": True,
                        "document_summary": (meta.get("summary") or "")[:500],
                        "enriched_chunk": json.dumps(meta, ensure_ascii=False),
                        "dependency_union": json.dumps(sorted(dependency_union_set), ensure_ascii=False),
                    }
                })
        else:
            # Single vector per file
            try:
                embedding = self.embedding_client.generate_embedding(aggregate_text)
            except LlmError as e:
                logging.error(f"Embedding error en {file_path}: {e}")
                embedding = []
            chunks_metadata = [c.to_metadata_dict(self.truncation_limit) for c in enriched_chunks]
            if self.compress_large_fields:
                chunks_metadata = [
                    {**m,
                     "analysis": self._maybe_compress(m["analysis"]),
                     "documentation": self._maybe_compress(m["documentation"]),
                     "refined_documentation": self._maybe_compress(m["refined_documentation"])}
                    for m in chunks_metadata
                ]
            documents.append({
                "id": _hash_text(file_path),
                "embedding": embedding,
                "metadata": {
                    "file_path": file_path,
                    "document_summary": aggregate_text[:2000],
                    "enriched_chunks": json.dumps(chunks_metadata, ensure_ascii=False),
                    "dependency_union": json.dumps(sorted(dependency_union_set), ensure_ascii=False),
                    "is_chunk": False,
                }
            })
        # Return unified structure (for now caller expects single; adapt if multi)
        if len(documents) == 1:
            return documents[0]
        return {"_batched": documents}


def build_dependency_graph(documents: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {}
    for doc in documents:
        meta = doc.get("metadata", {})
        path = meta.get("file_path")
        if not path:
            continue
        try:
            deps = json.loads(meta.get("dependency_union", "[]"))
        except Exception:
            deps = []
        graph.setdefault(path, set())
        for d in deps:
            graph[path].add(d)
    return graph


def invert_dependency_graph(graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    inverted: Dict[str, Set[str]] = {}
    for src, deps in graph.items():
        for d in deps:
            inverted.setdefault(d, set()).add(src)
    return inverted


def attach_dependents(documents: List[Dict[str, Any]], inverted: Dict[str, Set[str]]):
    for doc in documents:
        meta = doc.get("metadata", {})
        path = meta.get("file_path")
        dependents = sorted(inverted.get(path, []))
        meta["dependents_union"] = json.dumps(dependents, ensure_ascii=False)


def compute_quality_scores(documents: List[Dict[str, Any]]):
    for doc in documents:
        meta = doc.get("metadata", {})
        try:
            chunks_raw = meta.get("enriched_chunks")
            if not chunks_raw:
                continue
            chunks = json.loads(chunks_raw)
            total = len(chunks)
            if total == 0:
                continue
            completeness_scores = []
            for c in chunks:
                # Count how many expected keys present in refined_documentation or documentation
                try:
                    refined = json.loads(c.get("refined_documentation", "{}"))
                except Exception:
                    refined = {}
                expected = ["summary", "details", "examples", "notes", "cross_references", "context"]
                present = sum(1 for k in expected if k in refined and refined.get(k) not in (None, "", FALLBACK_VALUE))
                completeness_scores.append(present / len(expected))
            avg_score = sum(completeness_scores) / len(completeness_scores)
            meta["quality_score"] = round(avg_score, 3)
        except Exception as e:
            logging.warning(f"compute_quality_scores error: {e}")


__all__ = [
    "AgentOrchestrator",
    "DependencyExtractor",
    "ChunkEnriched",
    "build_dependency_graph",
    "invert_dependency_graph",
    "attach_dependents",
    "compute_quality_scores",
    # helpers
    "hash_code_chunk",
    "decompress_if_needed",
]
