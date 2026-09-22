import re
import json
import time
import os
from datetime import datetime
import logging
from typing import Any, Dict, List, Tuple, Set, Callable

from api.enhanced_prompts import (
    get_wiki_agent_orchestration_prompt,
    get_section_generation_prompt,
    get_wiki_refinement_prompt,
    get_wiki_structure_prompt,
)

Citation = Tuple[str, int]  # (file_path, chunk_id)


def extract_citations(markdown: str) -> Set[Citation]:
    """Extrae citas flexibles: admite '(path, chunk N)' y variantes en ES.
    Palabras clave aceptadas: chunk|trozo|fragmento|bloque (insensible a mayúsculas).
    También acepta 'Citation:' / 'Cita:' opcional antes.
    """
    cites: Set[Citation] = set()
    if not markdown:
        return cites
    patterns = [
        r"\((?:ver\s+)?([^,\)]+)\s*,\s*chunk\s+(\d+)\)",
        r"\((?:ver\s+)?([^,\)]+)\s*,\s*trozo\s+(\d+)\)",
        r"\((?:ver\s+)?([^,\)]+)\s*,\s*fragmento\s+(\d+)\)",
        r"\((?:ver\s+)?([^,\)]+)\s*,\s*bloque\s+(\d+)\)",
        r"(?:Cita(?:tion)?:\s*)?\(([^,\)]+)\s*,\s*chunk\s*:?\s*(\d+)\)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, markdown, flags=re.IGNORECASE):
            fp = (m.group(1) or '').strip()
            try:
                cid = int(m.group(2))
                cites.add((fp, cid))
            except Exception:
                continue
    return cites


def collect_all_chunks(list_chunks_fn: Callable[[], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Devuelve la lista de todos los chunks del repositorio (metadatos mínimos)."""
    t0 = time.time()
    chunks = list_chunks_fn() or []
    logging.info(f"[WIKI_AGENT] collect_all_chunks -> {len(chunks)} chunks (took {time.time() - t0:.2f}s)")
    return chunks


def compute_coverage(cited: Set[Citation], all_chunks: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    total = len(all_chunks)
    if total == 0:
        logging.info("[WIKI_AGENT] compute_coverage: no chunks available (coverage=1.0 by definition)")
        return 1.0, {"total": 0, "covered": 0, "by_file": {}}
    covered = 0
    by_file: Dict[str, Dict[str, Any]] = {}
    for md in all_chunks:
        fp = md.get("file_path")
        cid = md.get("chunk_index")
        hit = (fp, cid) in cited
        if hit:
            covered += 1
        if fp not in by_file:
            by_file[fp] = {"total": 0, "covered": 0, "missing_chunks": []}
        by_file[fp]["total"] += 1
        if hit:
            by_file[fp]["covered"] += 1
        else:
            by_file[fp]["missing_chunks"].append(cid)
    coverage = covered / total
    logging.info(f"[WIKI_AGENT] compute_coverage: covered={covered}/{total} -> {coverage:.3f}")
    return coverage, {"total": total, "covered": covered, "by_file": by_file}


def build_multi_queries(title_obj: Any) -> List[str]:
    """Genera variaciones del título/descripcion para mejorar el recall del buscador.
    - Acepta str o dict con 'title' y opcional 'description'.
    - Normaliza, extrae tokens y bigramas, agrega variantes de dominio y lenguaje.
    - Devuelve hasta ~20 queries deduplicadas (case-insensitive).
    """
    # 1) Normalizar entradas
    base = ""
    desc = ""
    if isinstance(title_obj, dict):
        base = str(title_obj.get("title") or title_obj.get("name") or title_obj.get("heading") or "").strip()
        desc = str(title_obj.get("description") or "").strip()
    else:
        base = str(title_obj or "").strip()

    seed = f"{base} {desc}".strip()

    # 2) Tokenizar simple y quitar stopwords comunes (es/en)
    import re as _re
    def _tokens(text: str) -> List[str]:
        toks = [t for t in _re.split(r"[^A-Za-z0-9_]+", text.lower()) if t]
        # Dividir camelCase/PascalCase sencillo
        split_more: List[str] = []
        for t in toks:
            split_more.extend(_re.sub(r"([a-z])([A-Z])", r"\1 \2", t).lower().split())
        return split_more or toks

    stop = {
        # ES
        "de","del","la","el","los","las","y","o","u","para","por","con","sin","en","un","una","unos","unas","al","como","sobre","entre","desde","hasta","segun","según","mas","más","menos","muy","seccion","sección","descripcion","descripción","resumen","detalle",
        # EN
        "the","a","an","and","or","to","for","of","in","on","by","with","from","as","about","into","over","under","between","section","description","summary","overview",
    }
    base_toks = [t for t in _tokens(base) if t not in stop]
    desc_toks = [t for t in _tokens(desc) if t not in stop]
    # Limitar tamaño de descripción para evitar ruido
    desc_toks = desc_toks[:12]

    def _unique(seq: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for s in seq:
            k = s.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(s.strip())
        return out

    def _bigrams(toks: List[str]) -> List[str]:
        return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks)-1)] if len(toks) > 1 else []

    main_phrase = " ".join(base_toks[:6]).strip() or base
    phrase_plus_desc = " ".join(_unique(base_toks + desc_toks))[:120].strip()
    bi = _bigrams(base_toks)[:4]

    # 3) Palabras clave de dominio/lenguaje frecuentes
    domain_kw = [
        "controller","service","endpoint","router","handler","view","model","entity","repository","repo","dao",
        "config","configuration","settings","properties","env",
        "job","task","batch","scheduler","worker","pipeline","etl",
        "spark","dataframe","dataset","rdd",
        "scala","python","java","sql","jdbc",
        "rest","api","graphql","http","client","server","middleware","logger","logging",
        "auth","security","jwt","oauth",
    ]

    # 4) Construir candidatos
    candidates: List[str] = []
    candidates.extend([q for q in [seed, base, main_phrase, phrase_plus_desc] if q])
    candidates.extend(bi)

    # Añadir combinaciones base/domain_kw (limitadas)
    base_for_combo = main_phrase or base
    for kw in domain_kw:
        if base_for_combo:
            candidates.append(f"{base_for_combo} {kw}")
    # Añadir variantes simples de lenguaje/tech si no hay base
    if not base_for_combo:
        candidates.extend(["scala", "python", "java", "spark", "sql", "rest api"])

    # 5) Deduplicar y recortar tamaño total
    candidates = _unique(candidates)
    MAX_Q = 20
    if len(candidates) > MAX_Q:
        candidates = candidates[:MAX_Q]

    return candidates


def build_multi_queries_smart(
    title_obj: Any,
    repo_context: Dict[str, Any],
    llm_expand_fn: Callable[[str], str],
    limit: int = 20,
) -> List[str]:
    """Usa el LLM para expandir queries con conocimiento ligero del repo.
    - repo_context: {"top_files": ["..."], "ext_counts": {"py":10,...}, "stack": ["python","scala",...]}
    - Output esperado: JSON array de strings; si falla, se hace fallback heurístico.
    """
    # Normalizar título/desc
    if isinstance(title_obj, dict):
        base = str(title_obj.get("title") or title_obj.get("name") or title_obj.get("heading") or "").strip()
        desc = str(title_obj.get("description") or "").strip()
    else:
        base = str(title_obj or "").strip()
        desc = ""

    stack = repo_context.get("stack") or []
    top_files = repo_context.get("top_files") or []
    ext_counts = repo_context.get("ext_counts") or {}

    steer = (
        "You are crafting high-recall search queries for a code-aware retriever.\n"
        "Constraints:\n"
        f"- Section title: '{base}'.\n"
        f"- Description: '{desc}'.\n"
        f"- Repo tech hints: {stack}. Extensions: {ext_counts}.\n"
        f"- Representative files: {top_files[:12]}.\n"
        f"- Produce up to {limit} concise queries, most specific first.\n"
        "- Mix exact phrases and key terms, include likely class/function/entity names if implied.\n"
        "- Prefer terms from the tech stack; avoid generic words.\n"
        "Output JSON array of strings only."
    )

    try:
        raw = llm_expand_fn(steer) or ""
        # Intentar extraer el primer array JSON
        import re as _re
        m = _re.search(r"\[.*?\]", raw, flags=_re.S)
        arr_txt = m.group(0) if m else raw.strip()
        data = json.loads(arr_txt)
        if isinstance(data, list):
            qs = [str(x).strip() for x in data if str(x).strip()]
            # recortar a límite y deduplicar
            seen: Set[str] = set()
            out: List[str] = []
            for q in qs:
                k = q.lower()
                if k and k not in seen:
                    seen.add(k)
                    out.append(q)
                if len(out) >= limit:
                    break
            if out:
                return out
    except Exception:
        # caemos a heurística
        pass

    # Fallback: heurístico existente
    return build_multi_queries(title_obj)[:limit]


def retrieve_for_section(
    queries: List[str],
    search_fn: Callable[[str, int, float], List[Dict[str, Any]]],
    top_k: int,
    score_threshold: float,
) -> List[Dict[str, Any]]:
    """
    Ejecuta búsquedas acumulando resultados y deduplicando por (file_path, chunk_index).
    search_fn: (query_text, top_k, score_threshold) -> docs estilo RAG
    """
    # Reciprocal Rank Fusion (RRF) across multi-queries
    K = 60
    scores: Dict[Tuple[str, int], float] = {}
    docmap: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for q in queries:
        q_log = (q[:120] + '...') if len(q) > 120 else q
        t0 = time.time()
        docs = search_fn(q, top_k=top_k, score_threshold=score_threshold) or []
        dt = time.time() - t0
        logging.info(f"[WIKI_AGENT] retrieve_for_section: query='{q_log}' got {len(docs)} doc groups in {dt:.2f}s (threshold={score_threshold}, top_k={top_k})")
        for rank, d in enumerate(docs, start=1):
            mds = d.get("metadatas") or []
            if not mds:
                continue
            md = mds[0] or {}
            fp = (md or {}).get("file_path")
            cid = (md or {}).get("chunk_index")
            if fp is None or cid is None:
                continue
            key = (fp, cid)
            docmap.setdefault(key, d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused = [docmap[k] for k, _ in ordered]
    logging.info(f"[WIKI_AGENT] retrieve_for_section: RRF fused results={len(fused)}")
    return fused


def generate_section_markdown(
    section_title: str,
    relevant_docs: List[Dict[str, Any]],
    max_context_tokens: int,
    llm_generate_markdown: Callable[[str], str],
    max_chars: int | None = None,
) -> Tuple[str, Set[Citation]]:
    logging.info(f"[WIKI_AGENT] generate_section_markdown: title='{section_title}', docs={len(relevant_docs)}, max_tokens={max_context_tokens}")
    t0 = time.time()
    prompt = get_section_generation_prompt(section_title, relevant_docs, max_context_tokens)
    md = llm_generate_markdown(prompt)
    dt = time.time() - t0
    # Clip long outputs proactively to avoid runaway verbosity
    if isinstance(max_chars, int) and max_chars > 0 and isinstance(md, str) and len(md) > max_chars:
        md = md[:max_chars] + "\n\n<!-- clipped -->\n"
    return md, extract_citations(md)


def generate_full_wiki(
    query: str,
    estructura: Any,
    todo_list: List[str],
    search_fn: Callable[[str, int, float], List[Dict[str, Any]]],
    list_chunks_fn: Callable[[], List[Dict[str, Any]]],
    llm_generate_markdown: Callable[[str], str],
    llm_refine_wiki: Callable[[str, str], str],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Orquesta la generación con control de cobertura, reintentos por pocas citas y refinamiento global.
    params:
      top_k:int, score_threshold:float, section_max_tokens:int, refine_passes:int,
      coverage_target:float (0-1), strict_citations:bool
    """
    top_k = int(params.get("top_k", 12))
    score_threshold = float(params.get("score_threshold", 0.5))
    section_max_tokens = int(params.get("section_max_tokens", 5000))
    refine_passes = int(params.get("refine_passes", 2))
    full_coverage = bool(params.get("full_coverage", False))
    smart_queries = bool(params.get("smart_queries", False))
    coverage_target = 1.0 if full_coverage else float(params.get("coverage_target", 0.9))
    strict_citations = bool(params.get("strict_citations", True))
    save_intermediate = bool(params.get("save_intermediate", False))
    dump_dir = params.get("dump_dir") or os.path.join(os.getcwd(), "Pruebas")
    # Guardrail parameters (caps and limits)
    max_sections = int(params.get("max_sections", 40))
    section_char_limit = int(params.get("section_char_limit", 6000))
    refine_char_limit = int(params.get("refine_char_limit", 7000))
    global_refine_char_limit = int(params.get("global_refine_char_limit", 120000))
    # Diversificación por MMR y condensación opcional
    use_mmr = bool(params.get("use_mmr", True))
    mmr_lambda = float(params.get("mmr_lambda", 0.7))
    embed_text_fn = params.get("embed_text_fn")
    get_embeddings_fn = params.get("get_embeddings_fn")
    condense_final = bool(params.get("condense_final", False))
    condense_target_words = int(params.get("condense_target_words", 1500))
    if save_intermediate:
        try:
            os.makedirs(dump_dir, exist_ok=True)
        except Exception as e:
            logging.warning(f"[WIKI_AGENT] could not create dump_dir '{dump_dir}': {e}")
            save_intermediate = False
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    logging.info(
        f"[WIKI_AGENT] START generate_full_wiki | top_k={top_k} thr={score_threshold} tokens/section={section_max_tokens} "
        f"refine_passes={refine_passes} coverage_target={coverage_target} strict_citations={strict_citations}"
    )

    all_chunks = collect_all_chunks(list_chunks_fn)
    # Construir un contexto ligero del repo para smart queries
    repo_ctx: Dict[str, Any] = {"top_files": [], "ext_counts": {}, "stack": []}
    try:
        exts: Dict[str, int] = {}
        paths: List[str] = []
        for ch in all_chunks[:1000]:  # limitar para costos
            fp = str(ch.get("file_path") or "")
            if fp:
                paths.append(fp)
                _, ext = os.path.splitext(fp)
                ext = (ext or "").lstrip(".").lower()
                if ext:
                    exts[ext] = exts.get(ext, 0) + 1
        _unique_files = sorted(set(paths))[:50]
        repo_ctx["top_files"] = _unique_files
        repo_ctx["ext_counts"] = exts
        stack = []
        # inferir stack simple
        if exts.get("py"):
            stack.append("python")
        if exts.get("scala"):
            stack.append("scala")
        if exts.get("java"):
            stack.append("java")
        if exts.get("js") or exts.get("ts"):
            stack.append("javascript/typescript")
        if exts.get("sql"):
            stack.append("sql")
        repo_ctx["stack"] = stack
    except Exception:
        pass
    draft_parts: List[str] = []
    cited: Set[Citation] = set()

    # Plan breve (opcional)
    try:
        plan_prompt = get_wiki_agent_orchestration_prompt(estructura, todo_list, query)
        t0 = time.time()
        _ = llm_generate_markdown(plan_prompt)
        logging.info(f"[WIKI_AGENT] plan generated (took {time.time() - t0:.2f}s)")
    except Exception:
        logging.info("[WIKI_AGENT] plan generation skipped (no-op or provider lacks method)")

    sections_meta: List[Dict[str, Any]] = []
    output_language = str(params.get("output_language", "en")).strip() or "en"
    final_language = str(params.get("final_language", "es")).strip() or "es"

    # Trim overly long todo lists
    if isinstance(todo_list, list) and len(todo_list) > max_sections:
        logging.info(f"[WIKI_AGENT] trimming todo_list from {len(todo_list)} to {max_sections}")
        _todo = (todo_list or [])[:max_sections]
    else:
        _todo = todo_list or []

    for idx, sec in enumerate(_todo, start=1):
        sec = sec["section"]
        sec_t0 = time.time()
        logging.info(f"[WIKI_AGENT] SECTION {idx}/{len(todo_list)}: '{sec}'")
        # Normalizar título/descr para prompts
        if isinstance(sec, dict):
            sec_title = str(sec.get("section") or sec.get("title") or sec.get("name") or sec.get("heading") or "").strip() or "Untitled Section"
            sec_desc = str(sec.get("task") or sec.get("description") or "").strip()
            title_for_prompt = sec_title if not sec_desc else f"{sec_title}: {sec_desc}"
        else:
            sec_title = str(sec or "").strip() or "Untitled Section"
            title_for_prompt = sec_title
        if smart_queries:
            # Reutiliza el generador de markdown como expansor de listas breves
            def _llm_expand(prompt: str) -> str:
                try:
                    return llm_generate_markdown(prompt)
                except Exception:
                    return ""
            queries = build_multi_queries_smart({"title": sec_title, "description": sec_desc if isinstance(sec, dict) else ""}, repo_ctx, _llm_expand)
            logging.info(f"[WIKI_AGENT] smart queries built: {queries}")
        else:
            queries = build_multi_queries({"title": sec_title, "description": sec_desc if isinstance(sec, dict) else ""})
        logging.info(f"[WIKI_AGENT] queries built: {queries}")
        docs = retrieve_for_section(queries, search_fn, top_k, score_threshold)
        # MMR diversification if possible
        if use_mmr and callable(embed_text_fn) and callable(get_embeddings_fn):
            try:
                def _norm(v: list[float]) -> float:
                    import math
                    return math.sqrt(sum((x or 0.0) * (x or 0.0) for x in v)) or 1.0
                def _cos(a: list[float], b: list[float]) -> float:
                    if not a or not b:
                        return 0.0
                    num = sum((ai or 0.0) * (bi or 0.0) for ai, bi in zip(a, b))
                    return num / (_norm(a) * _norm(b))
                cand_ids: list[str] = []
                for d in docs:
                    _ids = d.get('ids') or []
                    if _ids and isinstance(_ids, list) and _ids[0]:
                        cand_ids.append(str(_ids[0]))
                emb_map = get_embeddings_fn(cand_ids) if cand_ids else {}
                qemb = embed_text_fn(sec_title)
                if emb_map and qemb:
                    selected: list[int] = []
                    remaining = list(range(len(docs)))
                    sim_q = []
                    for i in range(len(docs)):
                        _ids = docs[i].get('ids') or []
                        emb = emb_map.get(str(_ids[0])) if _ids else None
                        sim_q.append(_cos(qemb, emb) if emb else 0.0)
                    while remaining and len(selected) < min(top_k, len(docs)):
                        best_idx = None
                        best_score = -1e18
                        for i in remaining:
                            red = 0.0
                            for j in selected:
                                _idi = docs[i].get('ids') or []
                                _idj = docs[j].get('ids') or []
                                ei = emb_map.get(str(_idi[0])) if _idi else None
                                ej = emb_map.get(str(_idj[0])) if _idj else None
                                if ei and ej:
                                    red = max(red, _cos(ei, ej))
                            score = mmr_lambda * sim_q[i] - (1.0 - mmr_lambda) * red
                            if score > best_score:
                                best_score, best_idx = score, i
                        selected.append(best_idx)
                        remaining = [r for r in remaining if r != best_idx]
                    docs = [docs[i] for i in selected]
                    logging.info(f"[WIKI_AGENT] MMR applied: selected {len(docs)} docs")
            except Exception:
                logging.info("[WIKI_AGENT] MMR skipped (error)")
        # Language steering: small prefix
        lang_prefix = f"[Write this section in {output_language}.]\n"
        logging.info(f"[WIKI_AGENT] section_title -> '{title_for_prompt}'")
        md, cits = generate_section_markdown(
            lang_prefix + title_for_prompt,
            docs,
            section_max_tokens,
            llm_generate_markdown,
            max_chars=section_char_limit,
        )
        logging.info(f"[WIKI_AGENT] section '{sec}' -> citations={len(cits)}")
        retried = False
        if strict_citations and len(cits) < 2:
            logging.info(f"[WIKI_AGENT] few citations ({len(cits)}). Retrying with broader recall...")
            docs = retrieve_for_section(queries, search_fn, top_k=min(top_k * 2, 36), score_threshold=max(0.4, score_threshold - 0.05))
            md_retry, cits_retry = generate_section_markdown(
                lang_prefix + sec,
                docs,
                section_max_tokens,
                llm_generate_markdown,
                max_chars=section_char_limit,
            )
            if len(cits_retry) >= len(cits):
                md, cits = md_retry, cits_retry
                logging.info(f"[WIKI_AGENT] retry improved citations -> {len(cits)}")
            else:
                logging.info(f"[WIKI_AGENT] retry did not improve citations ({len(cits_retry)} <= {len(cits)})")
            retried = True
        # If still no citations under strict mode, append fallback citations from retrieved docs
        if strict_citations and len(cits) == 0:
            fallback = []
            for d in (docs or [])[:3]:
                md0 = (d.get('metadatas') or [{}])[0] or {}
                fp0 = md0.get('file_path')
                cid0 = md0.get('chunk_index')
                if fp0 is None or cid0 is None:
                    continue
                fallback.append(f"({fp0}, chunk {cid0})")
            if fallback:
                md = (md or '').rstrip() + "\n\nFuentes:\n" + "\n".join(f"- {f}" for f in fallback)
                cits = extract_citations(md)
                logging.info(f"[WIKI_AGENT] appended fallback citations -> {len(cits)}")
            else:
                safe_title = sec_title if isinstance(sec, dict) else str(sec or "Sección")
                md = f"## {safe_title}\n\n_No hay suficiente contexto con citas para redactar esta sección de forma fundamentada. TODO: ampliar cobertura de chunks relevantes._"
        draft_parts.append(md)
        cited |= cits

        # Save section artifacts
        try:
            meta = {
                "index": idx,
                "title": sec_title,
                "queries": queries,
                "docs_count": len(docs),
                "citations_count": len(cits),
                "citations": sorted(list(cits)),
                "retried": retried,
                "chars": len(md or ''),
            }
            sections_meta.append(meta)
            if save_intermediate:
                # sanitize title for filename
                safe_title = ''.join(ch if ch.isalnum() or ch in (' ', '_', '-') else '_' for ch in (sec_title or 'section')).strip().replace(' ', '_')
                sec_path = os.path.join(dump_dir, f"{ts}_section_{idx:02d}_{safe_title}.md")
                with open(sec_path, 'w', encoding='utf-8') as f:
                    f.write(md or '')
                meta_path = os.path.join(dump_dir, f"{ts}_section_{idx:02d}_{safe_title}.json")
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as se:
            logging.warning(f"[WIKI_AGENT] could not save section {idx} artifacts: {se}")
        # record elapsed per section
        try:
            sections_meta[-1]["elapsed_sec"] = round(time.time() - sec_t0, 3)
        except Exception:
            pass

    draft = "\n\n".join(draft_parts)
    logging.info(f"[WIKI_AGENT] draft assembled: {len(draft)} chars")

    coverage, detail = compute_coverage(cited, all_chunks)
    # Pass 1: si falta cobertura, anexar apéndice informativo
    if coverage < coverage_target:
        lines = ["\n\n## Apéndice: Archivos no cubiertos completamente\n"]
        for fp, s in (detail.get("by_file") or {}).items():
            if s.get("covered", 0) < s.get("total", 0):
                miss = s.get("missing_chunks", [])
                lines.append(f"- {fp}: chunks sin cubrir {sorted(miss)}")
        draft += "\n" + "\n".join(lines)
        logging.info("[WIKI_AGENT] coverage below target -> appended appendix of missing chunks")

    # Pass 2: si se requiere cobertura total o falta cobertura, generar contenido por chunk faltante
    def _best_section_index_for(fp: str, sec_titles: List[str]) -> int | None:
        """Empareja heurísticamente por ruta/filename con títulos de secciones."""
        base = os.path.basename(fp).lower()
        tokens = set(re.split(r"[^a-z0-9]+", base)) - {"", "scala", "py", "java", "md"}
        best_idx = None
        best_score = 0
        for i, title in enumerate(sec_titles):
            tks = set(re.split(r"[^a-z0-9]+", (title or "").lower())) - {""}
            score = len(tokens & tks)
            if score > best_score:
                best_score, best_idx = score, i
        return best_idx if best_score >= 1 else None

    if full_coverage or coverage < coverage_target:
        missing: List[Tuple[str, int]] = []
        for fp, s in (detail.get("by_file") or {}).items():
            miss = s.get("missing_chunks", [])
            for cid in sorted(miss):
                missing.append((fp, cid))
        logging.info(f"[WIKI_AGENT] full_coverage pass: missing chunks to cover = {len(missing)}")

        # Index rápido de chunks por (fp, cid)
        chunk_index_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for ch in all_chunks:
            key = (ch.get("file_path"), ch.get("chunk_index"))
            if key not in chunk_index_map:
                chunk_index_map[key] = ch

        # Generar contenido por chunk y colocarlo en la mejor sección o como sección adicional
        added_sections: List = []

        logging.info(f"[WIKI_AGENT] todo_list -> '{todo_list}' -> citations={len(cits)}")
        sec_titles = [i["section"] for i in todo_list]
        for m_idx, (fp, cid) in enumerate(missing, start=1):
            ch = chunk_index_map.get((fp, cid)) or {}
            base = os.path.basename(fp or "")
            ch_summary = (ch.get("summary") or "").strip()
            ch_code = (ch.get("original_code") or "").strip()
            # Armar doc relevante mínimo para el prompt
            md_obj = {"file_path": fp, "chunk_index": cid, "chunk_type": ch.get("chunk_type")}
            doc_obj = ch_code or ch_summary or f"Chunk {cid} from {fp}"
            relevant_docs = [{"metadatas": [md_obj], "documents": [doc_obj]}]
            # Título y prompt con steering de idioma
            lang_prefix = f"[Write this section in {output_language}.]\n"
            title = f"Additional coverage: {base} (chunk {cid})"
            add_md, add_cits = generate_section_markdown(lang_prefix + title, relevant_docs, max(800, section_max_tokens // 8), llm_generate_markdown)
            # Asegurar al menos una cita explícita si el modelo no incluyó
            if (fp, cid) not in add_cits:
                add_md = (add_md or "") + f"\n\n(Citation: ({fp}, chunk {cid}))\n"
                add_cits.add((fp, cid))

            # Elegir sección destino
            dst = _best_section_index_for(fp, sec_titles)
            if dst is None:
                added_sections.append(add_md)
            else:
                # Adjuntar como subsección al final de la sección elegida
                draft_parts[dst] = (draft_parts[dst] or "") + "\n\n" + add_md
            cited |= add_cits

            # Guardar artefactos si aplica
            if save_intermediate:
                try:
                    safe_file = ''.join(ch if ch.isalnum() or ch in (' ', '_', '-') else '_' for ch in base).strip().replace(' ', '_')
                    miss_path = os.path.join(dump_dir, f"{ts}_missing_{m_idx:04d}_{safe_file}_chunk_{cid}.md")
                    with open(miss_path, 'w', encoding='utf-8') as f:
                        f.write(add_md or '')
                except Exception as me:
                    logging.warning(f"[WIKI_AGENT] could not save missing-chunk artifact: {me}")

        # Si hubo secciones adicionales, anexarlas al final en bloque
        if added_sections:
            draft_parts.append("\n\n".join(added_sections))

        # Recalcular draft y cobertura tras la ampliación
        draft = "\n\n".join(draft_parts)
        coverage, detail = compute_coverage(cited, all_chunks)
        logging.info(f"[WIKI_AGENT] coverage after full_coverage pass: {coverage:.3f}")

    # Helpers: limpieza de artefactos de prompt y división por secciones H2
    def _clean_artifacts(text: str) -> str:
        if not text:
            return text
        # Quitar bloques de reglas/prompts comunes
        patterns = [
            r"\*\*Reglas:\*\*[\s\S]*?(?=(\n##|\Z))",
            r"\*\*Contexto relevante:\*\*[\s\S]*?(?=(\n##|\Z))",
            r"Entrega la sección en Markdown, encabezada por:[\s\S]*?(?=(\n##|\Z))",
            r"\[Write this section in [^\]]+\]\.?\s*\n?",
            r"```markdown\s*",
        ]
        out = text
        import re as _re
        for p in patterns:
            out = _re.sub(p, "", out, flags=_re.IGNORECASE)
        # Normalizar dobles saltos y espacios
        out = _re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()

    def _split_h2_sections(text: str) -> List[Tuple[str, str]]:
        import re as _re
        lines = text.splitlines()
        sections: List[Tuple[str, List[str]]] = []
        current_title = None
        current_buf: List[str] = []
        for ln in lines:
            if ln.startswith("## "):
                if current_title is not None:
                    sections.append((current_title, current_buf))
                current_title = ln[3:].strip()
                current_buf = []
            else:
                current_buf.append(ln)
        if current_title is None:
            # No H2 found; treat whole as one section with generic title
            return [("Documento", text)]
        sections.append((current_title or "Sección", current_buf))
        return [(t, "\n".join(buf).strip()) for t, buf in sections]

    # Preparar títulos objetivo (si existen)
    target_titles: List[str] = []
    for t in (todo_list or []):
        if isinstance(t, dict):
            nm = str(t.get("title") or t.get("name") or t.get("heading") or "").strip()
        else:
            nm = str(t or "").strip()
        if nm:
            target_titles.append(nm)

    # Refinamiento por sección: corrige ortografía, limpia artefactos, conserva citas, permite crear nueva sección si no encaja
    h2_sections = _split_h2_sections(draft)
    refined_sections: List[str] = []
    for i, (title, body) in enumerate(h2_sections, start=1):
        sec_md = f"## {title}\n\n{body}".strip()
        sec_md = _clean_artifacts(sec_md)
        steer = (
            f"You are editing a single wiki section. Tasks:\n"
            f"1) Proofread and fix spelling/grammar in {final_language}.\n"
            f"2) Remove generator artifacts and prompts (e.g., 'Reglas:', 'Contexto relevante:', 'Entrega la sección...', stray ```markdown fences).\n"
            f"3) Keep all grounded technical details, code blocks, and preserve any citations like (file_path, chunk N). Do not invent new file paths or chunk IDs.\n"
            f"4) Evaluate fit against these target section titles: {target_titles}. If it doesn't fit, you may rename the H2 heading to a better, concise title in {final_language}.\n"
            f"5) If the content clearly contains multiple unrelated topics, split into multiple H2 sections; otherwise keep a single H2.\n"
            f"5) When appropriate, add a concise Mermaid diagram (as ```mermaid) for data schema lists (classDiagram) or simple flows (flowchart LR).\n"
            f"Output only valid Markdown for this single section, starting with an H2 heading."
        )
        t0 = time.time()
        try:
            sec_refined = llm_refine_wiki(sec_md, steer) or sec_md
        except Exception:
            logging.warning(f"[WIKI_AGENT] per-section refine failed; keeping original for '{title}'")
            sec_refined = sec_md
        dt = time.time() - t0
        logging.info(f"[WIKI_AGENT] refined section {i}/{len(h2_sections)} ('{title[:40]}...') in {dt:.2f}s -> {len(sec_refined)} chars")
        refined_sections.append(_clean_artifacts(sec_refined))

    refined = "\n\n".join(refined_sections).strip()
    # Refinamiento global ligero para cohesión, manteniendo contenido
    if refine_passes > 0:
        t0 = time.time()
        global_steer = (
            f"Light touch: ensure overall cohesion and consistent style in {final_language}."
            f" Do not add new content; do not expand length; keep section structure; fix minor formatting only."
        )
        try:
            refined2 = llm_refine_wiki(refined, global_steer)
            if refined2 and len(refined2) >= len(refined) * 0.7:  # evitar compresión agresiva
                refined = refined2
        except Exception:
            pass
        if len(refined) > global_refine_char_limit:
            refined = refined[:global_refine_char_limit] + "\n\n<!-- clipped final -->\n"
        logging.info(f"[WIKI_AGENT] global light refine done in {time.time() - t0:.2f}s (size={len(refined)} chars)")

    # Save final artifacts
    if save_intermediate:
        try:
            draft_path = os.path.join(dump_dir, f"{ts}_wiki_draft.md")
            with open(draft_path, 'w', encoding='utf-8') as f:
                f.write(draft or '')
            final_path = os.path.join(dump_dir, f"{ts}_wiki_final.md")
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(refined or '')
            coverage_path = os.path.join(dump_dir, f"{ts}_coverage.json")
            with open(coverage_path, 'w', encoding='utf-8') as f:
                json.dump({"coverage": coverage, "detail": detail}, f, ensure_ascii=False, indent=2)
            summary_path = os.path.join(dump_dir, f"{ts}_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "query": query,
                    "params": {
                        "top_k": top_k,
                        "score_threshold": score_threshold,
                        "section_max_tokens": section_max_tokens,
                        "refine_passes": refine_passes,
                        "coverage_target": coverage_target,
                        "strict_citations": strict_citations,
                    },
                    "sections": sections_meta,
                    "coverage": coverage,
                }, f, ensure_ascii=False, indent=2)
            logging.info(f"[WIKI_AGENT] artifacts saved to {dump_dir}")
        except Exception as e:
            logging.warning(f"[WIKI_AGENT] could not save final artifacts: {e}")

    # Optional condensation pass for coherence/concisión
    if condense_final:
        try:
            steer = (
                f"Reduce y unifica el siguiente documento para máxima coherencia en ~{condense_target_words} palabras. "
                f"Elimina repeticiones y detalles triviales; preserva la información clave y TODAS las citas (ruta, chunk N). "
                f"No agregues Tabla de Contenidos ni contenido nuevo. Devuelve Markdown en español."
            )
            condensed = llm_generate_markdown(steer + "\n\n" + refined)
            if condensed and len(condensed) >= max(int(len(refined) * 0.5), 1000):
                refined = condensed
                logging.info("[WIKI_AGENT] applied condensation pass")
        except Exception:
            pass

    # Compute and log execution metrics
    try:
        nsec = len(sections_meta)
        avg_citations = (sum(s.get("citations_count", 0) for s in sections_meta) / nsec) if nsec else 0.0
        avg_chars = (sum(s.get("chars", 0) for s in sections_meta) / nsec) if nsec else 0.0
        avg_elapsed = (sum(s.get("elapsed_sec", 0.0) for s in sections_meta) / nsec) if nsec else 0.0
        stub_sections = sum(1 for s in sections_meta if s.get("citations_count", 0) == 0)
        pct_stub = (stub_sections / nsec) if nsec else 0.0
        metrics = {
            "sections": nsec,
            "avg_citations_per_section": round(avg_citations, 3),
            "avg_chars_per_section": round(avg_chars, 1),
            "avg_elapsed_sec_per_section": round(avg_elapsed, 2),
            "stub_sections_pct": round(pct_stub, 3),
        }
        logging.info(f"[WIKI_AGENT] metrics: {metrics}")
    except Exception:
        metrics = {}

    return {
        "draft": draft,
        "final": refined,
        "coverage": coverage,
        "coverage_detail": detail,
        "strict_citations": strict_citations,
        "metrics": metrics,
    }
