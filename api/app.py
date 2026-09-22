# app.py

import os
import json
import time
import uuid
import uvicorn
import logging
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from api.llm_providers import create_llm_client, LlmError, LlmClient
from api.repo_analyzer import CHUNKING_STRATEGIES
from api.dependency_orchestrator import (
    AgentOrchestrator,
    build_dependency_graph,
    invert_dependency_graph,
    attach_dependents,
    compute_quality_scores,
    decompress_if_needed,
    hash_code_chunk,
)
from api.vector_db_manager import VectorDBManager
from api.enhanced_prompts import get_documentation_prompt, get_wiki_generation_prompt, get_wiki_agent_orchestration_prompt, get_wiki_refinement_prompt, get_wiki_structure_prompt, get_section_generation_prompt
from api.wiki_agent_orchestrator import generate_full_wiki
from api.logging_config import setup_logging
from starlette.requests import Request
from starlette.responses import Response
from datetime import datetime
import threading

# Inicializar la configuración de logging al inicio de la aplicación
setup_logging()

# Inicializar la aplicación FastAPI (debe ocurrir antes de usar decoradores @app.middleware)
app = FastAPI(
    title="Generador de Wiki de Repositorio",
    description="Una API para documentar un repositorio de código y generar una wiki.",
    version="1.0.0"
)

# Middleware manual para logging por endpoint
LOG_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
_endpoint_log_lock = threading.Lock()

@app.middleware("http")
async def per_request_file_logger(request: Request, call_next):
    start = time.time()
    try:
        body_bytes = await request.body()
    except Exception:
        body_bytes = b""
    response: Response | None = None
    error_text = None
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        raise
    finally:
        duration = time.time() - start
        endpoint_name = request.url.path.replace('/', '_').strip('_') or 'root'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        status = getattr(response, 'status_code', 'ERR')
        filename = f"{timestamp}_{endpoint_name}.log"
        log_path = os.path.join(LOG_DIR, filename)
        log_record = {
            "timestamp": timestamp,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
            "status": status,
            "duration_sec": round(duration, 4),
            "request_body_preview": body_bytes.decode('utf-8', errors='replace')[:2000],
            "error": error_text,
        }
        try:
            with _endpoint_log_lock:
                with open(log_path, 'w', encoding='utf-8') as lf:
                    json.dump(log_record, lf, ensure_ascii=False, indent=2)
        except Exception as fe:
            logging.warning(f"No se pudo escribir log de endpoint {endpoint_name}: {fe}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Cargar variables de entorno y configuraciones una vez al inicio
load_dotenv()
try:
    with open('api/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open('api/llm_models.json', 'r', encoding='utf-8') as f:
        llm_models_config = json.load(f)
    with open('api/embedding_config.json', 'r', encoding='utf-8') as f:
        embedding_config = json.load(f)
except FileNotFoundError as e:
    logging.error(f"Error: No se encontró un archivo de configuración. {e}")
    llm_models_config = {}
    config = {}
    embedding_config = {}

# Pydantic model para validar el cuerpo de la solicitud
class DocRequest(BaseModel):
    provider: str
    llm_model: str
    repo_path: str
    doc_type: str = "extensive_analysis"
    embedding_provider: str
    embedding_model: str
    # Nuevos campos para listas de inclusión y exclusión
    include_list: list[str] = []
    exclude_list: list[str] = []
    # Nuevo campo para número de reintentos
    doc_attempts: int = 3
    chunking: bool
    reprocess_changed: bool = False  # Si True, intenta saltar archivos sin cambios usando manifest
    # Nueva opción: si se especifica, SOLO se procesarán estos archivos (rutas relativas o absolutas).
    only_files: list[str] = []


# Extiende wikiRequest para pruebas avanzadas del agente
class WikiAgentRequest(BaseModel):
    provider: str
    llm_model: str
    repo_path: str
    embedding_provider: str
    embedding_model: str
    # Parámetros opcionales para pruebas avanzadas
    query: str = "Genera una documentación completa de wiki para todo el repositorio."
    k: int = 8
    score_threshold: float = 0.4
    max_context_tokens: int = 6000
    max_attempts: int = 3
    sleep_between_attempts: float = 1.0
    # Nuevos opcionales para cobertura/estrictez
    coverage_target: float = 0.9
    strict_citations: bool = True
    # Persistencia de artefactos
    save_intermediate: bool = True
    dump_dir: str | None = None
    # Idioma de salida de secciones y del final
    output_language: str = "en"  # idioma de las secciones
    final_language: str = "es"   # idioma del documento final tras refinamiento
    # Cobertura máxima
    full_coverage: bool = False
    # Guardrails configurables
    max_sections: int = 16
    section_char_limit: int = 3500
    refine_char_limit: int = 4500
    global_refine_char_limit: int = 80000
    # Diversificación y condensación
    use_mmr: bool = True
    mmr_lambda: float = 0.7
    condense_final: bool = False
    condense_target_words: int = 1500
    # Fallbacks entre proveedores para generación LLM
    enable_fallbacks: bool = False
    # Consultas inteligentes por sección
    smart_queries: bool = True

class wikiRequest(BaseModel):
    provider: str
    llm_model: str
    repo_path: str
    embedding_provider: str
    embedding_model: str

async def generate_wiki_agent(llm_client_embedding: LlmClient, llm_client: LlmClient, db_manager: VectorDBManager, query: str = "Genera una documentación completa de wiki para todo el repositorio.", k: int = 8, score_threshold: float = 0.4, max_context_tokens: int = 10000, max_attempts: int = 3, sleep_between_attempts: float = 1.0):
    """
    Agente iterativo para generar una wiki exhaustiva y coherente usando RAG y planificación LLM.
    """
    if db_manager.is_empty():
        logging.error("La base de datos está vacía. Por favor, documente el repositorio primero usando el endpoint /document.")
        raise HTTPException(status_code=404, detail="La base de datos está vacía. Por favor, documente el repositorio primero usando el endpoint /document.")

    # 1. Recuperar contexto general (top-k, score_threshold)
    query_embedding = llm_client_embedding.generate_embedding(query)
    results = db_manager.query(query_embedding, n_results=k*3)
    # Preferir distancias normalizadas devueltas por la consulta
    try:
        normalized_distances = results.get('norm_distances', [[]])[0] or []
    except Exception:
        normalized_distances = []

    # Estadísticas y normalización por percentiles (p10–p90) con clipping [0,1]
    try:
        raw_distances = results.get('distances', [[]])[0] or []
    except Exception:
        raw_distances = []

    if raw_distances and not normalized_distances:
        raw_min = float(min(raw_distances))
        raw_avg = float(sum(raw_distances) / len(raw_distances))
        raw_max = float(max(raw_distances))
        p10 = float(np.percentile(raw_distances, 10))
        p50 = float(np.percentile(raw_distances, 50))
        p90 = float(np.percentile(raw_distances, 90))
        logging.info(f"[AGENTE WIKI] Distancias RAW (min/avg/max): {raw_min:.4f}/{raw_avg:.4f}/{raw_max:.4f} (n={len(raw_distances)})")
        logging.info(f"[AGENTE WIKI] Percentiles RAW (p10/p50/p90): {p10:.4f}/{p50:.4f}/{p90:.4f}")
        denom = (p90 - p10) if p90 > p10 else (raw_max - raw_min) if raw_max > raw_min else 1.0
        normalized_distances = [max(0.0, min(1.0, (d - p10) / denom)) for d in raw_distances]
        if normalized_distances:
            norm_min = min(normalized_distances)
            norm_avg = sum(normalized_distances) / len(normalized_distances)
            norm_max = max(normalized_distances)
            logging.info(f"[AGENTE WIKI] Distancias NORMALIZADAS (min/avg/max): {norm_min:.4f}/{norm_avg:.4f}/{norm_max:.4f}")
    else:
        normalized_distances = []

    if not results or not results.get('ids') or len(results['ids'][0]) == 0:
        logging.error("No se encontraron documentos en la base de datos para generar la wiki.")
        raise HTTPException(status_code=404, detail="No se encontraron documentos en la base de datos para generar la wiki.")

    # Filtrar por score_threshold usando distancia NORMALIZADA (0..1, p10–p90)
    # Preferir distancias normalizadas del query si estA;n disponibles
    try:
        _pre_norms = results.get('norm_distances', [[]])[0] or []
    except Exception:
        _pre_norms = []
    try:
        _has_local_norms = bool(normalized_distances)
    except Exception:
        _has_local_norms = False
    if _pre_norms and not _has_local_norms:
        normalized_distances = _pre_norms

    all_docs_with_distances = []
    ids_len = len(results['ids'][0]) if results.get('ids') and results['ids'] and results['ids'][0] else 0
    for i in range(ids_len):
        norm_distance = normalized_distances[i] if i < len(normalized_distances) else 1.0
        if norm_distance <= score_threshold:
            doc_metadata = results['metadatas'][0][i]
            doc_text = results['documents'][0][i] if 'documents' in results and results['documents'][0] else None
            all_docs_with_distances.append((norm_distance, {"metadatas": [doc_metadata], "documents": [doc_text]}))

    all_docs_with_distances.sort(key=lambda x: x[0])
    sorted_docs = [doc[1] for doc in all_docs_with_distances][:k]

    if not sorted_docs:
        logging.error("[AGENTE WIKI] No quedó contexto tras filtrar por score_threshold (distancia normalizada). Ajusta el umbral o verifica embeddings.")
        raise HTTPException(
            status_code=422,
            detail="No hay contexto relevante tras filtrar por score_threshold (normalizado 0..1). Intenta aumentar k o el score_threshold, "
                   "o verifica que el modelo de embeddings coincida con el usado al documentar."
        )

    # 2. Generar estructura y to-do list con reintentos
    estructura = None
    todo_list = None
    for attempt in range(1, max_attempts+1):
        try:
            response = llm_client.generate_wiki_structure_and_todo(sorted_docs, query)
            estructura = response.get("estructura")
            todo_list = response.get("todo_list")
            if estructura and todo_list:
                break
        except Exception as e:
            logging.warning(f"[AGENTE WIKI] Error al generar estructura/to-do list (intento {attempt}): {e}")
            time.sleep(sleep_between_attempts)
    if not estructura or not todo_list:
        raise HTTPException(status_code=500, detail="No se pudo generar la estructura de la wiki tras varios intentos.")

    # 3. (Opcional) Mostrar plan de acción del agente
    try:
        plan_accion = llm_client.generate_wiki_markdown(get_wiki_agent_orchestration_prompt(estructura, todo_list, query))
        logging.info(f"[AGENTE WIKI] Plan de acción del agente:\n{plan_accion}")
    except Exception as e:
        logging.warning(f"[AGENTE WIKI] No se pudo generar el plan de acción: {e}")

    # 4. Generar cada sección iterativamente con reintentos
    secciones = []
    for section_title in todo_list:
        # Recuperar contexto relevante para la sección
        section_embedding = llm_client_embedding.generate_embedding(section_title)
        section_results = db_manager.query(section_embedding, n_results=k)
        # Distancias normalizadas precomputadas de la consulta
        try:
            sec_norm_pre = section_results.get('norm_distances', [[]])[0] or []
        except Exception:
            sec_norm_pre = []

        # Estadísticas y normalización por percentiles (p10–p90) con clipping [0,1] para la sección
        section_docs = []
        try:
            sec_raw_distances = section_results.get('distances', [[]])[0] or []
        except Exception:
            sec_raw_distances = []

        if sec_raw_distances and not sec_norm_pre:
            sec_raw_min = float(min(sec_raw_distances))
            sec_raw_avg = float(sum(sec_raw_distances) / len(sec_raw_distances))
            sec_raw_max = float(max(sec_raw_distances))
            p10 = float(np.percentile(sec_raw_distances, 10))
            p50 = float(np.percentile(sec_raw_distances, 50))
            p90 = float(np.percentile(sec_raw_distances, 90))
            logging.info(f"[AGENTE WIKI] Distancias RAW sección '{section_title}' (min/avg/max): "
                         f"{sec_raw_min:.4f}/{sec_raw_avg:.4f}/{sec_raw_max:.4f} (n={len(sec_raw_distances)})")
            logging.info(f"[AGENTE WIKI] Percentiles sección '{section_title}' (p10/p50/p90): "
                         f"{p10:.4f}/{p50:.4f}/{p90:.4f}")
            sec_denom = (p90 - p10) if p90 > p10 else (sec_raw_max - sec_raw_min) if sec_raw_max > sec_raw_min else 1.0
            sec_norm_distances = [max(0.0, min(1.0, (d - p10) / sec_denom)) for d in sec_raw_distances]
            if sec_norm_distances:
                sec_norm_min = min(sec_norm_distances)
                sec_norm_avg = sum(sec_norm_distances) / len(sec_norm_distances)
                sec_norm_max = max(sec_norm_distances)
                logging.info(f"[AGENTE WIKI] Distancias NORMALIZADAS sección '{section_title}' "
                             f"(min/avg/max): {sec_norm_min:.4f}/{sec_norm_avg:.4f}/{sec_norm_max:.4f}")
        else:
            sec_norm_distances = []

        # Preferir norm_distances del query si aAUn no estA�n calculadas
        try:
            _pre_norms = section_results.get('norm_distances', [[]])[0] or []
        except Exception:
            _pre_norms = []
        try:
            _has_local_norms = bool(sec_norm_distances)
        except Exception:
            _has_local_norms = False
        # Forzar uso de distancias normalizadas del query
        if 'sec_norm_pre' in locals() and sec_norm_pre:
            sec_norm_distances = sec_norm_pre

        ids_len_sec = len(section_results['ids'][0]) if section_results.get('ids') and section_results['ids'] and section_results['ids'][0] else 0
        for i in range(ids_len_sec):
            sec_norm = sec_norm_distances[i] if i < len(sec_norm_distances) else 1.0
            if sec_norm <= score_threshold:
                doc_metadata = section_results['metadatas'][0][i]
                doc_text = section_results['documents'][0][i] if 'documents' in section_results and section_results['documents'][0] else None
                section_docs.append({"metadatas": [doc_metadata], "documents": [doc_text]})

        if not section_docs:
            logging.warning(f"[AGENTE WIKI] Sin contexto para la sección '{section_title}' con el threshold actual. "
                            "Aplicando fallback sin umbral.")
            # Fallback: usa top-k sin filtrar
            section_docs = []
            for i in range(len(section_results['ids'][0])):
                doc_metadata = section_results['metadatas'][0][i]
                doc_text = section_results['documents'][0][i] if 'documents' in section_results and \
                                                                 section_results['documents'][0] else None
                section_docs.append({"metadatas": [doc_metadata], "documents": [doc_text]})
            # Si aun así no hay nada, deja constancia y sigue
            if not section_docs:
                logging.warning(f"[AGENTE WIKI] No se encontraron documentos ni con fallback para '{section_title}'.")

        # Limitar contexto por tokens
        section_context = get_wiki_generation_prompt(section_title, section_docs, max_context_tokens // max(1, len(todo_list)))
        section_text = None
        for attempt in range(1, max_attempts+1):
            try:
                section_text = llm_client.generate_section(section_context, section_title)
                if section_text and section_text.strip():
                    break
            except Exception as e:
                logging.warning(f"[AGENTE WIKI] Error al generar sección '{section_title}' (intento {attempt}): {e}")
                time.sleep(sleep_between_attempts)
        if not section_text:
            section_text = f"## {section_title}\n_No se pudo generar esta sección tras varios intentos._"
        secciones.append(section_text)

    # 5. Ensamblar y refinar la wiki globalmente
    wiki_borrador = "\n\n".join(secciones)
    wiki_final = None
    for attempt in range(1, max_attempts+1):
        try:
            wiki_final = llm_client.refine_wiki(wiki_borrador, query)
            if wiki_final and wiki_final.strip():
                break
        except Exception as e:
            logging.warning(f"[AGENTE WIKI] Error al refinar la wiki (intento {attempt}): {e}")
            time.sleep(sleep_between_attempts)
    if not wiki_final:
        wiki_final = wiki_borrador

    return {"wiki": wiki_final, "estructura": estructura, "todo_list": todo_list}


@app.post("/wiki_agent")
async def generate_wiki_agent_endpoint(request: WikiAgentRequest = Body(...)):
    """
    Endpoint experimental/aislado para generar una wiki exhaustiva usando el agente iterativo RAG+LLM.
    Permite ajustar parámetros avanzados en el body del request, pero tiene valores por defecto.
    """
    logging.info("[WIKI_AGENT] Request received for wiki generation (isolated agent mode)")
    llm_client = create_llm_client(request.provider, request.llm_model)
    llm_client_embedding = create_llm_client(request.embedding_provider, request.embedding_model)

    if not llm_client:
        raise HTTPException(status_code=500, detail="Error: No se pudo inicializar el cliente LLM.")

    # Normalizar y validar score_threshold a rango [0,1]
    try:
        _thr = float(request.score_threshold)
    except Exception:
        _thr = 0.4
    if _thr < 0.0 or _thr > 1.0:
        logging.info(f"[WIKI_AGENT] score_threshold fuera de rango ({request.score_threshold}); normalizando a [0,1]")
    score_threshold = max(0.0, min(1.0, _thr))

    repo_name = os.path.basename(os.path.normpath(request.repo_path))
    collection_name = f"repo_documentation_{repo_name.replace('-', '_').lower()}"
    logging.info(f"[WIKI_AGENT] collection: {collection_name} | repo_path: {request.repo_path}")
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=f"./{collection_name}")

    # Fallback wrapper para generación LLM (markdown/structure/refine)
    def _default_model_for(prov: str) -> str:
        prov = prov.lower()
        cfg = llm_models_config.get(prov) or {}
        if isinstance(cfg, dict) and cfg:
            try:
                return next(iter(cfg.keys()))
            except Exception:
                pass
        return {
            'openai': 'gpt-4o',
            'google': 'gemini-1.0-pro',
            'groq': 'llama3-8b-8192',
            'ollama': 'llama3:8b',
        }.get(prov, request.llm_model)

    class _FallbackWrapper:
        def __init__(self, clients: list):
            self.clients = clients
        def generate_wiki_structure_and_todo(self, context, query):
            last = None
            for c in self.clients:
                try:
                    return c.generate_wiki_structure_and_todo(context, query)
                except Exception as e:
                    last = e
                    logging.warning(f"[FALLBACK] structure failed on {getattr(c,'provider','?')}: {e}")
            raise last or Exception("All providers failed (structure)")
        def generate_wiki_markdown(self, prompt: str) -> str:
            last = None
            for c in self.clients:
                try:
                    return c.generate_wiki_markdown(prompt)
                except Exception as e:
                    last = e
                    logging.warning(f"[FALLBACK] markdown failed on {getattr(c,'provider','?')}: {e}")
            raise last or Exception("All providers failed (markdown)")
        def refine_wiki(self, md: str, q: str) -> str:
            last = None
            for c in self.clients:
                try:
                    return c.refine_wiki(md, q)
                except Exception as e:
                    last = e
                    logging.warning(f"[FALLBACK] refine failed on {getattr(c,'provider','?')}: {e}")
            raise last or Exception("All providers failed (refine)")

    if request.enable_fallbacks:
        order = ['openai', 'google', 'groq', 'ollama']
        try:
            start = order.index(request.provider.lower())
        except Exception:
            start = 0
        chain = []
        # first client is the chosen one (already created)
        chain.append(llm_client)
        for prov in order[start+1:]:
            try:
                model = _default_model_for(prov)
                alt = create_llm_client(prov, model)
                chain.append(alt)
            except Exception as e:
                logging.info(f"[FALLBACK] skipping {prov}: {e}")
        llm_fallback = _FallbackWrapper(chain)
        llm_for_orchestrator = llm_fallback
    else:
        llm_for_orchestrator = llm_client

    # Embedding/IDs helpers for orchestrator (MMR)
    def _embed_text_fn(text: str) -> list[float]:
        try:
            return llm_client_embedding.generate_embedding(text)
        except Exception:
            return []
    def _get_embeddings_fn(ids: list[str]) -> dict[str, list[float]]:
        try:
            data = db_manager.get_by_ids(ids, include_embeddings=True) or {}
            out: dict[str, list[float]] = {}
            id_list = data.get('ids') or []
            emb_list = data.get('embeddings') or []
            for i, _id in enumerate(id_list):
                try:
                    out[str(_id)] = emb_list[i]
                except Exception:
                    continue
            return out
        except Exception:
            return {}

    # Helper: función de búsqueda adaptada a la firma del orquestador, con normalización p10–p90
    # Simple cache for embeddings por consulta
    _embed_cache: dict[str, list[float]] = {}

    def search_fn(q: str, top_k: int, score_threshold: float):
        t0 = time.time()
        try:
            if q in _embed_cache:
                emb = _embed_cache[q]
            else:
                emb = llm_client_embedding.generate_embedding(q)
                if isinstance(emb, list):
                    if len(_embed_cache) > 256:
                        _embed_cache.clear()
                    _embed_cache[q] = emb
            raw = db_manager.query(emb, n_results=max(top_k, 1)) or {}
            # Filtrado por threshold usando distancias normalizadas si estA;n presentes
            docs: List[Dict[str, Any]] = []
            try:
                nd = raw.get('norm_distances', [[]])[0] or []
                distances = raw.get('distances', [[]])[0] or []
                ids = raw.get('ids', [[]])[0] or []
                mds = raw.get('metadatas', [[]])[0] or []
                docs_txt = raw.get('documents', [[]])[0] or []
                if nd:
                    norm = nd
                else:
                    if distances:
                        p10 = float(np.percentile(distances, 10))
                        p90 = float(np.percentile(distances, 90))
                        dmin, dmax = float(min(distances)), float(max(distances))
                        denom = (p90 - p10) if p90 > p10 else (dmax - dmin) if dmax > dmin else 1.0
                        norm = [max(0.0, min(1.0, (float(d) - p10) / denom)) for d in distances]
                    else:
                        norm = [0.0] * len(ids)
                for i in range(len(ids)):
                    if norm[i] <= score_threshold:
                        docs.append({"ids": [ids[i]], "metadatas": [mds[i]], "documents": [docs_txt[i]]})
            except Exception:
                pass
            dt = time.time() - t0
            logging.info(f"[WIKI_AGENT] search_fn: q='{(q[:80] + '...') if len(q)>80 else q}' -> {len(docs)} docs in {dt:.2f}s (top_k={top_k}, thr={score_threshold})")
            return docs
        except Exception:
            logging.exception("[WIKI_AGENT] search_fn error")
            return []

    # Helper: listar todos los chunks disponibles desde la colección
    def list_chunks_fn():
        data = db_manager.get_all(include_embeddings=False)
        out = []
        try:
            ids = data.get('ids') or []
            mds = data.get('metadatas') or []
            for i in range(len(ids)):
                md = mds[i] or {}
                fp = md.get('file_path')
                # per-chunk path: enriched_chunk json
                enriched = md.get('enriched_chunk')
                if enriched:
                    try:
                        ec = json.loads(enriched)
                    except Exception:
                        ec = {}
                    out.append({
                        'file_path': fp,
                        'chunk_index': ec.get('chunk_index', md.get('chunk_index')),
                        'chunk_type': ec.get('chunk_type', md.get('chunk_type')),
                        'summary': ec.get('summary', ''),
                        'original_code': ec.get('original_code', ''),
                    })
                else:
                    # file-level doc; try to expand 'enriched_chunks'
                    ec_list = md.get('enriched_chunks') or md.get('chunks')
                    try:
                        arr = json.loads(ec_list) if isinstance(ec_list, str) else []
                    except Exception:
                        arr = []
                    for ch in arr or []:
                        out.append({
                            'file_path': fp,
                            'chunk_index': ch.get('chunk_index') or ch.get('chunk_id'),
                            'chunk_type': ch.get('chunk_type'),
                            'summary': ch.get('summary') or '',
                            'original_code': ch.get('original_code') or '',
                        })
        except Exception:
            pass
        return out

    # Detección rápida de hints del repo para perfilar la consulta
    def _detect_repo_hints() -> set[str]:
        hints: set[str] = set()
        try:
            data = db_manager.get_all(include_embeddings=False, limit=2000) or {}
            mds = data.get('metadatas') or []
            for md in mds:
                fp = (md or {}).get('file_path') or ''
                fp_l = fp.lower()
                if any(x in fp_l for x in ['spark', 'build.sbt']):
                    hints.add('spark')
                if fp_l.endswith('.scala'):
                    hints.add('scala')
                if fp_l.endswith('.py'):
                    hints.add('python')
                if 'fastapi' in fp_l or 'app.py' in fp_l:
                    hints.add('fastapi')
                if fp_l.endswith('.java'):
                    hints.add('java')
        except Exception:
            pass
        return hints

    repo_hints = _detect_repo_hints()
    logging.info(f"[WIKI_AGENT] repo_hints detected: {sorted(list(repo_hints))}")

    # Construir una consulta efectiva en inglés, más amplia y técnica
    base_query = (
        "Generate an exhaustive, technical wiki focused on internal functionality, architecture, data flows, modules, "
        "classes, functions, configuration, and dependencies. Use only grounded claims with precise citations to (file_path, chunk N). "
        "Avoid audience-specific tone and avoid assumptions about web frameworks unless evidence exists."
    )
    if 'spark' in repo_hints or 'scala' in repo_hints:
        base_query += " Emphasize Apache Spark jobs, datasets, transformations, actions, and pipelines, and Scala/ScalaTest where applicable."
    effective_query = base_query

    # Estructura inicial robusta con reintentos + guardado JSON
    try:
        logging.info("[WIKI_AGENT] generating initial structure from query...")
        t0 = time.time()
        estructura, todo_list = [], []
        method_used = None
        raw_texts: list[str] = []

        # Carpeta de dumps
        dump_dir = request.dump_dir or os.path.join(os.getcwd(), 'Pruebas')
        if request.save_intermediate:
            os.makedirs(dump_dir, exist_ok=True)

        # 1) Método estructurado con reintentos
        for attempt in range(1, max(1, request.max_attempts) + 1):
            try:
                q_emb = llm_client_embedding.generate_embedding(request.query)
                raw = db_manager.query(q_emb, n_results=max(request.k * 2, 8)) or {}
                pairs = []
                ids = raw.get('ids', [[]])[0] or []
                mds = raw.get('metadatas', [[]])[0] or []
                docs_txt = raw.get('documents', [[]])[0] or []
                nds = raw.get('norm_distances', [[]])[0] or []
                dists = nds if nds else (raw.get('distances', [[]])[0] or [])
                for i in range(len(ids)):
                    pairs.append((dists[i] if i < len(dists) else 0.0, {"metadatas": [mds[i]], "documents": [docs_txt[i]]}))
                pairs.sort(key=lambda x: x[0])
                context_docs = [p[1] for p in pairs[: request.k]]
                structured = (llm_for_orchestrator if request.enable_fallbacks else llm_client).generate_wiki_structure_and_todo(context_docs, effective_query)
                estructura = structured.get("estructura") or structured.get("structure") or []
                todo_list = structured.get("todo_list") or estructura or []
                method_used = "structured"
                logging.info(f"[WIKI_AGENT] structure via structured method (attempt {attempt}) -> sections={len(todo_list)}")
                break
            except Exception as e1:
                logging.info(f"[WIKI_AGENT] structured method failed (attempt {attempt}): {type(e1).__name__}: {e1}")
                time.sleep(max(0.0, request.sleep_between_attempts))

        # 2) Fallback: prompt + extracción JSON con reintentos
        if not todo_list:
            for attempt in range(1, max(1, request.max_attempts) + 1):
                try:
                    structure_prompt = get_wiki_structure_prompt([], effective_query)
                    struct_raw = (llm_for_orchestrator if request.enable_fallbacks else llm_client).generate_wiki_markdown(structure_prompt)
                    raw_texts.append(struct_raw or "")
                    def _try_extract_json(text: str):
                        import re as _re, json as _json
                        if not text:
                            return None
                        m = _re.search(r"```(?:json)?\\s*([\\s\\S]*?)\\s*```", text, _re.IGNORECASE)
                        candidates = []
                        if m:
                            candidates.append(m.group(1).strip())
                        candidates.append(text.strip())
                        objs = _re.findall(r'([\\{\\[][\\s\\S]*?[\\}\\]])', text)
                        candidates.extend(objs[:2])
                        for c in candidates:
                            try:
                                return _json.loads(c)
                            except Exception:
                                continue
                        return None
                    struct = _try_extract_json(struct_raw)
                    if struct:
                        estructura = struct.get('estructura') or struct.get('structure') or []
                        todo_list = struct.get('todo_list') or estructura or []
                        method_used = "json_extraction"
                        logging.info(f"[WIKI_AGENT] structure via JSON extraction (attempt {attempt}) -> sections={len(todo_list)}")
                        break
                    else:
                        raise ValueError("LLM returned non-JSON structure text")
                except Exception as e2:
                    logging.info(f"[WIKI_AGENT] json extraction failed (attempt {attempt}): {type(e2).__name__}: {e2}")
                    time.sleep(max(0.0, request.sleep_between_attempts))

        # Guardar estructura en JSON si procede
        if request.save_intermediate:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out = {
                'query': effective_query,
                'method_used': method_used,
                'estructura': estructura,
                'todo_list': todo_list,
                'duration_sec': round(time.time()-t0, 3),
            }
            try:
                with open(os.path.join(dump_dir, f"{ts}_wiki_agent_structure.json"), 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                if raw_texts:
                    with open(os.path.join(dump_dir, f"{ts}_wiki_agent_structure_raw.txt"), 'w', encoding='utf-8') as f:
                        f.write("\n\n-----\n\n".join(raw_texts))
                logging.info(f"[WIKI_AGENT] structure artifacts saved to {dump_dir}")
            except Exception as fe:
                logging.warning(f"[WIKI_AGENT] could not save structure artifacts: {fe}")

        logging.info(f"[WIKI_AGENT] structure ready (took {time.time()-t0:.2f}s) | method={method_used} | sections={len(todo_list)}")
        if not todo_list:
            raise RuntimeError("Structure generation failed after retries")
    except Exception as e:
        logging.info(f"[WIKI_AGENT] structure generation failed; using fallback sections | reason={type(e).__name__}: {e}")
        estructura = []
        todo_list = [
            'Resumen del repositorio',
            'Arquitectura y módulos',
            'Servicios y controladores',
            'Configuración y entorno',
            'Riesgos y recomendaciones',
        ]

    logging.info("[WIKI_AGENT] invoking orchestrator...")
    t0 = time.time()
    result = generate_full_wiki(
        query=effective_query,
        estructura=estructura,
        todo_list=todo_list,
        search_fn=search_fn,
        list_chunks_fn=list_chunks_fn,
        llm_generate_markdown=lambda p: llm_for_orchestrator.generate_wiki_markdown(p),
        llm_refine_wiki=lambda md, q: llm_for_orchestrator.refine_wiki(md, q),
        params={
            'top_k': request.k,
            'score_threshold': score_threshold,
            'section_max_tokens': request.max_context_tokens,
            'refine_passes': max(1, request.max_attempts - 1),
            'coverage_target': request.coverage_target,
            'strict_citations': request.strict_citations,
            'save_intermediate': request.save_intermediate,
            'dump_dir': request.dump_dir or os.path.join(os.getcwd(), 'Pruebas'),
            'output_language': request.output_language,
            'final_language': request.final_language,
            'full_coverage': request.full_coverage,
            'max_sections': request.max_sections,
            'section_char_limit': request.section_char_limit,
            'refine_char_limit': request.refine_char_limit,
            'global_refine_char_limit': request.global_refine_char_limit,
            # Diversificación y condensación
            'use_mmr': request.use_mmr,
            'mmr_lambda': request.mmr_lambda,
            'embed_text_fn': _embed_text_fn,
            'get_embeddings_fn': _get_embeddings_fn,
            'condense_final': request.condense_final,
            'condense_target_words': request.condense_target_words,
            'smart_queries': request.smart_queries,
        }
    )
    logging.info(f"[WIKI_AGENT] orchestrator finished in {time.time()-t0:.2f}s | coverage={result.get('coverage'):.3f}")

    return {
        'wiki': result['final'],
        'coverage': result['coverage'],
        'coverage_detail': result['coverage_detail'],
        'estructura': estructura,
        'todo_list': todo_list,
    }

async def generate_repo_wiki(llm_client_embedding: LlmClient, llm_client: LlmClient, db_manager: VectorDBManager):
    """
    Genera un documento de wiki completo basándose en toda la documentación almacenada.
    """
    # Se obtienen los valores de las variables de configuración
    max_context_tokens = llm_models_config.get(llm_client.provider, {}).get(llm_client.model_name, {}).get("max_context_tokens", 10000)
    db_results_count = config.get("db_results_count", 100)

    # Pregunta amplia para obtener toda la documentación del repositorio
    query = "Genera una documentación completa de wiki para todo el repositorio."
    
    try:
        if db_manager.is_empty():
            logging.error("La base de datos está vacía. Por favor, documente el repositorio primero usando el endpoint /document.")
            raise HTTPException(status_code=404, detail="La base de datos está vacía. Por favor, documente el repositorio primero usando el endpoint /document.")

        # 1. Generar el embedding de la pregunta
        query_embedding = llm_client_embedding.generate_embedding(query)

        # 2. Consultar la base de datos vectorial para obtener todos los documentos
        # Se usa el valor de la configuración para el número de resultados
        results = db_manager.query(query_embedding, n_results=db_results_count)
        logging.info(f"[WIKI] Resultados crudos de la base de datos: {json.dumps(results, ensure_ascii=False)[:2000]} ...")

        if not results or not results.get('ids') or len(results['ids'][0]) == 0:
            logging.error("No se encontraron documentos en la base de datos para generar la wiki.")
            raise HTTPException(status_code=404, detail="No se encontraron documentos en la base de datos para generar la wiki.")

        # 3. Combinar documentos y distancias para ordenar por relevancia (usar normalizadas si existen)
        all_docs_with_distances = []
        nds = (results.get('norm_distances', [[]]) or [[]])[0]
        use_nd = bool(nds)
        for i in range(len(results['ids'][0])):
            distance = (nds[i] if use_nd and i < len(nds) else results['distances'][0][i])
            doc_metadata = results['metadatas'][0][i]
            # logging.info(f"[WIKI] Metadata documento {i}: {json.dumps(doc_metadata, ensure_ascii=False)}")
            all_docs_with_distances.append((distance, {"metadatas": [doc_metadata], "documents": [results['documents'][0][i]]}))

        # 4. Ordenar los documentos por distancia (de menor a mayor) para priorizar los más relevantes
        all_docs_with_distances.sort(key=lambda x: x[0])
        # logging.info(f"[WIKI] all_docs_with_distances: {all_docs_with_distances} ...")
        
        # 5. Extraer solo los documentos ordenados para pasarlos al prompt
        sorted_docs = [doc[1] for doc in all_docs_with_distances]

        # 6. Utilizar el prompt para la generación de la wiki con los documentos ordenados.
        # Se pasa el límite de tokens configurado al prompt.
        wiki_prompt = get_wiki_generation_prompt(query, sorted_docs, max_context_tokens)
        
        # 7. Enviar el prompt al LLM para generar la wiki final (Markdown)
        logging.info(f"Generando wiki del repositorio con {len(sorted_docs)} documentos ordenados...")
        # logging.info(f"Haciendo uso del siguiente prompt: \n{wiki_prompt}, \n\nademás de su consulta:\n{query}\n\n\nY el contexto de: {sorted_docs}")
        
        try:
            wiki_markdown = llm_client.generate_wiki_markdown(wiki_prompt)
            if wiki_markdown and wiki_markdown.strip():
                logging.info("Wiki generada con éxito.")
                return {"wiki": wiki_markdown}
            else:
                logging.error("Fallo al generar la wiki: respuesta vacía.")
                raise HTTPException(status_code=500, detail="Fallo al generar la wiki: respuesta vacía.")
        except Exception as e:
            logging.error(f"Fallo al generar la wiki (Markdown): {e}")
            raise HTTPException(status_code=500, detail=f"Fallo al generar la wiki (Markdown): {e}")
            
    except LlmError as e:
        logging.exception("Error al generar la wiki: %s", e)
        raise HTTPException(status_code=500, detail=f"Error al generar la wiki: {e}")
    except Exception as e:
        logging.exception("Ocurrió un error inesperado al generar la wiki: %s", e)
        raise HTTPException(status_code=500, detail=f"Ocurrió un error inesperado al generar la wiki: {e}")

@app.post("/document")
async def document_repo_endpoint(request: DocRequest):
    """
    Endpoint para documentar el repositorio y almacenarlo en la base de datos vectorial.
    """
    logging.info("Solicitud recibida para documentar el repositorio.")
    llm_client = create_llm_client(request.provider, request.llm_model)
    llm_client_embedding = create_llm_client(request.embedding_provider, request.embedding_model)

    if not llm_client:
        raise HTTPException(status_code=500, detail="Error: No se pudo inicializar el cliente LLM.")

    repo_name = os.path.basename(os.path.normpath(request.repo_path))
    collection_name = f"repo_documentation_{repo_name.replace('-', '_').lower()}"
    logging.info(f"[DOCUMENT] Nombre de colección: {collection_name} (repo_path: {request.repo_path})")
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=f"./{collection_name}")
    
    all_documents_new: list[dict] = []  # nuevos/reprocesados
    all_documents_reused: list[dict] = []  # reutilizados (skip)
    reuse_potential = 0  # chunks candidatos a reutilizar
    reuse_found = 0       # chunks efectivamente reutilizados
    reuse_mismatch_files: list[str] = []
    orchestrator = AgentOrchestrator(
        llm_client=llm_client,
        llm_client_embedding=llm_client_embedding,
        max_attempts=2,
        per_chunk_embeddings=True,  # Activamos embeddings por chunk
        compress_large_fields=True,
        truncation_limit=3500,
        compression_min_ratio=0.9,
    )
    logging.info("[DOCUMENT] Orchestrator configurado con per_chunk_embeddings=True truncation_limit=3500 compression_min_ratio=0.9")
    
    try:
        # Se normalizan las rutas para una comparación consistente
        exclude_paths = [os.path.normpath(os.path.join(request.repo_path, p)) for p in request.exclude_list]
        include_paths = [os.path.normpath(os.path.join(request.repo_path, p)) for p in request.include_list]

        # Manifest (hashes por archivo)
        manifest_path = os.path.join(request.repo_path, ".deepwiki_manifest.json")
        try:
            with open(manifest_path, 'r', encoding='utf-8') as mf:
                manifest = json.load(mf)
        except Exception:
            manifest = {"files": {}}
        files_section = manifest.setdefault("files", {})

        def _chunk_id(fp: str, idx: int) -> str:
            import hashlib
            return hashlib.sha256(f"{fp}::{idx}".encode('utf-8')).hexdigest()

        # --- NUEVO: rama para procesar sólo archivos específicos ---
        def _normalize_target(p: str) -> str:
            if not os.path.isabs(p):
                p = os.path.join(request.repo_path, p)
            return os.path.normpath(p)

        only_mode = bool(request.only_files)
        target_files: list[str] = []
        if only_mode:
            for p in request.only_files:
                fp = _normalize_target(p)
                if not fp.startswith(os.path.normpath(request.repo_path)):
                    logging.warning(f"[ONLY_FILES] Ignorando fuera de repo: {fp}")
                    continue
                if not os.path.isfile(fp):
                    logging.warning(f"[ONLY_FILES] No existe archivo: {fp}")
                    continue
                target_files.append(fp)
            if not target_files:
                raise HTTPException(status_code=400, detail="Ningún archivo válido en only_files")

        def _iter_repo_files():
            dir_blocklist = {
                '.git','node_modules','vendor','dist','build','target','.venv','venv','env','__pycache__',
                '.mypy_cache','.pytest_cache','.next','.cache'
            }
            for dirpath, dirnames, filenames in os.walk(request.repo_path):
                # prune directories
                if '.git' in dirnames:
                    dirnames.remove('.git')
                dirnames[:] = [
                    d for d in dirnames
                    if d not in dir_blocklist and os.path.normpath(os.path.join(dirpath, d)) not in exclude_paths
                ]
                for filename in filenames:
                    yield os.path.normpath(os.path.join(dirpath, filename))

        logging.warning(f"[DOCUMENT] Variable only_mode: {only_mode}")
        files_to_process = target_files if only_mode else list(_iter_repo_files())

        for file_path in files_to_process:
            filename = os.path.basename(file_path)
            # skip common lockfiles and artifacts
            lock_names = {
                'package-lock.json','yarn.lock','pnpm-lock.yaml','poetry.lock','Pipfile.lock','Cargo.lock','Gemfile.lock'
            }
            if filename in lock_names and not only_mode:
                continue
            if (file_path in exclude_paths) or (include_paths and file_path not in include_paths and not only_mode):
                continue
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext in config.get('extensiones_a_ignorar', []):
                if not only_mode:  # en only_mode respetamos de todos modos para evitar ruido innecesario
                    continue
            if ext not in config.get('extensiones_a_procesar', []) and not only_mode:
                continue
            # En only_mode si la extensión no está listada pero queremos forzar, la procesamos igual

            logging.info(f"[AGENTIC] Archivo candidato: {file_path}")
            logging.info(f"[AGENTIC] Extensión de archivo candidato: {ext}")
            chunk_function = CHUNKING_STRATEGIES.get(ext) if ext in CHUNKING_STRATEGIES else None
            if request.chunking and chunk_function:
                chunks = chunk_function(file_path)
                if chunks is None:
                    logging.warning(f"[AGENTIC] Estrategia de chunking devolvió None para {file_path}; usando archivo completo")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            fallback_content = f.read()
                    except Exception:
                        fallback_content = ''
                    chunks = [(fallback_content, 'full_file')]
                if len(chunks) > 1 and any(ct == 'full_file' for _, ct in chunks):
                    orig_len = len(chunks)
                    chunks = [(c, t) for c, t in chunks if t != 'full_file'] or chunks
                    if len(chunks) != orig_len:
                        logging.debug(f"[AGENTIC] Removido chunk 'full_file' redundante en {file_path}")
                logging.info(f"[AGENTIC] {len(chunks)} chunks generados")
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception:
                    content = ''
                chunks = [(content, 'complete')]
                logging.info("[AGENTIC] Archivo completo como único chunk (sin chunking o sin estrategia)")

            # Filtrado de vacíos/duplicados
            if chunks:
                filtered = []
                seen = set()
                for text, ctype in chunks:
                    if not text or not text.strip():
                        continue
                    key = (text.strip(), ctype)
                    if key in seen:
                        continue
                    seen.add(key)
                    filtered.append((text, ctype))
                if len(filtered) != len(chunks):
                    logging.debug(f"[AGENTIC] Filtrados {len(chunks)-len(filtered)} chunks vacíos/duplicados en {file_path}")
                chunks = filtered or chunks

            # Hashes y reuse selectivo (solo si no estamos forzando formato distinto)
            chunk_hashes = [hash_code_chunk(c[0]) for c in chunks]
            file_hash = hash_code_chunk("::".join(chunk_hashes))
            existing_meta = files_section.get(file_path)
            can_skip = False
            if request.reprocess_changed and existing_meta and existing_meta.get("file_hash") == file_hash:
                expected_ids = [_chunk_id(file_path, i) for i in range(len(chunks))]
                reuse_potential += len(expected_ids)
                existing = db_manager.get_by_ids(expected_ids, include_embeddings=True)
                ids_returned = existing.get('ids') if existing else []
                if ids_returned and len(ids_returned) == len(expected_ids):
                    embeddings_raw = existing.get('embeddings') if existing else None
                    # Normalizar embeddings a lista indexable segura
                    if embeddings_raw is None:
                        embeddings_seq = []
                    elif isinstance(embeddings_raw, list):
                        embeddings_seq = embeddings_raw
                    else:
                        try:
                            import numpy as _np
                            if isinstance(embeddings_raw, _np.ndarray):
                                embeddings_seq = list(embeddings_raw)
                            else:
                                embeddings_seq = list(embeddings_raw)
                        except Exception:
                            embeddings_seq = []
                    for i, cid in enumerate(ids_returned):
                        md = existing.get('metadatas', [])[i]
                        emb = embeddings_seq[i] if i < len(embeddings_seq) else []
                        all_documents_reused.append({"id": cid, "embedding": emb, "metadata": md})
                    reuse_found += len(ids_returned)
                    logging.info(f"[SELECTIVE] Skip sin cambios: {file_path} (chunks={len(ids_returned)})")
                    can_skip = True
                elif ids_returned:
                    reuse_mismatch_files.append(file_path)
                    logging.info(f"[SELECTIVE] Inconsistencia reuse parcial en {file_path} (esperados={len(expected_ids)} obtenidos={len(ids_returned)}), reprocesando")
            if can_skip:
                continue

            try:
                processed = orchestrator.process_file(file_path, chunks)
                if isinstance(processed, dict) and "_batched" in processed:
                    all_documents_new.extend(processed["_batched"])
                else:
                    all_documents_new.append(processed)
                files_section[file_path] = {
                    "file_hash": file_hash,
                    "chunk_hashes": chunk_hashes,
                    "updated_at": time.time(),
                    "embedding_mode": "chunk",
                    "chunk_count": len(chunks),
                }
                logging.info(f"[DOCUMENT] Procesado {file_path} (chunks={len(chunks)})")
            except Exception as e:
                logging.error(f"[AGENTIC] Error procesando {file_path}: {e}")

        combined_docs = all_documents_new + all_documents_reused
        if combined_docs:
            # Post-pass: dependencia inversa + quality scores
            try:
                logging.info("[DOCUMENT] Post-pass: construyendo grafo de dependencias")
                graph = build_dependency_graph(combined_docs)
                logging.info(f"[DOCUMENT] Grafo: nodos={len(graph)}")
                inverted = invert_dependency_graph(graph)
                logging.info(f"[DOCUMENT] Grafo invertido: nodos={len(inverted)}")
                attach_dependents(combined_docs, inverted)
                logging.info("[DOCUMENT] Dependents añadidos")
                compute_quality_scores(combined_docs)
                logging.info("[DOCUMENT] Quality scores calculados")
            except Exception as e:
                logging.warning(f"Post-pass enrichment error: {e}")
            if all_documents_new:
                db_manager.add_documents(all_documents_new)
                logging.info(f"Nuevos documentos agregados: {len(all_documents_new)} | Reusados: {len(all_documents_reused)} | Total combinados: {len(combined_docs)}")
            else:
                logging.info(f"Sin nuevos documentos. Reusados: {len(all_documents_reused)}")
            # Persistir manifest
            try:
                with open(manifest_path, 'w', encoding='utf-8') as mf:
                    json.dump(manifest, mf, ensure_ascii=False, indent=2)
                logging.info(f"[SELECTIVE] Manifest actualizado {manifest_path}")
            except Exception as me:
                logging.warning(f"[SELECTIVE] No se pudo escribir manifest: {me}")
            # Comprobación inmediata de si la colección está vacía tras agregar
            if db_manager.is_empty():
                logging.error("[DOCUMENT] ¡La colección sigue vacía tras agregar documentos!")
            else:
                logging.info("[DOCUMENT] La colección contiene documentos tras agregar.")
        else:
            logging.warning("[DOCUMENT] No se agregaron documentos a la base de datos.")
        total_processed = len(all_documents_new) + len(all_documents_reused)
        savings_pct = (reuse_found / (reuse_found + len(all_documents_new)) * 100.0) if (reuse_found + len(all_documents_new)) > 0 else 0.0
        return {
            "message": "Repositorio documentado y almacenado exitosamente.",
            "added": len(all_documents_new),
            "reused": len(all_documents_reused),
            "reuse_potential_chunks": reuse_potential,
            "reuse_realized_chunks": reuse_found,
            "reuse_savings_pct": round(savings_pct, 2),
            "reuse_mismatch_files": reuse_mismatch_files,
            "total_documents_postpass": total_processed,
        }
    except Exception as e:
        logging.exception("Ocurrió un error inesperado durante el análisis: %s", e)
        raise HTTPException(status_code=500, detail=f"Ocurrió un error inesperado durante el análisis: {e}")

@app.get("/wiki")
async def generate_wiki_endpoint(request: wikiRequest):
    """
    Endpoint para generar la wiki del repositorio. La base de datos debe estar previamente poblada.
    """
    logging.info("Solicitud recibida para generar la wiki.")
    llm_client = create_llm_client(request.provider, request.llm_model)
    llm_client_embedding = create_llm_client(request.embedding_provider, request.embedding_model)

    if not llm_client:
        raise HTTPException(status_code=500, detail="Error: No se pudo inicializar el cliente LLM.")

    repo_name = os.path.basename(os.path.normpath(request.repo_path))
    collection_name = f"repo_documentation_{repo_name.replace('-', '_').lower()}"
    logging.info(f"[WIKI] Nombre de colección: {collection_name} (repo_path: {request.repo_path})")
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=f"./{collection_name}")

    return await generate_repo_wiki(llm_client_embedding, llm_client, db_manager)

if __name__ == "__main__":
    # Para ejecutar la API, se utiliza Uvicorn.
    logging.info(f"Puerto de la API: {os.getenv('PORT', 8010)}")
    logging.info("Iniciando Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8010)))


@app.get("/inspect")
async def inspect_file_metadata(repo_path: str, file_rel_path: str, provider: str, llm_model: str, embedding_provider: str, embedding_model: str):
    """Devuelve metadatos enriquecidos para un archivo/documento específico si existe en la colección.

    Params:
      repo_path: ruta raíz del repo documentado
      file_rel_path: ruta relativa del archivo a inspeccionar
    """
    repo_name = os.path.basename(os.path.normpath(repo_path))
    collection_name = f"repo_documentation_{repo_name.replace('-', '_').lower()}"
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=f"./{collection_name}")
    full_path_norm = os.path.normpath(os.path.join(repo_path, file_rel_path))
    # Query naive: embed path string to pull its vector neighbors (approx); could be improved with metadata filter if supported.
    embedding_client = create_llm_client(embedding_provider, embedding_model)
    query_embedding = embedding_client.generate_embedding(full_path_norm)
    results = db_manager.query(query_embedding, n_results=25)
    matches = []
    try:
        for i, meta in enumerate(results.get('metadatas', [[]])[0]):
            if meta.get('file_path') == full_path_norm:
                matches.append(meta)
    except Exception:
        pass
    if not matches:
        return {"status": "not_found", "file_path": full_path_norm}
    return {"status": "ok", "count": len(matches), "metadatas": matches[:3]}


@app.get("/graph")
async def get_dependency_graph(repo_path: str):
    """Devuelve grafo de dependencias y dependents + métricas básicas.

    Construye el grafo a partir de la colección documentada. Ideal para inspección rápida
    sin recalcular embeddings.
    """
    repo_name = os.path.basename(os.path.normpath(repo_path))
    collection_name = f"repo_documentation_{repo_name.replace('-', '_').lower()}"
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=f"./{collection_name}")
    data = db_manager.get_all(include_embeddings=False, limit=None)
    ids = data.get('ids') or []
    metadatas = data.get('metadatas') or []
    documents = []
    for i, mid in enumerate(ids):
        documents.append({"id": mid, "metadata": metadatas[i]})
    graph = build_dependency_graph(documents)
    inverted = invert_dependency_graph(graph)
    # Stats
    out_degree = {k: len(v) for k, v in graph.items()}
    in_degree = {k: len(v) for k, v in inverted.items()}
    return {
        "nodes": len(graph),
        "edges": sum(len(v) for v in graph.values()),
        "graph": {k: sorted(list(v)) for k, v in graph.items()},
        "inverted": {k: sorted(list(v)) for k, v in inverted.items()},
        "top_out_degree": sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:15],
        "top_in_degree": sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:15],
    }


@app.get("/decompress")
async def decompress_field(value: str):
    """Utilidad para inspeccionar campos comprimidos (::zlib64::...)."""
    return {"original_len": len(value), "decompressed": decompress_if_needed(value)}
@app.get("/db_quality")
async def db_quality(repo_path: str, include_embeddings: bool = False, sample: int = 2000):
    """Métricas de calidad/diagnóstico sobre la colección del repo.

    - include_embeddings: si True, calcula normas de embeddings (costo mayor).
    - sample: límite de elementos a leer para no cargar toda la colección.
    """
    repo_name = os.path.basename(os.path.normpath(repo_path))
    collection_name = f"repo_documentation_{repo_name.replace('-', '_').lower()}"
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=f"./{collection_name}")
    data = db_manager.get_all(include_embeddings=include_embeddings, limit=sample) or {}
    ids = data.get('ids') or []
    mds = data.get('metadatas') or []
    embs = data.get('embeddings') if include_embeddings else None
    n = len(ids)
    by_file = {}
    enriched = 0
    dup_keys = set()
    seen_keys = set()
    for i in range(n):
        md = mds[i] or {}
        fp = md.get('file_path') or 'unknown'
        ch = md.get('chunk_index')
        key = (fp, ch)
        if key in seen_keys:
            dup_keys.add(key)
        else:
            seen_keys.add(key)
        by_file[fp] = by_file.get(fp, 0) + 1
        if md.get('enriched_chunk') or md.get('enriched_chunks'):
            enriched += 1
    # Embedding norms
    norms = []
    if include_embeddings and embs:
        import math
        for e in embs[:n]:
            try:
                norms.append(math.sqrt(sum((x or 0.0)*(x or 0.0) for x in (e or []))))
            except Exception:
                continue
    stats = {
        'total_docs': n,
        'files_count': len(by_file),
        'docs_per_file_top10': sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:10],
        'enriched_pct': round((enriched / n) if n else 0.0, 3),
        'duplicate_keys_count': len(dup_keys),
    }
    if norms:
        stats['embedding_norm_min'] = round(min(norms), 4)
        stats['embedding_norm_avg'] = round(sum(norms)/len(norms), 4)
        stats['embedding_norm_max'] = round(max(norms), 4)
    return stats

@app.get("/db_sample")
async def db_sample(repo_path: str, limit: int = 50):
    """Devuelve una muestra de documentos (ids + metadatos básicos) para inspección rápida."""
    repo_name = os.path.basename(os.path.normpath(repo_path))
    collection_name = f"repo_documentation_{repo_name.replace('-', '_').lower()}"
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=f"./{collection_name}")
    data = db_manager.get_all(include_embeddings=False, limit=limit) or {}
    ids = data.get('ids') or []
    mds = data.get('metadatas') or []
    out = []
    for i in range(min(limit, len(ids))):
        md = mds[i] or {}
        ec = None
        try:
            if md.get('enriched_chunk'):
                import json as _json
                ec = _json.loads(md.get('enriched_chunk'))
        except Exception:
            ec = None
        out.append({
            'id': ids[i],
            'file_path': md.get('file_path'),
            'chunk_index': (ec or {}).get('chunk_index', md.get('chunk_index')),
            'chunk_type': (ec or {}).get('chunk_type', md.get('chunk_type')),
            'summary_preview': (ec or {}).get('summary', '')[:200],
        })
    return {'count': len(out), 'items': out}

@app.get("/db_dump_ids")
async def db_dump_ids(repo_path: str, limit: int = 200):
    """Devuelve pares id -> (file_path, chunk_index) para auditoría/inspección."""
    repo_name = os.path.basename(os.path.normpath(repo_path))
    collection_name = f"repo_documentation_{repo_name.replace('-', '_').lower()}"
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=f"./{collection_name}")
    data = db_manager.get_all(include_embeddings=False, limit=limit) or {}
    ids = data.get('ids') or []
    mds = data.get('metadatas') or []
    items = []
    for i in range(min(limit, len(ids))):
        md = mds[i] or {}
        items.append({
            'id': ids[i],
            'file_path': md.get('file_path'),
            'chunk_index': md.get('chunk_index')
        })
    return {'count': len(items), 'items': items}
