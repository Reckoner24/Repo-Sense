import os
import json
import textwrap
from typing import Any, Dict, List, Tuple, Union

# ------------------------------------------------------------
# Utilidades internas
# ------------------------------------------------------------

def _safe_json_loads(s: Union[str, None]) -> Any:
    if not s or not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def _approx_token_len(text: str) -> int:
    # Aproximación simple: ~4 caracteres por token
    return max(1, len(text) // 4)

def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    approx_chars = max_tokens * 4
    if len(text) <= approx_chars:
        return text
    return text[:approx_chars]

def _normalize_section_title(title: str) -> str:
    return title.strip().replace("#", "").strip()

def _format_code_block(code: str, language_hint: str = "") -> str:
    language = language_hint or ""
    code = code or ""
    # Limitar tamaño de bloques de código extensos en el contexto
    if len(code) > 4000:
        code = code[:3800] + "\n# ... (truncado) ..."
    return f"```{language}\n{code}\n```"

def _chunk_preview(chunk: Dict[str, Any]) -> str:
    summary = chunk.get("summary") or ""
    original_code = chunk.get("original_code") or ""
    # pequeña vista previa
    code_preview = original_code[:800]
    preview = ""
    if summary:
        preview += f"Resumen: {summary}\n"
    if code_preview:
        preview += "Código:\n" + _format_code_block(code_preview)
    return preview.strip()

def _context_from_docs(
    relevant_docs: List[Dict[str, Any]],
    max_context_tokens: int,
    include_code: bool = True,
    prefer_summaries: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Construye un contexto textual robusto a partir de la estructura:
      relevant_docs = [{"metadatas": [metadata_dict], "documents": [text_or_none]}, ...]
    Retorna:
      - contexto_final (str)
      - lista de citas (cada item: {"file_path": str, "chunk_id": int, "chunk_type": str})
    """
    if not isinstance(relevant_docs, list):
        return "", []

    budget = max(256, max_context_tokens)
    consumed = 0
    parts: List[str] = []
    citations: List[Dict[str, Any]] = []

    for doc in relevant_docs:
        metadatas = doc.get("metadatas") or []
        documents = doc.get("documents") or []

        # Para cada metadata asociada (comúnmente 1)
        for md in metadatas:
            if consumed >= budget:
                break

            file_path = (md or {}).get("file_path") or "desconocido"
            chunks_json = (md or {}).get("chunks")
            doc_summary = (md or {}).get("document_summary") or ""

            chunks = _safe_json_loads(chunks_json)
            # Fallback: cuando trabajamos con embeddings por chunk, la metadata trae 'enriched_chunk'
            if (not chunks) and (md or {}).get("enriched_chunk"):
                ec = _safe_json_loads(md.get("enriched_chunk")) or {}
                # Construye una lista con un solo "chunk" compatible
                pseudo = {
                    "chunk_id": ec.get("chunk_index") if isinstance(ec.get("chunk_index"), int) else md.get("chunk_index"),
                    "chunk_type": ec.get("chunk_type") or md.get("chunk_type"),
                    "summary": ec.get("summary") or ec.get("refined_documentation") or "",
                    "original_code": ec.get("original_code") or "",
                }
                chunks = [pseudo]
            header = f"\n### Archivo: {file_path}\n"
            if consumed + _approx_token_len(header) < budget:
                parts.append(header)
                consumed += _approx_token_len(header)

            # Preferir summaries por chunk si hay presupuesto
            if isinstance(chunks, list) and chunks:
                for ch in chunks:
                    if consumed >= budget:
                        break
                    cid = ch.get("chunk_id")
                    ctype = ch.get("chunk_type") or "desconocido"

                    # armar bloque por chunk
                    block_lines = [f"- Chunk {cid} ({ctype})"]
                    if prefer_summaries and ch.get("summary"):
                        block_lines.append(f"  - {ch['summary']}")
                    chunk_block = "\n".join(block_lines) + "\n"
                    tok = _approx_token_len(chunk_block)
                    if consumed + tok >= budget:
                        break
                    parts.append(chunk_block)
                    consumed += tok
                    citations.append({"file_path": file_path, "chunk_id": cid, "chunk_type": ctype})

                    if include_code and ch.get("original_code"):
                        code_block = _format_code_block(ch["original_code"])
                        tok_code = _approx_token_len(code_block)
                        # Reduce ratio of raw code in context to keep prompts concise
                        if consumed + tok_code < budget and tok_code < (budget // 5):
                            parts.append(code_block)
                            consumed += tok_code
                        # si el bloque de código es muy grande, ya se truncó internamente

            # Si no hay chunks, intentar incluir el documento text plano si existe
            elif documents and documents[0]:
                raw_doc = str(documents[0])
                raw_block = _truncate_by_tokens(raw_doc, min(256, budget - consumed))
                tok = _approx_token_len(raw_block)
                if tok > 0 and consumed + tok < budget:
                    parts.append(raw_block)
                    consumed += tok

            # Añadir resumen del documento si queda presupuesto
            if doc_summary and consumed < budget:
                summary_block = f"\nResumen del documento:\n{doc_summary}\n"
                tok = _approx_token_len(summary_block)
                if consumed + tok < budget:
                    parts.append(summary_block)
                    consumed += tok

        if consumed >= budget:
            break

    context_text = "\n".join(parts).strip()
    return context_text, citations

# ------------------------------------------------------------
# Prompts enriquecidos
# ------------------------------------------------------------

def get_documentation_prompt(code_chunk: str, file_path: str, doc_type: str = "extensive_analysis") -> str:
    """
    Prompt robusto para documentar un chunk/archivo de código y retornar JSON estricto.
    """
    guidance = {
        "extensive_analysis": (
            "- Proporciona un análisis detallado de responsabilidades, flujo de datos, dependencias, efectos secundarios, excepciones posibles, "
            "complejidad aproximada (Big-O si aplica) y consideraciones de seguridad y rendimiento.\n"
        ),
        "superficial_summary": (
            "- Proporciona un resumen corto y claro del propósito y uso principal.\n"
        ),
        "api_reference": (
            "- Documenta parámetros (nombre, tipo, requerido/opcional, valores válidos), retornos, errores y ejemplos mínimos reproducibles.\n"
        ),
    }.get(doc_type, "- Ajusta el nivel de detalle al tipo solicitado.\n")

    prompt = f"""
Eres un asistente técnico que documenta código de manera rigurosa. Trabaja con el archivo:
- Ruta del archivo: {file_path}
- Tipo de documentación: {doc_type}

Instrucciones clave:
- Responde exclusivamente en JSON válido y nada más (sin comentarios, sin texto fuera del JSON).
- Si careces de información suficiente, pon valores vacíos pero no inventes.
- Deduce el lenguaje del bloque de código si es posible.
- No incluyas información externa; describe solo lo que esté evidenciado en el código.
- Si detectas patrones peligrosos (p. ej., claves, contraseñas), inclúyelos en 'security' como advertencias.

Alcance adicional:
{guidance.strip()}

Notas estrictas de formato:
- Devuelve SOLO un objeto JSON, sin backticks ni texto adicional.
- Usa exactamente las claves del esquema; no agregues otras.

Esquema JSON requerido:
{{
  "summary": "Descripción general concisa.",
  "functions": [
    {{
      "name": "string",
      "signature": "string (si es posible)",
      "purpose": "string",
      "parameters": [{{"name": "string", "type": "string", "description": "string"}}],
      "returns": "string",
      "exceptions": ["string"],
      "side_effects": ["string"],
      "complexity": "string"
    }}
  ],
  "classes": [
    {{
      "name": "string",
      "purpose": "string",
      "methods": ["string"],
      "relationships": ["string"]
    }}
  ],
  "dependencies": ["import/require/..."],
  "security": ["posibles riesgos o datos sensibles"],
  "performance": ["hotspots o recomendaciones"],
  "edge_cases": ["casos especiales/errores comunes"],
  "example": {{
    "description": "cómo usar el código",
    "code": "bloque de ejemplo en el lenguaje correcto"
  }}
}}

Código a documentar:
{_format_code_block(code_chunk)}
"""
    # sin textwrap.dedent para mantener formato controlado
    return prompt.strip()


def get_wiki_generation_prompt(query: str, relevant_docs: List[Dict[str, Any]], max_context_tokens: int) -> str:
    """
    Prompt para generar la wiki global a partir de documentos relevantes.
    Construye contexto desde metadatos/chunks y exige citar rutas de archivo + chunk_id.
    """
    context_text, citations = _context_from_docs(
        relevant_docs=relevant_docs,
        max_context_tokens=max_context_tokens,
        include_code=True,
        prefer_summaries=True,
    )

    # Inferir extensiones observadas en el contexto para orientar la sección de tecnologías
    observed_exts = set()
    try:
        for doc in relevant_docs or []:
            for md in (doc.get("metadatas") or []):
                fp = (md or {}).get("file_path") or ""
                _, ext = os.path.splitext(fp)
                if ext:
                    observed_exts.add(ext.lower())
    except Exception:
        pass
    observed_exts_text = ", ".join(sorted(observed_exts)) if observed_exts else "(no detectadas)"

    citations_text = ""
    if citations:
        unique = []
        seen = set()
        for c in citations:
            key = (c.get("file_path"), c.get("chunk_id"))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        lines = [f"- {c['file_path']} (chunk {c['chunk_id']}, tipo: {c.get('chunk_type')})" for c in unique]
        citations_text = "\n".join(lines)

    prompt = f"""
Rol:
Eres un redactor técnico que elabora documentación tipo wiki exhaustiva, consistente y navegable en Markdown.

Reglas de oro:
- Trabaja únicamente con la información del contexto provisto; NO inventes ni extrapoles más allá de lo evidenciado.
- Si el contexto es insuficiente para una sección, indícalo explícitamente: "_No hay suficiente contexto para detallar esta sección_".
- Escribe en español claro y profesional.
- Incluye referencias en línea a las fuentes del repo (ruta de archivo y chunk) cuando cites detalles específicos.
- No generes "Tabla de Contenidos" ni "Target Audience".
- No incluyas una sección de "Tecnologías" salvo que haya evidencia directa en el contexto. Usa como guía las extensiones observadas y archivos/manifiestos. No menciones JavaScript/React/Node.js si no existen .js/.jsx/.tsx o package.json; no menciones Git salvo evidencia explícita.
- En "Arquitectura", cuando cites archivos, incluye 2-4 bullets concretos por archivo (responsabilidades, funciones/entrypoints clave, dependencias, relación con otros módulos). Si falta contexto para un archivo, omite esa entrada o márcala con un TODO breve en español. No uses "(Details missing)".

Consulta:
{query}

Extensiones de archivo observadas en el contexto: {observed_exts_text}

Contexto del repositorio (recuperado vía RAG, con fragmentos relevantes):
{context_text if context_text else "[Contexto vacío]"}

Citas disponibles:
{citations_text if citations_text else "- (sin citas recopiladas, el contexto fue limitado)"}

Tarea:
- Construye la wiki completa en Markdown sin incluir "Tabla de Contenidos" ni "Target Audience".
- Solo incluye "Tecnologías utilizadas" si hay evidencia en el contexto (basada en extensiones/archivos); de lo contrario, omite esa sección.
- Cuando menciones detalles puntuales (interfaces, funciones, rutas, configuraciones), agrega una notación en paréntesis con la referencia al archivo y chunk, p. ej.: (ver src/app.py, chunk 3).
- Evita repetir texto innecesariamente; sé jerárquico y evita contradicciones.
"""
    return prompt.strip()


def get_section_generation_prompt(section_title: str, relevant_docs: List[Dict[str, Any]], max_context_tokens: int) -> str:
    """
    Prompt para generar una sección específica de la wiki.
    """
    section_title = _normalize_section_title(section_title)
    context_text, citations = _context_from_docs(
        relevant_docs=relevant_docs,
        max_context_tokens=max_context_tokens,
        include_code=True,
        prefer_summaries=True,
    )
    citations_text = "\n".join(
        [f"- {c['file_path']} (chunk {c['chunk_id']}, tipo: {c.get('chunk_type')})" for c in citations]
    ) if citations else "- (sin citas recopiladas)"

    prompt = f"""
Eres un redactor técnico. Escribe la sección en español con rigor, sin inventar, basada solo en el contexto.
Sección objetivo: {section_title}

Reglas:
- No generes "Tabla de Contenidos" ni "Target Audience".
- Cita fuentes del repo cuando des detalles: (archivo, chunk).
- Si la sección incluye "Arquitectura", ofrece bullets concretos (2-4) por archivo con responsabilidades, entrypoints, dependencias y relaciones; evita placeholders como "(Details missing)". Si no hay contexto suficiente, omite esa entrada o marca un TODO breve en español.
- Solo menciona "Tecnologías" si hay evidencia directa en el contexto; de lo contrario, omite esa parte.
 - Mantén cohesión con una wiki mayor (tono y estilo consistentes).
 - Extensión objetivo: 150-300 palabras; evita relleno y repeticiones.

Contexto relevante:
{context_text if context_text else "[Contexto vacío]"}

Citas:
{citations_text}

Entrega la sección en Markdown (150-300 palabras), encabezada por:
## {section_title}
"""
    return prompt.strip()


def get_wiki_refinement_prompt(draft_markdown: str, query: str) -> str:
    """
    Prompt para refinar un borrador de wiki: unificar estilo, corregir inconsistencias y pulir contenido.
    """
    prompt = f"""
Actúa como editor técnico. Revisa y refina el siguiente borrador de wiki para la consulta: "{query}".

Criterios de calidad:
- No añadas contenido nuevo ni amplíes la longitud (>110%).
- Unifica el tono y formato. No añadas "Tabla de Contenidos".
- Elimina secciones "Tabla de Contenidos" y "Target Audience" si aparecen.
- Corrige/retira la sección de "Tecnologías utilizadas" si no hay evidencia explícita en el propio borrador que la respalde (extensiones/archivos mencionados). Elimina menciones a JavaScript/React/Node.js si no hay artefactos afines (.js/.jsx/.tsx, package.json).
- En "Arquitectura", elimina placeholders como "(Details missing)"; si falta contexto, marca un TODO breve en español o reestructura para omitir entradas vacías.
- Verifica que no existan contradicciones internas. Si detectas carencias evidentes, señala TODOs claros.
- Mantén los bloques de código y corrige errores de formato Markdown.
- No inventes funcionalidades que no aparezcan en el borrador.

Borrador:
{draft_markdown}

Devuelve la versión refinada en Markdown completo.
"""
    return prompt.strip()


def get_wiki_structure_prompt(context: Union[str, List[Dict[str, Any]]], query: str) -> str:
    """
    Prompt para proponer una estructura (índice) y una to-do list ordenada.
    Acepta contexto como texto o como docs estilo RAG.
    """
    if isinstance(context, list):
        ctx_text, _ = _context_from_docs(context, max_context_tokens=2000, include_code=False, prefer_summaries=True)
    else:
        ctx_text = str(context or "")

    prompt = f"""
Eres un planificador de documentación. Con base en la consulta "{query}" y el contexto,
propón una estructura de wiki y una lista de tareas mínima para completarla.

Contexto (resumen):
{ctx_text if ctx_text else "[Contexto vacío]"}

Instrucciones:
- Dado que el contexto puede ser incompleto, propone una estructura razonable y adaptable.
- No inventes detalles, mantén los títulos neutrales y específicos al dominio del repo.
- Ordena la to-do list en el orden lógico de elaboración (de base a avanzado).
- Limita la to-do list a 12-20 títulos.
- Responde en JSON estricto con las claves: "estructura" (lista de secciones/subsecciones) y "todo_list" (lista de títulos), sin backticks ni texto adicional.
"""
    return prompt.strip()


def get_wiki_agent_orchestration_prompt(estructura: Union[str, List[str], Dict[str, Any]], todo_list: List[str], query: str) -> str:
    """
    Prompt para que el LLM explique el plan de acción del agente sobre la wiki.
    """
    estr = estructura
    if not isinstance(estr, str):
        try:
            estr = json.dumps(estructura, ensure_ascii=False, indent=2)
        except Exception:
            estr = str(estructura)

    todo = "\n".join([f"- {t}" for t in (todo_list or [])]) if todo_list else "- (vacío)"

    prompt = f"""
Eres un agente de IA experto en documentación técnica de repositorios.
Debes explicar de forma breve y accionable cómo abordarás la generación de la wiki para la consulta: "{query}".

Estructura propuesta:
{estr}

To-do list:
{todo}

Explica tu plan de acción en 3-5 pasos, en español, considerando:
- Recuperación iterativa de contexto por sección y verificación de relevancia.
- Citas a archivos/chunks del repositorio en cada sección.
- Manejo de carencias de contexto (cómo reaccionar y qué solicitar).
- Refinamiento global final para consistencia y enlaces internos.
"""
    return prompt.strip()


def get_summary_prompt(text: str) -> str:
    """
    Prompt simple de resumen, usado cuando se requiera.
    """
    return f"Resume en 3-5 oraciones, con enfoque técnico y sin inventar:\n\n{text}"
