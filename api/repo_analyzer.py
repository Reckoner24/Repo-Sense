# repo_analyzer.py

import os
import re
import csv
import logging
import subprocess
import json
import cssutils
import javalang
import tree_sitter_scala

from tree_sitter import Language, Parser
from typing import List, Tuple
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup

# Inicializar logging para el módulo
logger = logging.getLogger(__name__)

SCALA_LANGUAGE = Language(tree_sitter_scala.language())

# Definir las estrategias de chunking para diferentes tipos de archivos
def python_chunking_strategy(file_path: str) -> List[Tuple[str, str]]:
    """Chunking semántico de Python.

    Emite (en orden):
      imports, module_docstring, class_docstring, class, method_docstring, method,
      function_docstring, function, important_comments, top_level_code.

    Agrupa imports consecutivos y normaliza funciones/métodos async.
    """
    import ast

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return [("", "unreadable_file")]

    try:
        tree = ast.parse(content)
    except Exception:
        return [(content.strip(), "full_file")]

    lines = content.splitlines()
    chunks: List[Tuple[str, str]] = []
    covered_line_ranges: List[tuple[int, int]] = []

    # 1. Imports (consolidado)
    import_snippets: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            s = node.lineno - 1
            e = getattr(node, 'end_lineno', node.lineno)
            import_snippets.append('\n'.join(lines[s:e]))
            covered_line_ranges.append((s, e))
    if import_snippets:
        chunks.append(('\n'.join(import_snippets), 'imports'))

    # 2. Docstring de módulo
    module_doc = ast.get_docstring(tree)
    if module_doc:
        # localizar rango físico si está como primer statement
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], 'value', None), ast.Constant):
            ds_node = tree.body[0]
            ds_s = ds_node.lineno - 1
            ds_e = getattr(ds_node, 'end_lineno', ds_node.lineno)
            covered_line_ranges.append((ds_s, ds_e))
        chunks.append((module_doc, 'module_docstring'))

    # 3. Clases y métodos
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            s = node.lineno - 1
            e = getattr(node, 'end_lineno', node.lineno)
            class_code = '\n'.join(lines[s:e])
            covered_line_ranges.append((s, e))
            class_doc = ast.get_docstring(node)
            if class_doc:
                chunks.append((class_doc, 'class_docstring'))
            chunks.append((class_code, 'class'))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ms = sub.lineno - 1
                    me = getattr(sub, 'end_lineno', sub.lineno)
                    covered_line_ranges.append((ms, me))
                    method_code = '\n'.join(lines[ms:me])
                    method_doc = ast.get_docstring(sub)
                    if method_doc:
                        chunks.append((method_doc, 'method_docstring'))
                    chunks.append((method_code, 'method'))

    # 4. Funciones top-level (incluye async) que no están dentro de clases
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s = node.lineno - 1
            e = getattr(node, 'end_lineno', node.lineno)
            # evitar duplicados si ya se cubrió (no debería para top-level)
            already = False
            for a, b in covered_line_ranges:
                if s >= a and e <= b:
                    already = True
                    break
            if already:
                continue
            covered_line_ranges.append((s, e))
            func_code = '\n'.join(lines[s:e])
            func_doc = ast.get_docstring(node)
            if func_doc:
                chunks.append((func_doc, 'function_docstring'))
            chunks.append((func_code, 'function'))

    # 5. Comentarios importantes
    important = [ln for ln in lines if ln.strip().startswith('#') and any(tag in ln for tag in ('TODO', 'FIXME', 'NOTE'))]
    if important:
        chunks.append(('\n'.join(important), 'important_comments'))

    # 6. Código top-level residual
    residual_parts: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        s = node.lineno - 1
        e = getattr(node, 'end_lineno', node.lineno)
        # evitar solapados
        overlap = False
        for a, b in covered_line_ranges:
            if s >= a and e <= b:
                overlap = True
                break
        if overlap:
            continue
        snippet = '\n'.join(lines[s:e]).strip()
        if snippet:
            residual_parts.append(snippet)
    if residual_parts:
        chunks.append(('\n'.join(residual_parts), 'top_level_code'))

    if not chunks:
        chunks.append((content.strip(), 'full_file'))

    logger.info(f"[REPO_ANALYZER] Chunks generados: {[(t, len(c)) for c, t in chunks]}")
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:60] + '...' if len(c) > 60 else c)) for c, t in chunks]}")
    return chunks

def js_chunking_strategy(file_path: str) -> List[Tuple[str, str]]:
    """Chunking robusto de JS usando esprima, con tolerancia a errores de encoding.

    - Evita UnicodeDecodeError forzando decodificación binaria + fallback.
    - Si el archivo está dentro de node_modules (tercero), se devuelve un único chunk resumido
      para ahorrar costo (estos archivos suelen ser dependencias externas).
    """
    if 'node_modules' in file_path.replace('\\', '/').lower():
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(8000)  # solo primeros 8KB
            try:
                snippet = raw.decode('utf-8')
            except Exception:
                snippet = raw.decode('latin-1', errors='replace')
            return [(snippet, 'third_party_truncated')]
        except Exception:
            return [(file_path, 'unreadable_third_party')]

    # Lectura robusta
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        try:
            content = raw.decode('utf-8')
        except Exception:
            content = raw.decode('latin-1', errors='replace')
    except Exception as e:
        logger.error(f"No se pudo leer {file_path}: {e}")
        return [(file_path, 'unreadable')]

    try:
        node_script = """
        const esprima = require('esprima');
        const fs = require('fs');
        const code = fs.readFileSync(process.argv[2], 'utf8');
        const ast = esprima.parseScript(code, {range: true});
        process.stdout.write(JSON.stringify(ast));
        """
        with open('temp_esprima.js', 'w', encoding='utf-8') as temp_js:
            temp_js.write(node_script)
        result = subprocess.run(['node', 'temp_esprima.js', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_text = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
        ast = json.loads(stdout_text)
    except Exception as e:
        logger.error(f"Error al parsear AST JS: {e}")
        return [(content.strip(), 'full_file')]

    chunks: List[Tuple[str, str]] = []
    for node in ast.get('body', []):
        try:
            if node['type'] == 'FunctionDeclaration':
                start, end = node['range']
                chunks.append((content[start:end], 'function'))
            elif node['type'] == 'ClassDeclaration':
                start, end = node['range']
                chunks.append((content[start:end], 'class'))
            elif node['type'] == 'VariableDeclaration':
                for decl in node.get('declarations', []):
                    init = decl.get('init', {})
                    if init.get('type') in ['ArrowFunctionExpression', 'FunctionExpression']:
                        start, end = decl['range']
                        chunks.append((content[start:end], 'arrow_or_expr_function'))
            else:
                start, end = node['range']
                chunks.append((content[start:end], 'top_level_code'))
        except Exception:
            continue
    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def jsx_chunking_strategy(file_path: str) -> List[Tuple[str, str]]:
    """Chunking robusto de JSX con tolerancia a encoding y node_modules."""
    if 'node_modules' in file_path.replace('\\', '/').lower():
        try:
            with open(file_path, 'rb') as f:
                raw = f.read(8000)
            try:
                snippet = raw.decode('utf-8')
            except Exception:
                snippet = raw.decode('latin-1', errors='replace')
            return [(snippet, 'third_party_truncated')]
        except Exception:
            return [(file_path, 'unreadable_third_party')]
    try:
        with open(file_path, 'rb') as f:
            raw = f.read()
        try:
            content = raw.decode('utf-8')
        except Exception:
            content = raw.decode('latin-1', errors='replace')
    except Exception as e:
        logger.error(f"No se pudo leer {file_path}: {e}")
        return [(file_path, 'unreadable')]
    try:
        node_script = """
        const parser = require('@babel/parser');
        const fs = require('fs');
        const code = fs.readFileSync(process.argv[2], 'utf8');
        const ast = parser.parse(code, {sourceType: 'module', plugins: ['jsx']});
        process.stdout.write(JSON.stringify(ast));
        """
        with open('temp_babel.js', 'w', encoding='utf-8') as temp_js:
            temp_js.write(node_script)
        result = subprocess.run(['node', 'temp_babel.js', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_text = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
        ast = json.loads(stdout_text)
    except Exception as e:
        logger.error(f"Error al parsear AST JSX: {e}")
        return [(content.strip(), 'full_file')]
    chunks: List[Tuple[str, str]] = []
    for node in ast.get('program', {}).get('body', []):
        try:
            if node['type'] == 'FunctionDeclaration':
                start, end = node['start'], node['end']
                chunks.append((content[start:end], 'function_component'))
            elif node['type'] == 'ClassDeclaration':
                start, end = node['start'], node['end']
                chunks.append((content[start:end], 'class_component'))
            elif node['type'] == 'VariableDeclaration':
                for decl in node.get('declarations', []):
                    init = decl.get('init', {})
                    if init.get('type') in ['ArrowFunctionExpression', 'FunctionExpression']:
                        start, end = decl['start'], decl['end']
                        chunks.append((content[start:end], 'arrow_component'))
            elif node['type'] == 'ImportDeclaration':
                start, end = node['start'], node['end']
                chunks.append((content[start:end], 'import'))
            else:
                start, end = node.get('start', 0), node.get('end', 0)
                if start != end:
                    chunks.append((content[start:end], 'top_level_code'))
        except Exception:
            continue
    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def html_chunking_strategy(file_path: str) -> List[Tuple[str, str]]:
    """
    Chunking robusto de HTML usando BeautifulSoup.
    Separa secciones, etiquetas, scripts, estilos y comentarios.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    chunks = []

    # Secciones principales
    for tag in ['head', 'body', 'nav', 'footer', 'aside', 'section', 'article', 'div']:
        for element in soup.find_all(tag):
            chunks.append((str(element), tag))

    # Scripts
    for script in soup.find_all('script'):
        chunks.append((str(script), 'script'))

    # Estilos
    for style in soup.find_all('style'):
        chunks.append((str(style), 'style'))

    # Comentarios
    for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.comment))):
        chunks.append((str(comment), 'comment'))

    # Texto fuera de etiquetas
    for string in soup.stripped_strings:
        chunks.append((string, 'text'))

    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def css_chunking_strategy(file_path: str) -> List[Tuple[str, str]]:
    """
    Chunking robusto de CSS usando cssutils.
    Separa reglas, media queries, comentarios y directivas especiales.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    sheet = cssutils.parseString(content)
    chunks = []

    for rule in sheet:
        if rule.type == rule.STYLE_RULE:
            chunks.append((rule.cssText.decode('utf-8'), 'style_rule'))
        elif rule.type == rule.MEDIA_RULE:
            chunks.append((rule.cssText.decode('utf-8'), 'media_rule'))
        elif rule.type == rule.FONT_FACE_RULE:
            chunks.append((rule.cssText.decode('utf-8'), 'font_face_rule'))
        elif rule.type == rule.IMPORT_RULE:
            chunks.append((rule.cssText.decode('utf-8'), 'import_rule'))
        elif rule.type == rule.COMMENT:
            chunks.append((rule.cssText.decode('utf-8'), 'comment'))
        else:
            chunks.append((rule.cssText.decode('utf-8'), 'other_rule'))

    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def java_chunking_strategy(file_path: str) -> List[Tuple[str, str]]:
    """
    Chunking robusto de Java usando javalang.
    Separa clases, interfaces, enums, métodos, constructores, imports y comentarios.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        tree = javalang.parse.parse(content)
    except Exception as e:
        logger.error(f"Error al parsear AST Java: {e}")
        return [(content.strip(), 'full_file')]

    chunks = []
    # Clases, interfaces, enums
    for type_decl in getattr(tree, 'types', []):
        start = type_decl.position.line - 1 if type_decl.position else 0
        end = start + len(str(type_decl))
        chunks.append((content.splitlines()[start:end], type_decl.__class__.__name__.lower()))
        # Métodos y constructores
        for method in getattr(type_decl, 'methods', []):
            m_start = method.position.line - 1 if method.position else 0
            m_end = m_start + len(str(method))
            chunks.append((content.splitlines()[m_start:m_end], 'method'))
        for ctor in getattr(type_decl, 'constructors', []):
            c_start = ctor.position.line - 1 if ctor.position else 0
            c_end = c_start + len(str(ctor))
            chunks.append((content.splitlines()[c_start:c_end], 'constructor'))
        # Campos
        for field in getattr(type_decl, 'fields', []):
            f_start = field.position.line - 1 if field.position else 0
            f_end = f_start + len(str(field))
            chunks.append((content.splitlines()[f_start:f_end], 'field'))
    # Imports
    for imp in getattr(tree, 'imports', []):
        i_start = imp.position.line - 1 if imp.position else 0
        i_end = i_start + len(str(imp))
        chunks.append((content.splitlines()[i_start:i_end], 'import'))
    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def scala_chunking_strategy(file_path: str) -> List[Tuple[str, str]]:
    """Chunking Scala vía tree-sitter (optimizado para menos fragmentación).

    Política nueva:
      - Agrupa TODOS los imports en un único chunk (imports).
      - Emite 1 chunk por class/object/trait completo.
      - Emite 1 chunk por función (anotaciones incluidas si están contiguas encima del def).
      - Emite val/var SOLO si son top-level (no dentro de una función) y no triviales.
      - Emite comentarios SOLO si son block comments /** */ o contienen marcadores (TODO|FIXME|NOTE|----|+++++).
      - Fallback full_file si parser falla o resultado vacío.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return [("", 'unreadable_file')]

    try:
        parser = Parser()
        try:
            parser.set_language(SCALA_LANGUAGE)
        except AttributeError:
            parser.language = SCALA_LANGUAGE
        tree = parser.parse(content.encode('utf-8'))
        root = tree.root_node
    except Exception as e:
        logger.error(f"[REPO_ANALYZER][SCALA] Parser fallo: {e}")
        return [(content.strip(), 'full_file')]

    # Primera pasada: recolectar spans de funciones
    function_nodes = []
    imports_nodes = []
    class_like_nodes = []  # class, object, trait
    valvar_nodes = []
    comment_nodes = []

    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type
        if t == 'function_definition':
            function_nodes.append(n)
        elif t in ('class_definition', 'object_definition', 'trait_definition'):
            class_like_nodes.append(n)
        elif t == 'import':
            imports_nodes.append(n)
        elif t in ('val_definition', 'var_definition'):
            valvar_nodes.append(n)
        elif t == 'comment':
            comment_nodes.append(n)
        for ch in n.children:
            stack.append(ch)

    function_spans = [(n.start_byte, n.end_byte) for n in function_nodes]

    def inside_function(start: int, end: int) -> bool:
        for fs, fe in function_spans:
            if start >= fs and end <= fe:
                return True
        return False

    # Agrupar imports (en orden de aparición). A veces tree-sitter da nodos muy cortos (solo 'import').
    imports_nodes.sort(key=lambda n: n.start_byte)
    imports_code_parts = []
    for n in imports_nodes:
        snippet = content[n.start_byte:n.end_byte]
        # Si el snippet es solo 'import' o muy corto, extender hasta el fin de la línea
        if snippet.strip() == 'import' or len(snippet.strip()) <= 7:
            line_end = content.find('\n', n.end_byte)
            if line_end == -1:
                line_end = len(content)
            snippet = content[n.start_byte:line_end]
        imports_code_parts.append(snippet)

    # Fallback regex si seguimos teniendo fragmentos pobres
    needs_regex_fallback = not imports_code_parts or all(s.strip() == 'import' for s in imports_code_parts)
    if needs_regex_fallback:
        regex_lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith('import '):
                regex_lines.append(line.rstrip())
        if regex_lines:
            imports_code_parts = regex_lines
            logger.info("[REPO_ANALYZER][SCALA] Usando fallback regex para imports")

    imports_chunk: List[Tuple[str, str]] = []
    if imports_code_parts:
        # Eliminar duplicados preservando orden
        seen_import_lines = set()
        ordered_unique = []
        for line in imports_code_parts:
            if line not in seen_import_lines:
                seen_import_lines.add(line)
                ordered_unique.append(line)
        merged_imports = '\n'.join(ordered_unique).strip()
        imports_chunk.append((merged_imports, 'imports'))

    # Preparar chunks con (start_byte, end_byte, text, label)
    ordered: List[Tuple[int,int,str,str]] = []

    # Clases / objects / traits: emitir solo encabezado (firma + llaves de apertura) para evitar duplicar métodos
    for n in class_like_nodes:
        full_snippet = content[n.start_byte:n.end_byte]
        brace_idx = full_snippet.find('{')
        header = full_snippet if brace_idx == -1 else full_snippet[:brace_idx + 1]
        # Incluir anotaciones/comentarios inmediatamente encima igual que para funciones
        start = n.start_byte
        back_seek = start
        while back_seek > 0:
            prev_nl = content.rfind('\n', 0, back_seek - 1)
            line_start = 0 if prev_nl == -1 else prev_nl + 1
            line = content[line_start:back_seek].strip()
            if line.startswith('@') or line.startswith('//') or line.startswith('/*'):
                start = line_start
                back_seek = line_start
                continue
            break
        header_with_anns = content[start:start + len(header) + (n.start_byte - start)]
        ordered.append((start, start + len(header_with_anns), header_with_anns.rstrip(), 'class_header'))
    for n in function_nodes:
        # Incluir anotaciones inmediatamente previas (comentarios o @...) pegadas
        start = n.start_byte
        # retrocede mientras haya '@' o comentario en líneas anteriores contiguas
        back_seek = start
        while back_seek > 0:
            # buscar salto de línea anterior
            prev_nl = content.rfind('\n', 0, back_seek - 1)
            line_start = 0 if prev_nl == -1 else prev_nl + 1
            line = content[line_start:back_seek].strip()
            if line.startswith('@') or line.startswith('//') or line.startswith('/*'):
                start = line_start
                back_seek = line_start
                continue
            break
        ordered.append((start, n.end_byte, content[start:n.end_byte], 'function'))

    for n in valvar_nodes:
        if inside_function(n.start_byte, n.end_byte):
            continue  # ignorar internos
        snippet = content[n.start_byte:n.end_byte].strip()
        # filtrar definiciones triviales muy cortas (< 10 chars)
        if len(snippet) < 10:
            continue
        ordered.append((n.start_byte, n.end_byte, snippet, 'val_or_var'))

    interesting_comment_markers = ('TODO', 'FIXME', 'NOTE', '----', '++++', '/**')
    for n in comment_nodes:
        if inside_function(n.start_byte, n.end_byte):
            continue
        text = content[n.start_byte:n.end_byte]
        if any(m in text for m in interesting_comment_markers):
            ordered.append((n.start_byte, n.end_byte, text, 'comment'))

    # Añadir imports (poner con start 0 si están al inicio, de lo contrario su real posición mínima)
    if imports_chunk:
        # Usar posición mínima de los imports para ordenarlo correctamente
        first_pos = imports_nodes[0].start_byte if imports_nodes else 0
        ordered.append((first_pos, first_pos, imports_chunk[0][0], 'imports'))

    if not ordered:
        return [(content.strip(), 'full_file')]

    ordered.sort(key=lambda t: t[0])

    # Consolidar: si funciones están contenidas dentro de class/object ya representado podríamos mantener ambas (es valioso)
    # No deduplicamos aquí; dejamos al pipeline global filtrar duplicados por hash si aplica.

    result: List[Tuple[str, str]] = []
    seen_spans = set()
    for start, end, text, label in ordered:
        span = (start, end, label)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        if text.strip():
            result.append((text, label))

    if not result:
        result.append((content.strip(), 'full_file'))

    logger.info(f"[REPO_ANALYZER][SCALA] Optimized chunks: {[(t, len(c)) for c, t in result]}")
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in result]}")
    return result

def md_chunking_strategy(file_path: str) -> list:
    """
    Chunking robusto de Markdown usando markdown-it-py.
    Separa encabezados, listas, tablas, bloques de código, citas y texto plano.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    md = MarkdownIt()
    tokens = md.parse(content)
    chunks = []
    current_chunk = []
    current_type = None

    for token in tokens:
        if token.type.startswith('heading_open'):
            if current_chunk:
                chunks.append((''.join(current_chunk), current_type))
                current_chunk = []
            current_type = f'heading_{token.tag}'
        elif token.type == 'paragraph_open':
            if current_chunk:
                chunks.append((''.join(current_chunk), current_type))
                current_chunk = []
            current_type = 'paragraph'
        elif token.type == 'fence':
            if current_chunk:
                chunks.append((''.join(current_chunk), current_type))
                current_chunk = []
            current_type = 'code_block'
            chunks.append((token.content, current_type))
            current_type = None
        elif token.type in ['bullet_list_open', 'ordered_list_open']:
            if current_chunk:
                chunks.append((''.join(current_chunk), current_type))
                current_chunk = []
            current_type = 'list'
        elif token.type == 'table_open':
            if current_chunk:
                chunks.append((''.join(current_chunk), current_type))
                current_chunk = []
            current_type = 'table'
        elif token.type == 'blockquote_open':
            if current_chunk:
                chunks.append((''.join(current_chunk), current_type))
                current_chunk = []
            current_type = 'blockquote'
        elif token.type.endswith('_close'):
            if current_chunk:
                chunks.append((''.join(current_chunk), current_type))
                current_chunk = []
            current_type = None
        else:
            current_chunk.append(token.content if hasattr(token, 'content') else '')

    if current_chunk:
        chunks.append((''.join(current_chunk), current_type or 'miscellaneous_text'))
    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def json_chunking_strategy(file_path: str) -> list:
    """
    Chunking robusto de JSON. Separa objetos, arrays y propiedades individuales.
    """
    import os as _os
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        data = json.loads(content)
    except Exception as e:
        logger.error(f"Error al parsear JSON: {e}")
        return [(content.strip(), "json_content")]

    basename = _os.path.basename(file_path).lower()
    max_recursive_chunks = 80  # umbral para frenar explosión
    is_lock_like = basename.endswith('package-lock.json') or basename.endswith('yarn.lock') or 'lock' in basename
    is_large_file = len(content) > 250_000  # bytes aproximados

    # Heurística: archivos lock / muy grandes -> estrategia reducida para evitar cientos de chunks
    if is_lock_like or is_large_file:
        simplified_chunks = []
        if isinstance(data, dict):
            # Collapsar claves gigantes conocidas
            giant_keys = {'packages', 'dependencies', 'devDependencies'}
            for k, v in data.items():
                if k in giant_keys and isinstance(v, (dict, list)):
                    try:
                        size = len(v)
                    except Exception:
                        size = '?'
                    simplified_chunks.append((json.dumps({k: f"<COLLAPSED {size} entries>"}, ensure_ascii=False, indent=2), f"collapsed:{k}"))
                else:
                    # Solo serializa nivel superior sin recursión profunda
                    try:
                        simplified_chunks.append((json.dumps({k: v}, ensure_ascii=False, indent=2), f"top_property:{k}"))
                    except Exception:
                        simplified_chunks.append((f"{k}: <unserializable>", f"top_property:{k}"))
            meta = {
                "_note": "Simplified JSON chunking applied (lock/large file)",
                "file": basename,
                "original_size": len(content),
                "top_level_keys": list(data.keys()) if isinstance(data, dict) else None,
            }
            simplified_chunks.insert(0, (json.dumps(meta, ensure_ascii=False, indent=2), "meta:simplified"))
    
            logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
            return simplified_chunks or [(content[:50_000], "json_content_truncated")]
        
        # Si no es dict, caer a fallback común

        logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
        return [(content[:50_000], "json_content_truncated")]  # evita exceso

    chunks = []

    def extract_chunks(obj, path="root"):
        """Extrae chunks evitando duplicar hojas (Opción A: sin value:* para escalares).

        Regla: Para dict -> añadimos property:{path}.{k} y SOLO recursamos si v es dict/list.
               Para list -> añadimos array_item:{path}[i] y SOLO recursamos si item es dict/list.
               Para escalar raíz (no dict/list) añadimos value:root.
        """
        if len(chunks) >= max_recursive_chunks:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if len(chunks) >= max_recursive_chunks:
                    break
                chunks.append((json.dumps({k: v}, ensure_ascii=False, indent=2), f"property:{path}.{k}"))
                # Recurse solo si es dict, o si es lista con al menos un elemento compuesto
                if isinstance(v, dict):
                    # Si TODOS los valores del dict hijo son escalares, no descender (reduce fragmentación)
                    if any(isinstance(val, (dict, list)) for val in v.values()):
                        extract_chunks(v, f"{path}.{k}")
                elif isinstance(v, list):
                    has_complex = any(isinstance(it, (dict, list)) for it in v)
                    if has_complex:
                        extract_chunks(v, f"{path}.{k}")
        elif isinstance(obj, list):
            has_complex = any(isinstance(it, (dict, list)) for it in obj)
            if not has_complex:
                return  # lista solo de escalares -> ya cubierta por el chunk padre
            for idx, item in enumerate(obj):
                if len(chunks) >= max_recursive_chunks:
                    break
                chunks.append((json.dumps(item, ensure_ascii=False, indent=2), f"array_item:{path}[{idx}]"))
                if isinstance(item, (dict, list)):
                    extract_chunks(item, f"{path}[{idx}]")
        else:
            # Escalar raíz (o escalar bajo rama donde se llamó explícitamente) sólo se añade si path es root
            if path == "root":
                chunks.append((json.dumps(obj, ensure_ascii=False), f"value:{path}"))

    extract_chunks(data)
    if len(chunks) >= max_recursive_chunks:
        # Añade chunk final con resto colapsado
        chunks.append((json.dumps({"_notice": "CHUNK_LIMIT_REACHED", "limit": max_recursive_chunks}, ensure_ascii=False, indent=2), "meta:limit_reached"))
    if not chunks:
        chunks.append((content.strip(), "json_content"))
    
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def txt_chunking_strategy(file_path: str) -> list:
    """
    Chunking robusto de TXT. Detecta encabezados, listas, citas y párrafos.
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = []
    lines = content.splitlines()
    buffer = []
    current_type = None

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            if buffer:
                chunks.append(('\n'.join(buffer), current_type or 'paragraph'))
                buffer = []
                current_type = None
            continue
        # Encabezado: línea en mayúsculas o subrayada
        if re.match(r'^[A-Z][A-Z0-9 _-]+$', line_strip) or re.match(r'^=+$', line_strip):
            if buffer:
                chunks.append(('\n'.join(buffer), current_type or 'paragraph'))
                buffer = []
            current_type = 'header'
            buffer.append(line_strip)
        # Lista
        elif re.match(r'^(\*|-|•|[0-9]+[.)]) ', line_strip):
            if buffer and current_type != 'list':
                chunks.append(('\n'.join(buffer), current_type or 'paragraph'))
                buffer = []
            current_type = 'list'
            buffer.append(line_strip)
        # Cita
        elif line_strip.startswith('>'):
            if buffer and current_type != 'quote':
                chunks.append(('\n'.join(buffer), current_type or 'paragraph'))
                buffer = []
            current_type = 'quote'
            buffer.append(line_strip)
        else:
            if current_type not in [None, 'paragraph']:
                chunks.append(('\n'.join(buffer), current_type))
                buffer = []
            current_type = 'paragraph'
            buffer.append(line_strip)
    if buffer:
        chunks.append(('\n'.join(buffer), current_type or 'paragraph'))
    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def csv_chunking_strategy(file_path: str) -> list:
    """
    Chunking robusto de CSV. Separa encabezado, filas como dicts y celdas individuales.
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        chunks = []
        if header:
            chunks.append((str(header), 'csv_header'))
        for idx, row in enumerate(reader):
            chunks.append((str(row), f'csv_row_{idx}'))
            # Opcional: agregar celdas individuales
            for col, val in row.items():
                chunks.append((val, f'cell_{idx}_{col}'))
        if not chunks:
            f.seek(0)
            content = f.read()
            chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def sql_chunking_strategy(file_path: str) -> list:
    """
    Chunking para archivos SQL. Separa sentencias SQL individuales.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Dividir el contenido en sentencias SQL usando punto y coma como delimitador
    statements = [s.strip() for s in content.split(';') if s.strip()]
    chunks = [(stmt, 'sql_statement') for stmt in statements]
    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

def sh_chunking_strategy(file_path: str) -> list:
    """
    Chunking para archivos shell script. Separa funciones, comentarios y comandos.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.splitlines()
    chunks = []
    buffer = []
    current_type = None

    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            if buffer:
                chunks.append(('\n'.join(buffer), current_type or 'command'))
                buffer = []
                current_type = None
            continue

        if line_strip.startswith('#'):
            if buffer:
                chunks.append(('\n'.join(buffer), current_type or 'command'))
                buffer = []
            current_type = 'comment'
            buffer.append(line_strip)
        elif line_strip.startswith('function ') or re.match(r'^[a-zA-Z_[a-zA-Z0-9_]*\(\)', line_strip):
            if buffer:
                chunks.append(('\n'.join(buffer), current_type or 'command'))
                buffer = []
            current_type = 'function'
            buffer.append(line_strip)
        else:
            current_type = 'command'
            buffer.append(line_strip)

    if buffer:
        chunks.append(('\n'.join(buffer), current_type or 'command'))

    if not chunks:
        chunks.append((content.strip(), 'full_file'))
    logger.info(f"[REPO_ANALYZER] Chunks contents: {[(t, (c[:50] + '...' if len(c) > 50 else c)) for c, t in chunks]}")
    return chunks

# Diccionario exportado de estrategias de chunking por extensión
CHUNKING_STRATEGIES = {
    '.py': python_chunking_strategy,
    '.js': js_chunking_strategy,
    '.jsx': jsx_chunking_strategy,
    '.html': html_chunking_strategy,
    '.css': css_chunking_strategy,
    '.java': java_chunking_strategy,
    '.scala': scala_chunking_strategy,
    '.md': md_chunking_strategy,
    '.markdown': md_chunking_strategy,
    '.json': json_chunking_strategy,
    '.txt': txt_chunking_strategy,
    '.csv': csv_chunking_strategy,
    '.sql': sql_chunking_strategy,
    '.sh': sh_chunking_strategy,
}

__all__ = [
    'CHUNKING_STRATEGIES',
    'python_chunking_strategy', 'js_chunking_strategy', 'jsx_chunking_strategy',
    'html_chunking_strategy', 'css_chunking_strategy', 'java_chunking_strategy',
    'scala_chunking_strategy', 'md_chunking_strategy', 'json_chunking_strategy',
    'txt_chunking_strategy', 'csv_chunking_strategy', 'sql_chunking_strategy',
    'sh_chunking_strategy'
]
