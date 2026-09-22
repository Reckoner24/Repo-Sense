# Repo-Sense

> **Work in progress.** This is an unfinished experiment, published as-is. The pieces below work, but there is no test suite, the API is not stable, and parts of it will change. Read it as a sketch of an approach, not a tool to depend on.
>
> <sub><b>ES</b> — Proyecto en curso, publicado tal cual. Lo que hay funciona, pero no tiene suite de pruebas, la API no es estable y varias partes van a cambiar. Está aquí como muestra de un enfoque, no como herramienta terminada.</sub>

An attempt at rebuilding [DeepWiki](https://deepwiki.com) for local repositories: point it at a codebase and it reads the source, works out how the files depend on each other, and has an LLM write the documentation — without the code ever leaving the machine, if you run it against Ollama.

<sub><b>ES</b> — Mi propio intento de recrear DeepWiki para repositorios locales: le indicas una carpeta con código, lee las fuentes, deduce cómo dependen unos archivos de otros y hace que un LLM escriba la documentación. Con Ollama, el código nunca sale de la máquina.</sub>

## How it works

```
repository ──► repo_analyzer ──► chunks ──► vector_db_manager (ChromaDB)
                    │                              │
                    ▼                              ▼
          dependency_orchestrator          wiki_agent_orchestrator
          (tree-sitter, javalang, ast)     (plan → retrieve → write)
                    │                              │
                    └──────────► llm_providers ◄───┘
                            Gemini · OpenAI · Groq · Ollama
```

| Module | What it does |
|---|---|
| `repo_analyzer.py` | Walks the repository, filters by extension and splits files into chunks |
| `dependency_orchestrator.py` | Extracts imports and references per language — Python via `ast`, Java via `javalang`, Scala and others via tree-sitter |
| `vector_db_manager.py` | Embeddings and retrieval on ChromaDB |
| `wiki_agent_orchestrator.py` | The agent loop: draws up a plan, retrieves relevant context, writes each section |
| `llm_providers.py` | One interface (`LlmClient`) behind Gemini, OpenAI, Groq and Ollama |
| `prompt_templates.py`, `enhanced_prompts.py` | Prompts for planning, writing and reviewing |
| `app.py` | FastAPI surface |

## Endpoints

| Route | Purpose |
|---|---|
| `POST /wiki_agent` | Generate documentation for a repository through the agent loop |
| `POST /document` | Document a single target |
| `GET /wiki` | Retrieve generated documentation |
| `GET /graph` | Dependency graph of the analysed repository |
| `GET /inspect` | Inspect the indexed state |
| `GET /db_quality`, `/db_sample`, `/db_dump_ids` | Inspect the vector store |

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in whichever provider you plan to use
python build_tree_sitter.py # build the tree-sitter grammars
uvicorn api.app:app --reload
```

Provider and model are chosen in `api/config.json`; `api/embedding_config.json` selects the embedding backend. Running Ollama for both keeps everything local.

## Known gaps

- No tests.
- Errors from the providers are not handled uniformly.
- The agent loop has no token budget, so a large repository gets expensive.
- Only Python, Java and Scala have real dependency extraction; the rest fall back to chunking alone.

## Stack

Python · FastAPI · ChromaDB · tree-sitter · LangChain · Gemini · OpenAI · Groq · Ollama
