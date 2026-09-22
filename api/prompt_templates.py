# prompt_templates.py

import json
import logging
from langchain.prompts import PromptTemplate

json_structure = r'{summary: , "parameters": {}, "return_value": {}, "example": {}}'

def get_summary_prompt(code_chunk):
    """
    Generates a prompt to summarize a code fragment.

    Args:
        code_chunk (str): The code fragment to summarize.

    Returns:
        str: The complete prompt for the AI.
    """
    return f"""
    You are an expert programming assistant. Your task is to generate a concise and high-quality summary for the following code fragment.
    The summary should be detailed enough for a developer to understand the main purpose, key logic, and important dependencies, but without exceeding the limit of 500 tokens.

    Code to summarize:
    ```
    {code_chunk}
    ```

    Summary:
    """

def get_chunk_analysis_prompt(chunk_type, file_path, chunk_content):
    return f"""
You are a highly specialized software documentation agent. Your task is to analyze the following code or document chunk in depth.
Your analysis must be exhaustive, detailed, and cover every aspect of the chunk.
You must identify and describe:
- The main purpose and functionality of the chunk. Explain what it does, why it exists, and how it fits into the overall project.
- All internal dependencies (references to other parts of the codebase, functions, classes, variables, configuration, etc.).
- All external dependencies (libraries, frameworks, APIs, environment requirements, etc.).
- Every key element present in the chunk (functions, classes, methods, configuration blocks, data structures, constants, etc.).
- Any relevant context, including comments, docstrings, usage notes, and integration details.
- Any potential risks, limitations, or caveats associated with this chunk.
- Any relationships or interactions with other chunks or files.

Repeat: Your analysis must be complete, leaving no aspect unexplored. Be explicit and verbose in your explanations.

Chunk type: {chunk_type}
File path: {file_path}
Code/Content:
{chunk_content}

Return ONLY a valid JSON object with the following keys:
- summary: Detailed summary of the chunk's purpose and role.
- dependencies: List of all internal and external dependencies, with explanations.
- key_elements: List and description of all key elements found.
- context: All relevant context, comments, and integration notes.
- risks: Any risks, limitations, or caveats.
- relationships: Interactions with other chunks or files.

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid JSON structure specified above
- DO NOT wrap the JSON in markdown code blocks (no \\`\\`\\` or \\`\\`\\`JSON)
- DO NOT include any explanation text before or after the JSON
- Ensure the JSON is properly formatted and valid
- Start directly with "{" and end with "}"

IMPORTANT: Return ONLY valid JSON with the structure specified above, with no markdown code block delimiters

Do not omit any section. If a section is not applicable, state explicitly: 'None found'.
"""

def get_chunk_documentation_prompt(chunk_type, file_path, chunk_content):
    return f"""
You are an expert technical writer and documentation agent. Your task is to generate the most complete, clear, and useful documentation possible for the following code or document chunk.
Your documentation must be:
- Extremely detailed, covering every aspect of the chunk.
- Clear and understandable for both experts and beginners.
- Redundant: repeat key points and explanations in different ways to ensure clarity.
- Structured: use sections, lists, and examples wherever possible.

You must include:
- A comprehensive summary of the chunk's purpose and usage.
- Detailed explanations of every component, function, class, method, configuration, and data structure.
- Multiple usage examples, covering typical, edge, and error cases.
- Explicit notes on limitations, caveats, and best practices.
- Cross-references to related chunks, files, or documentation sections.
- Any relevant context, including comments, docstrings, and integration details.

Repeat: Your documentation must be exhaustive, redundant, and structured. Do not leave out any detail.

Chunk type: {chunk_type}
File path: {file_path}
Code/Content:
{chunk_content}

Return ONLY a valid JSON object with the following keys:
- summary: Comprehensive summary.
- details: Detailed explanations of all components.
- examples: Multiple usage examples.
- notes: Limitations, caveats, and best practices.
- cross_references: Related chunks or files.
- context: Additional context and integration notes.

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid JSON structure specified above
- DO NOT wrap the JSON in markdown code blocks (no \\`\\`\\` or \\`\\`\\`JSON)
- DO NOT include any explanation text before or after the JSON
- Ensure the JSON is properly formatted and valid
- Start directly with "{" and end with "}"

IMPORTANT: Return ONLY valid JSON with the structure specified above, with no markdown code block delimiters

If any section is not applicable, state explicitly: 'None found'.
"""

def get_chunk_refinement_prompt(documentation_json):
    return f"""
You are a documentation refinement agent. Your task is to review and improve the following documentation, making it as clear, complete, and useful as possible.
You must:
- Clarify and expand all explanations.
- Add missing details, examples, and notes.
- Repeat key points in different ways for redundancy.
- Ensure technical accuracy and completeness.
- Structure the documentation with clear sections and lists.
- Remove ambiguity and ensure every section is explicit.

Repeat: Your refinement must be exhaustive, redundant, and structured. Do not omit any section.

Original documentation (JSON):
{documentation_json}

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid JSON structure specified above
- DO NOT wrap the JSON in markdown code blocks (no \\`\\`\\` or \\`\\`\\`JSON)
- DO NOT include any explanation text before or after the JSON
- Ensure the JSON is properly formatted and valid
- Start directly with "{" and end with "}"

IMPORTANT: Return ONLY valid JSON with the structure specified above, with no markdown code block delimiters

If any section is not applicable, state explicitly: 'None found'.
"""

def get_orchestration_prompt(chunks_analysis_json):
    return f"""
You are an orchestration agent for software documentation. Your task is to analyze the following list of code/document chunks and their analysis, and decide:
- The optimal order to document and refine them, based on dependencies, complexity, and relevance.
- Which chunks require more attention, deeper analysis, or multiple refinement passes.
- Any recommendations for parallel, iterative, or agentic processing.
- Any risks, caveats, or special considerations for the documentation process.

Repeat: Your orchestration must be exhaustive, redundant, and explicit. Justify every decision and recommendation.

Chunks analysis (JSON list):
{chunks_analysis_json}

Return ONLY a valid JSON object with the following keys:
- ordered_chunks: List of chunks in optimal documentation order, with justification.
- priority_notes: Notes on which chunks require special attention and why.
- recommendations: Detailed recommendations for processing, including parallelization and iteration.
- risks: Any risks or caveats for the process.

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid JSON structure specified above
- DO NOT wrap the JSON in markdown code blocks (no \\`\\`\\` or \\`\\`\\`JSON)
- DO NOT include any explanation text before or after the JSON
- Ensure the JSON is properly formatted and valid
- Start directly with "{" and end with "}"

IMPORTANT: Return ONLY valid JSON with the structure specified above, with no markdown code block delimiters

If any section is not applicable, state explicitly: 'None found'.
"""

# --- NUEVOS PROMPTS PARA AGENTE ITERATIVO WIKI ---
def get_wiki_structure_prompt(context, query):
    """
    Prompt for the LLM to propose a wiki structure and a to-do list of sections to develop.
    """
    return f"""
    You are an expert in technical documentation and wiki writing for software projects.
    Based solely on the following context retrieved from the repository, analyze and propose an EXHAUSTIVE AND CONCRETE wiki structure.

    Context:
    ---
    {context}
    ---

    The user's question is: "{query}"

    IMPORTANT INSTRUCTIONS:
    - Each title and subtitle in the index must explicitly reference REAL files, paths, classes, functions, endpoints, or components of the project.
    - DO NOT use generic phrases or templates like "replace this with the correct information".
    - DO NOT include empty sections or placeholders.
    - The to-do list must be specific and aligned with real elements of the code and architecture.
    - If the context is insufficient, explicitly indicate what information is missing, but never invent or use generic templates.

    Return ONLY a JSON with two fields: "structure" (wiki index) and "todo_list" (list of sections to develop).
    """

def get_section_generation_prompt(context, section_title):
    """
    Prompt for the LLM to generate a wiki section based on the relevant context and the section title.
    """
    return f"""
    You are an expert technical writer for software wikis.
    Your task is to write the section "{section_title}" of a wiki, using only the provided context.

    Relevant context:
    ---
    {context}
    ---

    IMPORTANT INSTRUCTIONS:
    - Explicitly cite the relevant files, paths, classes, functions, endpoints, and code fragments of the project.
    - DO NOT use generic phrases or templates like "replace this with the correct information".
    - DO NOT include empty sections or placeholders.
    - If the context is insufficient, explicitly indicate what information is missing, but never invent or use generic templates.
    

    Write the section in Markdown, with clear explanations, examples, and Mermaid.js diagrams if useful. Cite the relevant files or fragments when appropriate.
    """

def get_wiki_refinement_prompt(wiki_draft, query):
    """
    Prompt for the LLM to review and improve the global coherence of the wiki, add transitions, and ensure Markdown/diagram formatting.
    """
    return f"""
    You are an expert editor in technical documentation.
    Your task is to review the following wiki draft generated for the query: "{query}".

    1. Improve the overall coherence and flow between sections.
    2. Add transitions and internal links if useful.
    3. Ensure the Markdown format is correct and consistent.
    4. Add or improve Mermaid.js diagrams where they help understanding.
    5. If a table of contents is missing, add it at the beginning.
    6. REMOVE any generic text, template, placeholder, or phrases like "replace this with the correct information". ALL content must be based on real code/documents.

    Wiki draft:
    ---
    {wiki_draft}
    ---

    Return only the final wiki text in Markdown.
    """

def get_wiki_agent_orchestration_prompt(estructura, todo_list, query):
    """
    Prompt for the LLM to act as a planning agent and briefly explain how it will approach the generation of the wiki, using the structure and to-do list.
    """
    return f"""
    You are an AI agent expert in technical documentation planning.
    Your task is, given the following wiki index and to-do list, to briefly explain how you will approach the generation of the wiki for the query: '{query}'.

    Proposed index:
    {estructura}

    To-do list:
    {todo_list}

    Explain your action plan in 3-5 steps, in English, to ensure the wiki will be exhaustive, coherent, and well-structured.
    """

def get_documentation_prompt(code_chunk, file_path, doc_type="superficial_summary"):
    """
    Genera un prompt para documentar un fragmento de código basado en el tipo de documentación deseado.

    Args:
        code_chunk (str): El fragmento de código a documentar (puede ser el código original o un resumen).
        file_path (str): La ruta al archivo original del chunk.
        doc_type (str): El tipo de documentación deseado ("superficial_summary" o "extensive_analysis").

    Returns:
        str: El prompt completo para la IA.
    """

    if doc_type == "extensive_analysis":
        # Prompt para un análisis detallado y exhaustivo
        return "\n".join(line.strip() for line in f"""
        You are a programming assistant expert in documentation. Your task is to generate complete and detailed technical documentation for a code snippet.
        The code snippet comes from the file: "{file_path}"

        Code to document:
        ```
        {code_chunk}
        ```

        Deeply analyze the code, including its purpose, logic, parameters, and return value. Generate a response in JSON format containing the following fields:
        1.  "summary": A comprehensive and clear description of the code functionality.
        2.  "parameters": A dictionary of parameters. The keys should be the parameter names and the values should be detailed descriptions of their purpose, expected data type, and, if applicable, default values. If there are no parameters, use an empty dictionary.
        3.  "return_value": A description of the return value, including the data type and its meaning. If it does not return anything, use an empty string.
        4.  "example": A complete and commented usage example, clearly showing the expected input and output.

        Make sure the response is a valid JSON object, without any additional text outside the object. You have to return your analysis in the following JSON format:

        {json_structure}

        IMPORTANT FORMATTING INSTRUCTIONS:
        - Return ONLY the valid JSON structure specified above
        - DO NOT wrap the JSON in markdown code blocks (no \\`\\`\\` or \\`\\`\\`JSON)
        - DO NOT include any explanation text before or after the JSON
        - Ensure the JSON is properly formatted and valid
        - Start directly with "{" and end with "}"

        IMPORTANT:
        1. Return ONLY valid JSON with the structure specified above, with no markdown code block delimiters
        """.splitlines())

    elif doc_type == "superficial_summary":
        # Prompt para un resumen rápido y conciso
        return "\n".join(line.strip() for line in f"""
        You are a programming assistant expert in documentation. Your task is to generate concise technical documentation for a code snippet.
        The code snippet comes from the file:: "{file_path}"

        Code to document:
        ```
        {code_chunk}
        ```

        Generate a response in JSON format containing the following fields:
        1.  "summary": A brief description of what the code does.
        2.  "parameters": A dictionary of parameters if it is a function or method. The keys should be the parameter names and the values should be descriptions. If there are no parameters, use an empty dictionary.
        3.  "return_value": A description of the value that the function or method returns. If it does not return anything, use an empty string.
        4.  "example": A quick usage example.
        
        Make sure the response is a valid JSON object, without any additional text outside the object. You have to return your analysis in the following JSON format:

        {json_structure}

        IMPORTANT FORMATTING INSTRUCTIONS:
        - Return ONLY the valid JSON structure specified above
        - DO NOT wrap the JSON in markdown code blocks (no \\`\\`\\` or \\`\\`\\`JSON)
        - DO NOT include any explanation text before or after the JSON
        - Ensure the JSON is properly formatted and valid
        - Start directly with "{" and end with "}"

    IMPORTANT:
    1. Return ONLY valid JSON with the structure specified above, with no markdown code block delimiters
        """.splitlines())

    else:
        raise ValueError("Tipo de documentación no válido. Usa 'superficial_summary' o 'extensive_analysis'.")

def get_wiki_generation_prompt(query, relevant_docs, max_context_tokens):
    """
    Generates a prompt to create a complete wiki from the relevant documents.
    
    Logic has been added to limit the size of the prompt, preventing
    it from exceeding the model's token limit.

    Args:
        query (str): The user's question.
        relevant_docs (list): A list of relevant documents from ChromaDB.
        max_context_tokens (int): The maximum number of tokens for the context.

    Returns:
        str: The complete prompt for the AI.
    """
    context_str = ""
    current_tokens = 0
    
    # Simple aproximación para contar tokens (asumiendo que 1 token ≈ 4 caracteres)
    # Esto es una manera rápida de evitar exceder el límite, aunque no es exacto.
    char_per_token = 4

    for doc in relevant_docs:
        # Estima el tamaño del siguiente documento a agregar
        try:
            logging.info(f"Analizando información del documento: {doc}")
            full_chunks = json.loads(doc['metadatas'][0]["chunks"])
            
            documentacion_completa=[]
            codigo_original=[]
            for chunk in full_chunks:
                documentacion_completa.append(chunk["full_documentation"])
                codigo_original.append(chunk["original_code"])

            doc_text = (
                f"### Archivo: {doc['metadatas'][0]['file_path']}\n"
                f"**Documentación completa:** {documentacion_completa}\n"
                f"**Código Original:**\n```\n{codigo_original}\n```\n\n"
            )
            # Verifica si el nuevo documento excederá el límite
            if current_tokens + (len(doc_text) / char_per_token) < max_context_tokens:
                context_str += doc_text
                current_tokens += len(doc_text) / char_per_token
            else:
                # Si se excede, detiene la adición de documentos y avisa
                logging.warning("Advertencia: Se alcanzó el límite de tokens para el contexto. Algunos documentos no fueron incluidos.")
                break
        except Exception as e:
            if str(e) in ("KeyError", "json.JSONDecodeError"):
                logging.error(f"prompt_templates. Se tuvo el error: (\"KeyError\", \"json.JSONDecodeError\") {e}")
                continue
            else:
                logging.error(f"prompt_templates. Se tuvo el error: {e}")
                raise e

    return "\n".join(line.strip() for line in f"""
    You are an expert programming assistant in technical documentation. Your task is to act as a technical writer and generate a complete and coherent "wiki" document for a code repository, based on the information provided.

    The user's question is: "{query}"

    Here is the contextual information extracted from the repository files:
    ---
    {context_str}
    ---
    
    Please use the above context to generate a well-structured and easy-to-read wiki. The document should:
    1.  Have a clear structure with Markdown headers and subheaders.
    2.  Use Markdown text styles (bold, italics, etc.) to highlight important points.
    3.  Include code blocks with syntax highlighting where necessary.
    4.  Generate visually attractive diagrams when useful to explain workflows or dependencies. For this, use **Mermaid.js** syntax. An example of the syntax is:
        ```mermaid
        graph TD
            A[User] --> B[Validate data]
            B -->|Yes| C[Process request]
            B -->|No| D[Show error]
        ```
    5.  Explain the general purpose of the repository.
    6.  Describe the key components, files, and their functionality concisely but in detail.
    7.  Be easy to understand for a new developer who wants to contribute to the project.

    Generate only the wiki content in Markdown format, with no preamble.
    """.splitlines())
