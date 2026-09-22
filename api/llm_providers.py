
# llm_providers.py
import os
import logging
import google.generativeai as genai
import requests
import re
import json as _json
import google.api_core.exceptions

from abc import ABC, abstractmethod
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from ollama import Client as OllamaClientLib
from api.prompt_templates import get_documentation_prompt
from time import sleep

# Cargar variables de entorno una vez
load_dotenv()

class LlmError(Exception):
    """Excepción personalizada para errores del LLM."""
    pass

class LlmClient(ABC):
    def generate_wiki_structure_and_todo(self, context: str, query: str) -> dict:
        """
        Usa el LLM para proponer una estructura de wiki y una to-do list, dado el contexto y la consulta general.
        Devuelve un dict con 'estructura' e 'todo_list'.
        """
        from api.prompt_templates import get_wiki_structure_prompt
        prompt = get_wiki_structure_prompt(context, query)
        # Se asume que el modelo responde en JSON
        response = self.generate_wiki_markdown(prompt)
        return self._extract_json(response)

    def generate_section(self, context: str, section_title: str) -> str:
        """
        Usa el LLM para generar una sección de la wiki a partir del contexto relevante y el título de la sección.
        Devuelve el texto Markdown de la sección.
        """
        from api.prompt_templates import get_section_generation_prompt
        prompt = get_section_generation_prompt(context, section_title)
        return self.generate_wiki_markdown(prompt)

    def refine_wiki(self, wiki_draft: str, query: str) -> str:
        """
        Usa el LLM para revisar y mejorar la coherencia global de la wiki, agregando transiciones y asegurando formato Markdown/diagramas.
        Devuelve el texto final de la wiki en Markdown.
        """
        from api.prompt_templates import get_wiki_refinement_prompt
        prompt = get_wiki_refinement_prompt(wiki_draft, query)
        return self.generate_wiki_markdown(prompt)
    def generate_wiki_markdown(self, prompt: str) -> str:
        """
        Método base para generar una wiki en formato Markdown usando el LLM. Cada cliente debe implementarlo si requiere lógica especial.
        Por defecto, usa el mismo mecanismo que generate_documentation pero devuelve el texto plano (Markdown).
        """
        raise NotImplementedError("Este cliente LLM no implementa generación de wiki Markdown.")

    # --- JSON generation contract (must be overridden per provider to avoid prompt wrapping side-effects) ---
    def generate_json(self, prompt: str, expected_keys: list[str], attempts: int = 2) -> dict:
        """Generate JSON with retries and minimal key validation.

        Subclasses must override. expected_keys: list of keys that MUST appear in top-level JSON.
        """
        raise NotImplementedError("Subclasses must implement generate_json for direct JSON prompts.")

    def _generic_generate_documentation(self, prompt: str, file_path: str, doc_type: str, system_message: str, call_fn) -> dict:
        """
        Lógica común para todos los clientes LLM: ejecuta el prompt, limpia y parsea la respuesta, maneja errores y logging.
        call_fn debe ser una función que reciba (system_message, prompt) y devuelva el texto de respuesta del modelo.
        """
        try:
            text = call_fn(system_message, prompt)
            if not text:
                raise Exception("No response from model")
            cleaned = self._extract_json(text)

            if not isinstance(cleaned, dict):
                raise Exception("The cleaned variable is not a JSON (dict)")
            
            return cleaned
            
        except Exception as e:
            logging.error(f"Error en {self.__class__.__name__}.generate_documentation: {e}, respuesta del modelo: \n{text if 'text' in locals() else ''}")
            # Adjuntar la respuesta cruda al error para que app.py la pueda usar
            setattr(e, 'raw_response', text if 'text' in locals() else '')
            raise e

    def classify_llm_error(self, raw_response: str, error_message: str = "") -> str:
        """
        Usa el propio LLM para clasificar el error de la respuesta generada, devolviendo una breve explicación del fallo relevante para el parseo JSON.
        """
        # Prompt para clasificación de error
        prompt = (
            "Eres un asistente experto en análisis de errores de modelos de lenguaje. "
            "Tu tarea es leer la siguiente respuesta generada por un LLM, junto con el mensaje de error de parseo, y clasificar brevemente la causa del fallo en relación a la generación de un bloque JSON válido. "
            "Indica si el error es por: no ser JSON válido, contener etiquetas, estar vacío, tener repeticiones, comentarios, o cualquier otro problema relevante. "
            "Responde en una sola frase clara y concisa, en español, sin rodeos ni explicaciones adicionales.\n"
            f"Respuesta del LLM:\n{raw_response}\n"
            f"Mensaje de error de parseo (si aplica): {error_message}\n"
            "Clasificación del error:"
        )
        # Usar el propio LLM para clasificar (por defecto usa el mismo modelo y provider)
        # Se asume que el método generate_documentation puede recibir prompts arbitrarios
        try:
            result = self.generate_documentation(prompt, file_path="error_classification.txt", doc_type="superficial_summary")
            # Tomar solo el campo 'summary' si es dict, o el string si es string
            if isinstance(result, dict) and 'summary' in result:
                return result['summary']
            elif isinstance(result, str):
                return result.strip()
            else:
                return str(result)
        except Exception as e:
            return f"No se pudo clasificar el error automáticamente: {e}"
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name

    @staticmethod
    def _extract_json(text):
        """
        Extrae y limpia un bloque JSON de una respuesta de LLM, y convierte recursivamente los strings que sean JSON válidos en objetos Python.
        """
        import re
        # 1. Elimina etiquetas tipo <think>...</think> o similares
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if match:
            json_block = match.group(1).strip()
        else:
            # Busca el bloque JSON raíz más grande (el de mayor longitud)
            def find_largest_json_block(s):
                stack = []
                start = None
                best = (None, None)
                for i, c in enumerate(s):
                    if c == '{':
                        if not stack:
                            start = i
                        stack.append('{')
                    elif c == '}':
                        if stack:
                            stack.pop()
                            if not stack and start is not None:
                                # Candidato a bloque JSON
                                if best[0] is None or (i - start) > (best[1] - best[0]):
                                    best = (start, i)
                                    
                    elif c == '[':
                        if not stack:
                            start = i
                        stack.append('[')
                    elif c == ']':
                        if stack:
                            stack.pop()
                            if not stack and start is not None:
                                # Candidato a bloque JSON
                                if best[0] is None or (i - start) > (best[1] - best[0]):
                                    best = (start, i)            
                if best[0] is not None and best[1] is not None:
                    return s[best[0]:best[1]+1]
                return s.strip()
            json_block = find_largest_json_block(text)

        # 3. Elimina todo antes del primer { o [
        json_block = re.sub(r'^.*?([\{\[])', r'\1', json_block, flags=re.DOTALL)

        # 4. Elimina comentarios tipo // ... (solo fuera de strings)
        def remove_json_comments(s):
            pattern = r'("(?:\\.|[^"\\])*"|//.*?$)'
            def repl(m):
                if m.group(1).startswith('"'):
                    return m.group(1)
                else:
                    return ''
            return re.sub(pattern, repl, s, flags=re.MULTILINE)
        json_block = remove_json_comments(json_block)

        # 5. Elimina repeticiones exactas consecutivas de bloques JSON
        json_block = re.sub(r'(\{[\s\S]+?\})(?:\s*\1)+', r'\1', json_block)

        # 6. Limpia saltos de línea no escapados dentro de strings
        def _clean_json_newlines(json_str):
            def replace_newlines_in_string(match):
                s = match.group(0)
                s = s.replace('\\\n', '<<TEMP_ESCAPED_NL>>')
                s = s.replace('\n', '\\n')
                s = s.replace('<<TEMP_ESCAPED_NL>>', '\\n')
                s = s.replace('\r', '')
                return s
            string_regex = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
            return re.sub(string_regex, replace_newlines_in_string, json_str)
        json_block = _clean_json_newlines(json_block)

        # 7. Intenta parsear y corregir el JSON
        try:
            data = _json.loads(json_block)
            return LlmClient._parse_json_strings(data)
        except _json.JSONDecodeError as e:
            # Corrección 1: Elimina la última coma si es la causa del error
            if "Expecting value" in str(e) and json_block.endswith(','):
                json_block = json_block.rstrip(',')
                try:
                    return LlmClient._parse_json_strings(_json.loads(json_block))
                except Exception:
                    pass
            # Corrección 2: Intenta reparar strings no cerradas
            if "Unterminated string literal" in str(e):
                json_block += '"'
                try:
                    return LlmClient._parse_json_strings(_json.loads(json_block))
                except Exception:
                    pass
            # Corrección 3: Remueve caracteres de escape incorrectos
            if "Invalid escape sequence" in str(e):
                json_block = re.sub(r'\\(.)', r'\1', json_block)
                try:
                    return LlmClient._parse_json_strings(_json.loads(json_block))
                except Exception:
                    pass
            # Corrección 4: Intenta agregar llaves faltantes al inicio o final
            if "Expecting property name enclosed in double quotes" in str(e):
                if not json_block.startswith('{') and not json_block.startswith('['):
                    json_block = '{' + json_block
                if not json_block.endswith('}') and not json_block.endswith(''):
                    json_block = json_block + '}'
                try:
                    return LlmClient._parse_json_strings(_json.loads(json_block))
                except Exception:
                    pass
            # Si ninguna corrección funciona
            logging.warning(f"No se pudo parsear JSON. Devolviendo texto para depuración. Error: {e}")
            logging.warning(json_block)
            return json_block

    @staticmethod
    def _parse_json_strings(obj):
        """
        Recorre recursivamente un dict/list y convierte los strings que sean JSON válidos en objetos Python.
        """
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    v_strip = v.strip()
                    # Solo intentamos si parece un JSON
                    if (v_strip.startswith('{') and v_strip.endswith('}')) or (v_strip.startswith('[') and v_strip.endswith(']')):
                        try:
                            obj[k] = LlmClient._parse_json_strings(_json.loads(v_strip))
                        except Exception:
                            obj[k] = v
                    else:
                        obj[k] = v
                elif isinstance(v, (dict, list)):
                    obj[k] = LlmClient._parse_json_strings(v)
            return obj
        elif isinstance(obj, list):
            return [LlmClient._parse_json_strings(i) for i in obj]
        else:
            return obj

    @abstractmethod
    def generate_documentation(self, code_chunk: str, file_path: str, doc_type: str) -> dict:
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> list[float]:
        pass

class GeminiClient(LlmClient):
    def _execute_with_retry(self, func, *args, **kwargs):
        max_retries = 3  # Define a maximum number of retries
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except google.api_core.exceptions.ResourceExhausted as e:
                if e.code == 429:
                    retry_delay = 60  # Default retry delay in seconds
                    if hasattr(e, 'violations'):
                        for violation in e.violations:
                            if violation.retry_delay and violation.retry_delay.seconds:
                                retry_delay = violation.retry_delay.seconds
                                break
                    logging.warning(f"Rate limit exceeded (attempt {attempt + 1}/{max_retries}). Retrying after {retry_delay} seconds...")
                    sleep(retry_delay)
                else:
                    logging.error(f"An error occurred during Gemini API call: {e}")
                    raise  # Re-raise the exception for non-rate-limit errors
        logging.error("Max retries reached. Gemini API call failed.")
        raise Exception("Gemini API call failed after multiple retries due to rate limits.")  # Or a custom exception
    
    def generate_wiki_markdown(self, prompt: str) -> str:
        # Options with optional deterministic seed
        opts = {
            "temperature": 0.1,
            "top_p": 0.5,
            "repeat_penalty": 1.1,
            "num_predict": 1200
        }
        try:
            seed_env = os.getenv("OLLAMA_SEED")
            if seed_env:
                opts["seed"] = int(seed_env)
        except Exception:
            pass
        # Add a brief system-style steer for concision and grounding
        steer = (
            "Eres un experto en documentación técnica. Responde en español, en Markdown. "
            "Sé conciso; no inventes; no incluyas 'Tabla de Contenidos' ni 'Target Audience'; "
            "cita fuentes del repo cuando des detalles con (ruta, chunk N)."
        )
        def generation_function(full_prompt):
            return self.model.generate_content(full_prompt).text

        return self._execute_with_retry(generation_function, f"{steer}\n{prompt}")
    
    def __init__(self, api_key: str, model_name: str):
        super().__init__(provider="google", model_name=model_name)
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_documentation(self, code_chunk: str, file_path: str, doc_type: str) -> dict:
        prompt = get_documentation_prompt(code_chunk, file_path, doc_type)
        def call_fn(system_message, prompt):
            def generation_function(system_message, prompt):
                return self.model.generate_content(f"{system_message}\n{prompt}").text
            return self._execute_with_retry(generation_function, system_message, prompt)
        
        system_message = "Eres un experto en documentación de código y responderás en español."
        return self._generic_generate_documentation(prompt, file_path, doc_type, system_message, call_fn)

    def generate_embedding(self, text: str) -> list[float]:
        # Gemini no soporta embeddings públicos, usar Ollama por defecto
        return OllamaClient(os.getenv("OLLAMA_URL", "http://localhost:11434"), "nomic-embed-text").generate_embedding(text)

    def generate_json(self, prompt: str, expected_keys: list[str], attempts: int = 2) -> dict:
        system_message_base = (
            "Eres un asistente experto en análisis y documentación de código. "
            "Debes responder SOLO un objeto JSON válido (sin markdown ni texto externo). Claves requeridas: {keys}. "
            "Si faltan claves, reintenta internamente y produce igualmente la mejor salida JSON posible."
        )
        for attempt in range(1, attempts + 1):
            try:
                system_message = system_message_base.format(keys=", ".join(expected_keys))
                def generation_function(system_message, prompt):
                    return self.model.generate_content(f"{system_message}\n{prompt}").text
                text = self._execute_with_retry(generation_function, system_message, prompt)

                cleaned = self._extract_json(text)
                if isinstance(cleaned, dict):
                    missing = [k for k in expected_keys if k not in cleaned]
                    if not missing:
                        return cleaned
                    logging.warning(f"GeminiClient.generate_json intento {attempt}: faltan claves {missing}")
                    # Añade claves faltantes vacías para próximo intento contexto
                    for k in missing:
                        cleaned[k] = ""
                    # Incrusta feedback en prompt para próximo intento
                    prompt = prompt + f"\nFALTABAN CLAVES: {missing}. AGREGA TODAS LAS CLAVES REQUERIDAS."  # noqa
                else:
                    logging.warning(f"GeminiClient.generate_json intento {attempt}: respuesta no dict")
            except Exception as e:
                logging.warning(f"GeminiClient.generate_json error intento {attempt}: {e}")
        return cleaned if isinstance(cleaned, dict) else {}

class OpenAIClient(LlmClient):
    def generate_wiki_markdown(self, prompt: str) -> str:
        client = OpenAI(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Eres un experto en documentación técnica y responderás en español usando Markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return completion.choices[0].message.content
    def __init__(self, api_key: str, model_name: str):
        super().__init__(provider="openai", model_name=model_name)
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def generate_documentation(self, code_chunk: str, file_path: str, doc_type: str) -> dict:
        prompt = get_documentation_prompt(code_chunk, file_path, doc_type)
        def call_fn(system_message, prompt):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2048
            )
            return response.choices[0].message.content
        system_message = "Eres un experto en documentación de código y responderás en español. Evita repetir frases y no uses etiquetas como <think> o <response>. Solo y únicamente responde con un bloque JSON válido. Si no puedes responder en JSON válido, responde solo un bloque JSON vacío: ```json {} ```, sin ningún texto adicional. Haz lo posible para brindar información válida y relevante."
        return self._generic_generate_documentation(prompt, file_path, doc_type, system_message, call_fn)

    def generate_embedding(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings.create(
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error en OpenAIClient.generate_embedding: {e}, respuesta del modelo: \n{text}")
            # fallback a Ollama
            return OllamaClient(os.getenv("OLLAMA_URL", "http://localhost:11434"), "nomic-embed-text").generate_embedding(text)

    def generate_json(self, prompt: str, expected_keys: list[str], attempts: int = 2) -> dict:
        base_system = (
            "Eres un asistente experto en análisis y documentación de código. "
            "Devuelve SOLO un objeto JSON válido sin markdown. Claves obligatorias: {keys}. "
            "No expliques."
        )
        cleaned = {}
        for attempt in range(1, attempts + 1):
            try:
                system_message = base_system.format(keys=", ".join(expected_keys))
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.05,
                    max_tokens=2200
                )
                text = completion.choices[0].message.content
                cleaned = self._extract_json(text)
                if isinstance(cleaned, dict):
                    missing = [k for k in expected_keys if k not in cleaned]
                    if not missing:
                        return cleaned
                    logging.warning(f"OpenAIClient.generate_json intento {attempt}: faltan {missing}")
                    prompt = prompt + f"\nFALTABAN CLAVES: {missing}. INCLUYELAS TODAS."  # noqa
                else:
                    logging.warning(f"OpenAIClient.generate_json intento {attempt}: respuesta no dict")
            except Exception as e:
                logging.warning(f"OpenAIClient.generate_json error intento {attempt}: {e}")
        return cleaned if isinstance(cleaned, dict) else {}


class GroqClient(LlmClient):
    def generate_wiki_markdown(self, prompt: str) -> str:
        client = Groq(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Eres un experto en documentación técnica y responderás en español usando Markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        return completion.choices[0].message.content
    def __init__(self, api_key: str, model_name: str):
        super().__init__(provider="groq", model_name=model_name)
        self.api_key = api_key
        self.client = Groq(api_key=api_key)

    def generate_documentation(self, code_chunk: str, file_path: str, doc_type: str) -> dict:
        prompt = get_documentation_prompt(code_chunk, file_path, doc_type)
        def call_fn(system_message, prompt):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4
            )
            sleep(5)
            return response.choices[0].message.content
        system_message = "Eres un experto en documentación de código y responderás en español."
        return self._generic_generate_documentation(prompt, file_path, doc_type, system_message, call_fn)

    def generate_embedding(self, text: str) -> list[float]:
        # Groq no soporta embeddings, usar Ollama por defecto
        return OllamaClient(os.getenv("OLLAMA_URL", "http://localhost:11434"), "nomic-embed-text").generate_embedding(text)

    def generate_json(self, prompt: str, expected_keys: list[str], attempts: int = 2) -> dict:
        base_system = (
            "Eres un asistente experto en análisis y documentación de código. "
            "Devuelve SOLO JSON válido (sin markdown). Claves obligatorias: {keys}."
        )
        cleaned = {}
        for attempt in range(1, attempts + 1):
            try:
                system_message = base_system.format(keys=", ".join(expected_keys))
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                text = completion.choices[0].message.content
                cleaned = self._extract_json(text)
                if isinstance(cleaned, dict):
                    missing = [k for k in expected_keys if k not in cleaned]
                    if not missing:
                        return cleaned
                    logging.warning(f"GroqClient.generate_json intento {attempt}: faltan {missing}")
                    prompt = prompt + f"\nFALTABAN CLAVES: {missing}. AGREGA TODAS."
                else:
                    logging.warning(f"GroqClient.generate_json intento {attempt}: respuesta no dict")
            except Exception as e:
                logging.warning(f"GroqClient.generate_json error intento {attempt}: {e}")
        return cleaned if isinstance(cleaned, dict) else {}


class OllamaClient(LlmClient):
    def generate_wiki_markdown(self, prompt: str) -> str:
        chat_opts = {
            "temperature": 0.0,
            "top_p": 0.4,
            "repeat_penalty": 1.15,
            "num_predict": 1200
        }
        try:
            seed_env = os.getenv("OLLAMA_SEED")
            if seed_env:
                chat_opts["seed"] = int(seed_env)
        except Exception:
            pass
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Eres un experto en documentación técnica y responderás en español usando Markdown."},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            options=chat_opts
        )
        return response['message']['content']
    def __init__(self, ollama_url: str, model_name: str):
        super().__init__(provider="ollama", model_name= model_name)
        self.ollama_url = ollama_url
        self.client = OllamaClientLib(host=ollama_url)

    def generate_documentation(self, code_chunk: str, file_path: str, doc_type: str) -> dict:
        prompt = get_documentation_prompt(code_chunk, file_path, doc_type)
        def call_fn(system_message, prompt):
            chat_opts = {
                "temperature": 0.0,
                "top_p": 0.4,
                "repeat_penalty": 1.15,
                "num_predict": 1200
            }
            try:
                seed_env = os.getenv("OLLAMA_SEED")
                if seed_env:
                    chat_opts["seed"] = int(seed_env)
            except Exception:
                pass
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                options=chat_opts
            )
            return response['message']['content']
        system_message = "Eres un experto en documentación de código y responderás en español."
        return self._generic_generate_documentation(prompt, file_path, doc_type, system_message, call_fn)

    def generate_embedding(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings(
                prompt=text,
                model=self.model_name
            )
            return response['embedding']
        except Exception as e:
            logging.error(f"Error en OllamaClient.generate_embedding: {e}, respuesta del modelo: \n{text}")
            raise LlmError(f"Error en embedding Ollama: {e}")

    def generate_json(self, prompt: str, expected_keys: list[str], attempts: int = 2) -> dict:
        base_system = (
            "Eres un asistente experto en análisis y documentación de código. "
            "Devuelve SOLO un objeto JSON válido (sin backticks). Claves obligatorias: {keys}."
        )
        cleaned = {}
        temp = 0.5
        for attempt in range(1, attempts + 1):
            try:
                temp = 0.0 if temp < 0.0 else temp
                system_message = base_system.format(keys=", ".join(expected_keys))

                gen_opts = {"temperature": temp}
                try:
                    seed_env = os.getenv("OLLAMA_SEED")
                    if seed_env:
                        gen_opts["seed"] = int(seed_env)
                except Exception:
                    pass
                response = self.client.generate(
                    model=self.model_name,
                    prompt=f"{system_message}\n{prompt}",
                    stream=False,
                    options=gen_opts
                )

                text = response['message']['content']
                cleaned = self._extract_json(text)
                if isinstance(cleaned, dict):
                    missing = [k for k in expected_keys if k not in cleaned]
                    if not missing:
                        return cleaned
                    logging.warning(f"OllamaClient.generate_json intento {attempt}: faltan {missing}")
                    prompt = prompt + f"\nFALTABAN CLAVES: {missing}. AGREGA TODAS."
                else:
                    logging.warning(f"OllamaClient.generate_json intento {attempt}: respuesta no dict")
                    base_system += f" Este es el intento {attempt}: Sigue estrictamente el formato JSON proporcionado"
                    temp -= 0.25
            except Exception as e:
                logging.warning(f"OllamaClient.generate_json error intento {attempt}: {e}")
                base_system += f" Este es el intento {attempt}: Anteriormente tuviste el error {e}, consideralo en los proximos intentos"
                temp -= 0.25
        return cleaned if isinstance(cleaned, dict) else {}


def create_llm_client(provider: str, model_name: str, ollama_url: str = None) -> LlmClient:
    """
    Crea y devuelve una instancia del cliente LLM apropiado.
    """
    provider = provider.lower()

    client_classes = {
        "ollama": OllamaClient,
        "google": GeminiClient,
        "openai": OpenAIClient,
        "groq": GroqClient,
    }

    if provider == "ollama":
        if not ollama_url:
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        return OllamaClient(ollama_url=ollama_url, model_name=model_name)
    else:
        api_key = os.getenv(f"{provider.upper()}_API_KEY")
        if not api_key:
            logging.error(f"Error: La clave API para {provider} no está configurada.")
            raise LlmError(f"La clave API para {provider} no está configurada.")

        ClientClass = client_classes.get(provider)
        if ClientClass:
            return ClientClass(api_key, model_name=model_name)

    raise LlmError(f"Proveedor de LLM '{provider}' no soportado.")
