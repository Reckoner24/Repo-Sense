# vector_db_manager.py

import os 
import chromadb
import logging
import numpy as np

class VectorDBManager:
    """Clase para gestionar la base de datos vectorial ChromaDB o LangChain."""

    def __init__(self, collection_name="repo_documentation", embedding_client=None, persist_directory=None):
        """
        Inicializa el cliente de ChromaDB usando la nueva API persistente (DuckDB backend).
        Args:
            collection_name (str): El nombre de la colección donde se almacenará la documentación.
            embedding_client: Cliente de embeddings (opcional).
            persist_directory (str): Directorio donde se almacenarán los datos persistentes de ChromaDB (único para todas las colecciones).
        """
        self.collection_name = collection_name
        self.embedding_client = embedding_client
        # Directorio de persistencia único para todas las colecciones
        if persist_directory is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chroma_collections"))
        else:
            base_dir = os.path.abspath(persist_directory)
        os.makedirs(base_dir, exist_ok=True)
        self.persist_directory = base_dir
        # Nueva API: PersistentClient
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logging.info(f"Base de datos vectorial ChromaDB inicializada con la colección: '{collection_name}' en '{self.persist_directory}' (PersistentClient)")

    def is_empty(self):
        """
        Verifica si la colección de la base de datos está vacía.
        Returns:
            bool: True si la colección no contiene documentos, False en caso contrario.
        """
        
        return self.collection.count() == 0

    def add_documents(self, documents):
        """
        Añade documentos (chunks de código y documentación) a la base de datos.
        Args:
            documents (list): Una lista de diccionarios, donde cada diccionario contiene 'id', 'embedding' y 'metadata'.
        """
        if not documents:
            return
        
        ids = [doc["id"] for doc in documents]
        embeddings = [doc["embedding"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        try:
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logging.info(f"Se agregaron {len(documents)} documentos a la base de datos.")
    
            # Persistencia automática: no es necesario llamar a persist() en versiones recientes de ChromaDB
            if hasattr(self.collection, 'persist'):
                self.collection.persist()
        except Exception as e:
            logging.error(f"Error al agregar documentos a ChromaDB: {e}")

    def query(self, query_embedding, n_results=5):
        """
        Busca documentos similares en la base de datos e incluye las distancias de similitud.
        Args:
            query_embedding (list o str): El vector de embedding de la consulta (ChromaDB).
            n_results (int): El número de resultados a devolver.
        Returns:
            dict o list: Los resultados de la consulta, incluyendo documentos, metadatos y distancias (ChromaDB) o lista de documentos (LangChain).
        """

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
            # Normalizar distancias por consulta usando percentiles p10–p90 y clipping [0,1]
            try:
                dist_lists = results.get('distances') or []
            except Exception:
                dist_lists = []
            norm_lists = []
            for dists in dist_lists or []:
                try:
                    dists = [float(d) for d in (dists or [])]
                    if not dists:
                        norm_lists.append([])
                        continue
                    dmin = float(min(dists))
                    dmax = float(max(dists))
                    p10 = float(np.percentile(dists, 10))
                    p90 = float(np.percentile(dists, 90))
                    denom = (p90 - p10) if p90 > p10 else (dmax - dmin) if dmax > dmin else 1.0
                    norm = [max(0.0, min(1.0, (float(d) - p10) / denom)) for d in dists]
                except Exception:
                    # Fallback min-max si falla percentil
                    try:
                        dmin = float(min(dists))
                        dmax = float(max(dists))
                        denom = (dmax - dmin) if dmax > dmin else 1.0
                        norm = [max(0.0, min(1.0, (float(d) - dmin) / denom)) for d in dists]
                    except Exception:
                        norm = []
                norm_lists.append(norm)
            if norm_lists:
                results['norm_distances'] = norm_lists
            return results
        
        except Exception as e:
            logging.error(f"Error al consultar la base de datos: {e}")
            return None

    def get_by_ids(self, ids, include_embeddings: bool = True):
        """Obtiene documentos por ids exactos (si existen)."""
        if not ids:
            return {}
        try:
            include = ["metadatas"]
            if include_embeddings:
                include.append("embeddings")
            data = self.collection.get(ids=ids, include=include)
            return data
        except Exception as e:
            logging.warning(f"get_by_ids error: {e}")
            return {}

    def get_all(self, include_embeddings: bool = False, limit: int | None = None):
        """Devuelve todos los documentos (cuidado en colecciones grandes)."""
        try:
            include = ["metadatas"]
            if include_embeddings:
                include.append("embeddings")
            data = self.collection.get(include=include)
            # Limitar si se solicita
            if limit is not None and data and data.get('ids'):
                data['ids'] = data['ids'][:limit]
                if 'metadatas' in data:
                    data['metadatas'] = data['metadatas'][:limit]
                if include_embeddings and 'embeddings' in data:
                    data['embeddings'] = data['embeddings'][:limit]
            return data
        except Exception as e:
            logging.warning(f"get_all error: {e}")
            return {}
