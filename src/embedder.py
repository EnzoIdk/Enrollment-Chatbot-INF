import json
import re
import unicodedata

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, WebBaseLoader, PyPDFLoader, TextLoader
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from src.text_cleaner import TextCleaner

"""
Este modulo contiene la clase 'Embedder', la cual se encargará de la creación 
de los vectores de embedding a partir de un modelo pre-entrenado. Luego, 
devuelve los vectores de embedding
"""
class Embedder(object):
    
    def __init__(self, model_name: str, database_path: str, chunk_size: int = 500, chunk_overlap: int = 100,
                 static_db_name: str = "static", historical_db_name: str = "historical",
                 dynamic_db_name: str = "dynamic", instructions_db_name: str = "instructions",
                 min_chunk_length: int = 30):
        assert model_name is not None, "Model name cannot be None"
        assert database_path is not None, "Database path cannot be None"

        try:
            __loaded_model = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = {"device": "cpu"})
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        self.model: HuggingFaceEmbeddings = __loaded_model
        self.database_path: str = database_path
        self.text_splitter: RecursiveCharacterTextSplitter = \
            RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        self.text_cleaner: TextCleaner = TextCleaner(min_chunk_length = min_chunk_length)
        self.static_db_name: str = static_db_name
        self.historical_db_name: str = historical_db_name
        self.dynamic_db_name: str = dynamic_db_name
        self.instructions_db_name: str = instructions_db_name


    def read_pdf_documents(self, documents_path: str) -> list[Document]:
        assert documents_path is not None, "Documents path cannot be None"

        pdf_files = sorted(Path(documents_path).glob("**/*.[pP][dD][fF]"))
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())

        if len(documents) == 0:
            raise ValueError("No documents found in the specified path.")

        cleaned_documents = self.text_cleaner.clean(documents)
        return self._create_chunks(cleaned_documents)

    def read_csv_documents(self, documents_path: str) -> list[Document]:
        assert documents_path is not None, "Documents path cannot be None"

        loader = DirectoryLoader(documents_path, glob = "**/*.csv", loader_cls = CSVLoader)
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError("No documents found in the specified path.")
        
        return self._create_chunks(documents)
    
    
    def read_text_documents(self, documents_path: str) -> list[Document]:
        assert documents_path is not None, "Documents path cannot be None"

        text_files = [
            path for pattern in ("**/*.md", "**/*.txt", "**/*.json")
            for path in Path(documents_path).glob(pattern)
        ]
        documents = []
        for text_file in sorted(text_files):
            loader = TextLoader(str(text_file), encoding="utf-8")
            documents.extend(loader.load())

        if len(documents) == 0:
            raise ValueError("No documents found in the specified path.")

        return self._create_chunks(documents)


    def read_json_documents(self, documents_path: str) -> list[Document]:
        assert documents_path is not None, "Documents path cannot be None"

        json_files = list(Path(documents_path).glob("**/*.json"))

        if len(json_files) == 0:
            raise ValueError("No documents found in the specified path.")

        documents = []

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            records = data if isinstance(data, list) else [data]

            for record in records:
                doc = self._build_qa_document(record, source = str(json_file))
                if doc is None: 
                    print(f"Se ha omitido un mensaje del archivo: {json_file}")
                    print(record)
                    continue
                documents.extend(doc)

        return self._create_chunks(documents)


    def read_web_documents(self, file_path: str) -> list[Document]:
        assert file_path is not None, "File path cannot be None"

        with open(file_path, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]

        if not urls:
            raise ValueError("No URLs found in the specified file.")

        loader = WebBaseLoader(urls)
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError("No documents found in the specified URLs.")

        return self._create_chunks(documents)


    def _build_qa_document(self, record: dict, source: str = "json") -> list[Document]:
        pregunta = record.get("pregunta", "").strip()
        respuesta = record.get("respuesta", "").strip()
        ciclo = record.get("ciclo", "").strip()

        if not pregunta or not respuesta or not ciclo:
            print("Se ha omitido un documento")
            return None
            # raise ValueError("All fields 'pregunta', 'respuesta', and 'ciclo' must be provided.")

        course_context = self._build_course_alias_context(pregunta, respuesta)
        page_content = (
            f"Ciclo: {ciclo}\n"
            f"Pregunta del alumno: {pregunta}\n"
            f"Respuesta del director de carrera: {respuesta}"
        )
        if course_context:
            page_content += f"\n{course_context}"

        metadata = {
            "source": source,
            "type": "question_answer",
            "ciclo": ciclo
        }

        return [Document(page_content = page_content, metadata = metadata)]


    def _build_course_alias_context(self, *texts: str) -> str:
        aliases = self._load_course_alias_records()
        if not aliases:
            return ""

        normalized_text = self._normalize_for_alias_matching(" ".join(texts))
        matches = []
        seen = set()
        for record in aliases:
            alias = record.get("alias", "")
            name = record.get("nombre", "")
            code = record.get("codigo", "")
            if not alias or not name:
                continue
            terms = {alias, name}
            if not any(self._contains_alias_term(normalized_text, term) for term in terms):
                continue
            key = (self._normalize_for_alias_matching(name), code)
            if key in seen:
                continue
            seen.add(key)
            code_text = f" ({code})" if code else ""
            matches.append(f"- {alias} => {name}{code_text}")

        if not matches:
            return ""
        return "Vocabulario de cursos detectado en esta conversación:\n" + "\n".join(matches)


    def _load_course_alias_records(self) -> list[dict]:
        path = Path("docs/curated/vocabulario_cursos.json")
        if not path.exists():
            return []
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]


    def _normalize_for_alias_matching(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", normalized.lower())
        return re.sub(r"\s+", " ", normalized).strip()


    def _contains_alias_term(self, normalized_text: str, term: str) -> bool:
        normalized_term = self._normalize_for_alias_matching(term)
        if not normalized_term:
            return False
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", normalized_text) is not None


    def _valid_collection_names(self) -> list[str]:
        return [
            self.static_db_name,
            self.historical_db_name,
            self.dynamic_db_name,
            self.instructions_db_name,
        ]

    def _get_vector_store(self, collection_name: str) -> Chroma:
        if collection_name not in self._valid_collection_names():
            raise ValueError(
                f"Collection name must be one of {self._valid_collection_names()}"
            )
        
        return Chroma(persist_directory = self.database_path, 
                      embedding_function = self.model,
                      collection_name = collection_name)


    def read_messages_from_discord(self, qa_message: dict) -> list[Document]:
        assert qa_message is not None, "QA - Message cannot be None"

        documents = self._build_qa_document(qa_message, source = "discord")

        return self._create_chunks(documents)


    def _create_chunks(self, documents: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(documents)
    

    def embed_and_store(self, chunks: list[Document], database_name: str) -> None:
        assert chunks is not None, "Chunks cannot be None"
        assert database_name in self._valid_collection_names(), \
            f"Database name must be one of {self._valid_collection_names()}"

        vector_store = self._get_vector_store(database_name)
        vector_store.add_documents(documents = chunks)
    
    def get_retriever(self, k: int = 3, static_weight: float = 0.4, dynamic_weight: float = 0.4,
                      historical_weight: float = 0.05, instructions_weight: float = 0.15) -> EnsembleRetriever:
        assert k > 0, "k must be greater than 0"
        assert 0 <= static_weight <= 1, "static_weight must be between 0 and 1"
        assert 0 <= dynamic_weight <= 1, "dynamic_weight must be between 0 and 1"
        assert 0 <= historical_weight <= 1, "historical_weight must be between 0 and 1"
        assert 0 <= instructions_weight <= 1, "instructions_weight must be between 0 and 1"
        # assert static_weight + dynamic_weight + historical_weight + instructions_weight == 1.0, "Weights must sum to 1"

        # Obtnemos el retriever estático
        vector_store_static = self._get_vector_store(self.static_db_name)
        retriever_static = vector_store_static.as_retriever(search_type = "mmr", search_kwargs = {"k": k, "fetch_k": k * 2})

        # Obtenemos el retriever dinámico
        vector_store_dynamic = self._get_vector_store(self.dynamic_db_name)
        retriever_dynamic = vector_store_dynamic.as_retriever(search_type = "mmr", search_kwargs = {"k": k, "fetch_k": k * 2})

        # Obtenemos el retriever histórico
        vector_store_historical = self._get_vector_store(self.historical_db_name)
        retriever_historical = vector_store_historical.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 2})

        # Obtenemos el retriever de instrucciones curadas
        vector_store_instructions = self._get_vector_store(self.instructions_db_name)
        retriever_instructions = vector_store_instructions.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 2})

        # Combinamos los retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_static, retriever_dynamic, retriever_historical, retriever_instructions],
            weights=[static_weight, dynamic_weight, historical_weight, instructions_weight]
        )
        return ensemble_retriever
    

    def reset_dynamic_database(self) -> None:
        # TODO: Agregar una confirmación antes de resetear la base de datos dinámica
        vector_store_dynamic = self._get_vector_store(self.dynamic_db_name)
        vector_store_dynamic.delete_collection()
        print("La base de datos dinámica ha sido reseteada.")
