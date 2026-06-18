from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, CSVLoader
from langchain_core.documents.base import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


"""
Este modulo contiene la clase 'Embedder', la cual se encargará de la creación 
de los vectores de embedding a partir de un modelo pre-entrenado. Luego, 
devuelve los vectores de embedding
"""
class Embedder(object):
    
    def __init__(self, model_name: str, database_path: str, chunk_size: int = 500, chunk_overlap: int = 100,
                 static_db_name: str = "static", dynamic_db_name:str = "dynamic"):
        assert model_name is not None, "Model name cannot be None"
        assert database_path is not None, "Database path cannot be None"

        try:
            __loaded_model = HuggingFaceEmbeddings(model_name = model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        self.model: HuggingFaceEmbeddings = __loaded_model
        self.database_path: str = database_path
        self.text_splitter: RecursiveCharacterTextSplitter = \
            RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        self.static_db_name: str = static_db_name
        self.dynamic_db_name: str = dynamic_db_name


    def _get_vector_store(self, collection_name: str) -> Chroma:
        if collection_name not in [self.static_db_name, self.dynamic_db_name]:
            raise ValueError(f"Collection name must be either '{self.static_db_name}' or '{self.dynamic_db_name}'")
        
        return Chroma(persist_directory = self.database_path, 
                      embedding_function = self.model,
                      collection_name = collection_name)


    def read_pdf_documents(self, documents_path: str) -> list[Document]:
        assert documents_path is not None, "Documents path cannot be None"

        loader = PyPDFDirectoryLoader(documents_path)    
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError("No documents found in the specified path.")

        return self._create_chunks(documents)
    
    def read_csv_documents(self, documents_path: str) -> list[Document]:
        assert documents_path is not None, "Documents path cannot be None"

        loader = DirectoryLoader(documents_path, glob = "**/*.csv", loader_cls = CSVLoader)
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError("No documents found in the specified path.")
        
        return self._create_chunks(documents)

    def read_message_from_discord(self, message: dict) -> list[Document]:
        assert message is not None, "Message cannot be None"

        pregunta = message.get("pregunta", "").strip()
        respuesta = message.get("respuesta", "").strip()

        if not pregunta or not respuesta:
            raise ValueError("Both 'pregunta' and 'respuesta' must be provided in the message.")

        page_content = f"Pregunta del alumno: {pregunta}\nRespuesta del director de carrera: {respuesta}"

        metadata = {
            "source": "discord",
            "type": "question_answer",
        }

        doc = Document(page_content = page_content, metadata = metadata)

        return self._create_chunks([doc])

    def _create_chunks(self, documents: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(documents)
    

    def embed_and_store(self, chunks: list[Document], is_dynamic: bool = True) -> None:
        assert chunks is not None, "Chunks cannot be None"

        collection_name = self.dynamic_db_name if is_dynamic else self.static_db_name
        vector_store = self._get_vector_store(collection_name)
        
        vector_store.add_documents(documents = chunks)
    

    def get_retriever(self, k: int = 3, static_weight: float = 0.5, dynamic_weight: float = 0.5) -> EnsembleRetriever:
        assert k > 0, "k must be greater than 0"
        assert 0 <= static_weight <= 1, "static_weight must be between 0 and 1"
        assert 0 <= dynamic_weight <= 1, "dynamic_weight must be between 0 and 1"
        assert static_weight + dynamic_weight == 1, "static_weight and dynamic_weight must sum to 1"

        # Obtnemos el retriever estático
        vector_store_static = self._get_vector_store(self.static_db_name)
        retriever_static = vector_store_static.as_retriever(search_type = "similarity", search_kwargs = {"k": k})

        # Obtenemos el retriever dinámico
        vector_store_dynamic = self._get_vector_store(self.dynamic_db_name)
        retriever_dynamic = vector_store_dynamic.as_retriever(search_type = "similarity", search_kwargs = {"k": k})

        # Combinamos ambos
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_static, retriever_dynamic],
            weights=[static_weight, dynamic_weight] # 50% de importancia a cada base de datos. Puedes ajustarlo.
        )
        return ensemble_retriever
    

    def reset_dynamic_database(self) -> None:
        # TODO: Agregar una confirmación antes de resetear la base de datos dinámica
        vector_store_dynamic = self._get_vector_store(self.dynamic_db_name)
        vector_store_dynamic.delete_collection()
        print("La base de datos dinámica ha sido reseteada.")