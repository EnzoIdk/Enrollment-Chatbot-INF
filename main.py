from src.embedder import Embedder
from src.language_model import LanguageModel

from dotenv import load_dotenv
import os
import time


# Funciones para el main
def read_static_documents(embedder: Embedder) -> None:
    static_docs_dir = os.getenv("STATIC_DOCS_DIR")
    chunks = embedder.read_pdf_documents(static_docs_dir)
    embedder.embed_and_store(chunks = chunks, database_name = "static")
    print("Documentos estáticos leídos")


def read_dynamic_documents(embedder: Embedder) -> None:
    dynamic_docs_dir = os.getenv("DYNAMIC_DOCS_DIR")
    chunks = embedder.read_pdf_documents(dynamic_docs_dir)
    embedder.embed_and_store(chunks = chunks, database_name = "dynamic")
    print("Documentos dinámicos leídos")


def read_historical_documents(embedder: Embedder) -> None:
    historical_docs_dir = os.getenv("HISTORICAL_DOCS_DIR")
    chunks = embedder.read_json_documents(historical_docs_dir)
    embedder.embed_and_store(chunks = chunks, database_name = "historical")
    print("Documentos históricos leídos")


def run_server(llm: LanguageModel) -> None:
    while True:
        query = input("Escribe un mensaje: ")
        if query == "quit": break
        if query == "": continue

        print(llm.generate_response(pregunta = query))

    print("Fin.")



def main() -> None:
    embedder_model_name = os.getenv("EMBEDDER_MODEL_NAME")
    llm_model_name = os.getenv("LLM_MODEL_NAME")
    db_dir = os.getenv("DB_DIR")
    print("Variables de entorno leídas...")

    with open(os.getenv("SYSTEM_PROMPT_FILE"), "r") as f:
        system_prompt = f.read()
    print("Prompt del sistema leído...")
    
    embedder = Embedder(model_name = embedder_model_name, database_path = db_dir)
    #read_static_documents(embedder)
    #read_dynamic_documents(embedder)
    #read_historical_documents(embedder)

    llm = LanguageModel(model_name = llm_model_name, initial_prompt = system_prompt)
    llm.define_rag_chain(embedder.get_retriever(k = 3, static_weight = 0.3, dynamic_weight = 0.6, 
                                                historical_weight = 0.1))
    
    run_server(llm)


if __name__ == "__main__":
    load_dotenv()
    input("Press any key to start the chatbot: ")
    main()
