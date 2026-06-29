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


def read_static_processed_documents(embedder: Embedder) -> None:
    static_processed_docs_dir = os.getenv("STATIC_PROCESSED_DOCS_DIR", "./docs/static/processed")
    if not os.path.exists(static_processed_docs_dir):
        print("No se detectó carpeta de documentos estáticos procesados")
        return
    chunks = embedder.read_text_documents(static_processed_docs_dir)
    embedder.embed_and_store(chunks = chunks, database_name = "static")
    print("Documentos estáticos procesados leídos")


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


def read_web_documents(embedder: Embedder) -> None:
    web_docs_file = os.getenv("WEB_DOCS_FILE")
    chunks = embedder.read_web_documents(web_docs_file)
    embedder.embed_and_store(chunks = chunks, database_name = "static")
    print("Documentos web leídos")

def read_curated_documents(embedder: Embedder) -> None:
    curated_docs_dir = os.getenv("CURATED_DOCS_DIR", "./docs/curated")
    if not os.path.exists(curated_docs_dir):
        print("No se detectó carpeta de documentos curados")
        return
    chunks = embedder.read_text_documents(curated_docs_dir)
    embedder.embed_and_store(chunks = chunks, database_name = "instructions")
    print("Documentos curados leídos")


import discord

def run_server(llm: LanguageModel) -> None:
    while True:
        query = input("Escribe un mensaje: ")
        if query == "quit": break
        if query == "": continue

        print(llm.generate_response(pregunta = query))
        print()

    print("Fin.")

def run_discord_bot(llm: LanguageModel) -> None:
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f'Hemos iniciado sesión en Discord como {client.user}')
        print("Modelo listo para responder preguntas!")

    @client.event
    async def on_message(message):
        # Evitamos que el bot se responda a sí mismo
        if message.author == client.user:
            return

        # Solo respondemos si el bot fue mencionado
        if client.user not in message.mentions:
            return
        
        async with message.channel.typing():
            try:
                # Generar respuesta usando la RAG chain
                response = llm.generate_response(pregunta=message.content)
                
                # Discord tiene un límite de 2000 caracteres por mensaje.
                # El primer bloque responde al mensaje original sin mencionar al autor.
                if len(response) > 2000:
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    await message.reply(chunks[0], mention_author=False)
                    for chunk in chunks[1:]:
                        await message.channel.send(chunk)
                else:
                    await message.reply(response, mention_author=False)
            except Exception as e:
                print(f"Error al generar respuesta: {e}")
                await message.reply("Ocurrió un error al intentar generar una respuesta.", mention_author=False)

    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("ERROR: No se encontró DISCORD_TOKEN en las variables de entorno.")
    else:
        client.run(token)

def setup_chatbot() -> LanguageModel:
    embedder_model_name = os.getenv("EMBEDDER_MODEL_NAME")
    llm_model_name = os.getenv("LLM_MODEL_NAME")
    db_dir = os.getenv("DB_DIR")
    print("Variables de entorno leídas...")

    with open(os.getenv("SYSTEM_PROMPT_FILE"), "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print("Prompt del sistema leído...")
    
    embedder = Embedder(model_name = embedder_model_name, database_path = db_dir, chunk_size = 1000, chunk_overlap = 200)
    
    if not os.path.exists(db_dir) or len(os.listdir(db_dir)) == 0:
        print(f"No se detectó base de datos en '{db_dir}'. Generando embeddings...")
        read_static_documents(embedder)
        read_static_processed_documents(embedder)
        read_dynamic_documents(embedder)
        read_curated_documents(embedder)
        read_historical_documents(embedder)
        read_web_documents(embedder)
        print("Embeddings generados exitosamente.")
    else:
        print(f"Base de datos detectada en '{db_dir}'.")

    llm = LanguageModel(model_name = llm_model_name, initial_prompt = system_prompt, temperature = 0.08)
    llm.define_rag_chain(embedder.get_retriever(k = 2, static_weight = 0.4, dynamic_weight = 0.4,
                                                historical_weight = 0.05, instructions_weight = 0.15))
    return llm

def main() -> None:
    print("Inicializando el modelo...")
    llm = setup_chatbot()
    
    # Elegir el modo de ejecución:
    # run_server(llm) # Para ejecutar por consola
    run_discord_bot(llm) # Para ejecutar el bot de Discord

if __name__ == "__main__":
    load_dotenv()
    main()
