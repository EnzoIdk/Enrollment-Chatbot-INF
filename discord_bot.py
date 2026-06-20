import discord
import os
from dotenv import load_dotenv
from main import setup_chatbot

# Configuramos los intents requeridos (necesitas habilitar Message Content Intent en el portal de Discord)
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

# Variable global para mantener el modelo cargado
llm = None

@client.event
async def on_ready():
    global llm
    print(f'Hemos iniciado sesión en Discord como {client.user}')
    print("Inicializando el modelo...")
    llm = setup_chatbot()
    print("Modelo listo para responder preguntas!")

@client.event
async def on_message(message):
    # Evitamos que el bot se responda a sí mismo
    if message.author == client.user:
        return

    # Si el bot fue mencionado o es un mensaje directo, o simplemente cualquier mensaje (default actual)
    # Por ahora respondemos a cualquier mensaje de un usuario
    
    # Nos aseguramos que el modelo haya terminado de cargar
    if llm is None:
        await message.channel.send("El modelo aún se está inicializando. Por favor espera un momento.")
        return

    async with message.channel.typing():
        try:
            # Generar respuesta usando la RAG chain
            response = llm.generate_response(pregunta=message.content)
            
            # Discord tiene un límite de 2000 caracteres por mensaje
            # Si la respuesta es muy larga, la partimos
            if len(response) > 2000:
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i:i+2000])
            else:
                await message.channel.send(response)
        except Exception as e:
            print(f"Error al generar respuesta: {e}")
            await message.channel.send("Ocurrió un error al intentar generar una respuesta.")

if __name__ == "__main__":
    load_dotenv()
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("ERROR: No se encontró DISCORD_TOKEN en las variables de entorno.")
    else:
        client.run(token)
