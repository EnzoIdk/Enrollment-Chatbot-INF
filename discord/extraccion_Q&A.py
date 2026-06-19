import discord
import json
import nest_asyncio
import os
import datetime
from dotenv import load_dotenv

# Configuración inicial
nest_asyncio.apply()

# Forzamos a encontrar el archivo .env sin importar desde dónde se ejecute el comando en la terminal
ruta_env = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(ruta_env):
    load_dotenv(ruta_env)
else:
    load_dotenv() # Intento por defecto si ya está en la raíz

TOKEN = os.getenv('DISCORD_TOKEN')

# Configuración del archivo
NOMBRE_ARCHIVO = 'dataset_entrenamiento.json'
DIRECTOR_ID = 733720277304737823 # ID del director de carrera (cambiarlo)
NOMBRE_DIRECTOR = "Luis Flores"
BOT_ID = 1516215867090931753 # ID del bot
CANALES_OBJETIVO = [873310192333377546, 1058544766142394379, 819657860627038228]

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def obtener_ciclo(fecha_utc):
    if fecha_utc is None:
        return "Desconocido"
        
    # Ajustamos la hora a Perú (UTC-5) para que los cambios de mes sean exactos
    fecha_local = fecha_utc - datetime.timedelta(hours=5)
    mes = fecha_local.month
    anio = fecha_local.year
    
    # De noviembre hasta enero: año-0 o (año+1)-0
    if mes in [11, 12]:
        return f"{anio + 1}-0"
    elif mes == 1:
        return f"{anio}-0"
    # De febrero a abril (incluyendo mayo como cobertura): año-1
    elif mes in [2, 3, 4, 5]:
        return f"{anio}-1"
    # De junio, julio y agosto (incluyendo sep y oct como cobertura): año-2
    elif mes in [6, 7, 8, 9, 10]:
        return f"{anio}-2"
        
    return f"{anio}-X"

# Función auxiliar para limpiar los tags del texto
def limpiar_texto(texto):
    if texto is None:
        return None
    # Reemplaza el tag estándar
    texto = texto.replace(f'<@{DIRECTOR_ID}>', NOMBRE_DIRECTOR)
    # Reemplaza el tag de aplicación móvil (por si acaso)
    texto = texto.replace(f'<@!{DIRECTOR_ID}>', NOMBRE_DIRECTOR)
    # Eliminamos el tag del bot (lo reemplazamos por nada)
    texto = texto.replace(f'<@{BOT_ID}>', '')
    texto = texto.replace(f'<@!{BOT_ID}>', '')
    # Devolvemos eliminando saltos de línea invisibles al inico o al final
    return texto.strip()

class VistaTicket(discord.ui.View):
    def __init__(self, mensaje_original, pregunta_alumno):
        # timeout=None asegura que el botón no deje de funcionar después de unos minutos
        super().__init__(timeout=None) 
        self.mensaje_original = mensaje_original
        self.pregunta_alumno = pregunta_alumno

    # Definimos el aspecto del botón (color primario/azul y un emoji)
    @discord.ui.button(label="Contactar a Luis Flores", style=discord.ButtonStyle.primary, emoji="📩")
    async def boton_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        # Verificamos si la persona que hizo clic es la misma que llamó al bot
        if interaction.user.id != self.mensaje_original.author.id:
            # Si es un intruso, le enviamos un mensaje de error que solo él puede ver
            await interaction.response.send_message("❌ Este botón es personal y solo puede usarlo quien hizo la consulta.", ephemeral=True)
            return
        # 1. Discord exige que respondamos al clic en menos de 3 segundos
        # Usamos ephemeral=True para que solo el alumno que hizo clic vea este aviso temporal
        await interaction.response.send_message("⏳ Procesando el envío del ticket...", ephemeral=True)
        # 2. Deshabilitamos el botón inmediatamente para que no le hagan spam de clics
        button.disabled = True
        await interaction.message.edit(view=self)
        # 3. Lógica de envío al Director
        try:
            director = await interaction.client.fetch_user(DIRECTOR_ID)
            reporte = (
                f"🚨 **NUEVA CONSULTA ESCALADA** 🚨\n"
                f"**Alumno:** {self.mensaje_original.author.mention}\n"
                f"**Canal:** {self.mensaje_original.channel.mention}\n"
                f"**Pregunta Original:** {self.pregunta_alumno}\n"
            )
            await director.send(reporte) 
            # 4. Actualizamos el mensaje efímero con el éxito
            await interaction.edit_original_response(content="✅ ¡Ticket enviado! Luis Flores te responderá a la brevedad.")
        except discord.Forbidden:
            await interaction.edit_original_response(content="❌ No pude enviar el ticket porque el director tiene los mensajes privados bloqueados.")
        except Exception as e:
            await interaction.edit_original_response(content=f"❌ Ocurrió un error inesperado: {e}")

@client.event
async def on_ready():
    print(f'✅ Bot {client.user} conectado y escuchando el chat...')

@client.event
async def on_message(message):
    # Evitar que el bot se responda a sí mismo
    if message.author == client.user:
        return

    # Si alguien escribe "hola", se activa el proceso
    if client.user in message.mentions:
        
        # 1. Avisar y guardar el mensaje de que el proceso comenzó
        mensaje_espera = await message.channel.send("⏳ Hola, papu. Agrupando preguntas con respuestas, dame un momento...")
        
        datos_qa = []
        
        for canal_id in CANALES_OBJETIVO:
            canal = client.get_channel(canal_id)

            if canal is None:
                try:
                    canal = await client.fetch_channel(canal_id)
                except Exception as e:
                    print(f"⚠️ No pude acceder al canal con ID {canal_id}: {e}")
                    continue # Saltamos al siguiente canal si hay error
            try:
                # Extraemos el historial de este canal en específico
                mensajes = [msg async for msg in canal.history(limit=None, oldest_first=True)]
            except discord.Forbidden:
                print(f"⚠️ No tengo permisos para leer el historial del canal {canal_id}")
                continue
        
            current_question = None
            current_answer = []
            current_cycle = None

            for i, msg in enumerate(mensajes):
                # Analizamos si el mensaje es del Director
                if msg.author.id == DIRECTOR_ID:
                    
                    # CASO 1: Responde directamente mediante Reply
                    if msg.reference is not None and msg.reference.resolved is not None:
                        # Si ya teníamos un par QA guardándose, lo cerramos y añadimos a la lista
                        if current_question and current_answer:
                            datos_qa.append({
                                "pregunta": current_question,
                                "respuesta": "\n".join(current_answer),
                                "ciclo": current_cycle
                            })
                        
                        # Iniciamos la captura de un nuevo par QA
                        msg_referenciado = msg.reference.resolved
                        if isinstance(msg_referenciado, discord.Message):
                            # Limpiamos la pregunta entrante
                            current_question = limpiar_texto(msg_referenciado.content)
                            # Limpiamos la respuesta por si el director se etiqueta a sí mismo o copia un texto
                            current_answer = [limpiar_texto(msg.content)]
                            current_cycle = obtener_ciclo(msg.created_at) # Capturamos el ciclo de la respuesta
                            
                    # CASO 2: No es reply, es un mensaje suelto
                    else:
                        # Subcaso 2.1: El mensaje justo anterior no es de él y tiene un "?"
                        if i > 0 and mensajes[i-1].author.id != DIRECTOR_ID and '?' in mensajes[i-1].content:
                            # Cerramos QAs previos
                            if current_question and current_answer:
                                datos_qa.append({
                                    "pregunta": current_question,
                                    "respuesta": "\n".join(current_answer),
                                    "ciclo": current_cycle
                                })
                            
                            current_question = limpiar_texto(mensajes[i-1].content)
                            current_answer = [limpiar_texto(msg.content)]
                            current_cycle = obtener_ciclo(msg.created_at)
                            
                        # Subcaso 2.2: Son mensajes seguidos del director (continuación de su respuesta)
                        elif i > 0 and mensajes[i-1].author.id == DIRECTOR_ID:
                            if current_question is not None: # Si pertenece a una pregunta activa
                                current_answer.append(limpiar_texto(msg.content))
                                
                # Si el mensaje NO es del Director
                else:
                    # Si había una respuesta del director capturándose, se cierra el bloque porque ya habló otra persona
                    if current_question and current_answer:
                        datos_qa.append({
                            "pregunta": current_question,
                            "respuesta": "\n".join(current_answer),
                            "ciclo": current_cycle
                        })
                        current_question = None
                        current_answer = []
                        current_cycle = None

            # 3. Al terminar todo el bucle, si quedó un último par abierto, lo guardamos
            if current_question and current_answer:
                datos_qa.append({
                    "pregunta": current_question,
                    "respuesta": "\n".join(current_answer),
                    "ciclo": current_cycle
                })

        # 4. Guardar en formato JSON explícitamente
        with open(NOMBRE_ARCHIVO, 'w', encoding='utf-8') as f:
            # ensure_ascii=False para que los acentos y ñ se vean bien (no en unicode \u00f1)
            json.dump(datos_qa, f, indent=4, ensure_ascii=False)
        
        # 5. Enviar el archivo de vuelta al chat
        if os.path.exists(NOMBRE_ARCHIVO):
            archivo_discord = discord.File(NOMBRE_ARCHIVO)
            usuario_etiquetado = message.author.mention

            # --- NUEVA LÓGICA DEL BOTÓN ---
            # Instanciamos la vista pasándole el mensaje actual
            # vista_contacto = VistaTicket(
            #     mensaje_original=message, 
            #     pregunta_alumno=message.content # Usamos el mensaje actual como "pregunta" de prueba
            # )
            
            await message.channel.send(
                file=archivo_discord,
                content=f"📋 ¡Hola, {usuario_etiquetado}! Aquí tienes tu `.json` listo con **{len(datos_qa)} pares** de Q&A estructurados."
            )
            await mensaje_espera.delete()
        else:
            await message.channel.send("❌ Hubo un error al generar el archivo JSON.")
            await mensaje_espera.delete()

# Iniciar el bot
client.run(TOKEN)