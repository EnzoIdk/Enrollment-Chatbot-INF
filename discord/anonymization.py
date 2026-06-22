import json
import re
import spacy
import os

# Cargar el modelo de NLP en español
try:
    nlp = spacy.load("es_core_news_md")
except OSError:
    print("Error: Falta instalar el modelo. Ejecuta: python -m spacy download es_core_news_md")

def limpiar_texto(texto):
    if not isinstance(texto, str):
        return texto
        
    texto = re.sub(r'[\r\n]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

def anonimizar_texto(texto, diccionario_memoria):
    if not isinstance(texto, str):
        return texto

    # Capa 1: Limpieza básica
    texto = re.sub(r'<@!?\d+>', '[USUARIO_DISCORD]', texto)
    texto = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[CORREO]', texto)
    # Ocultamos el código PUCP (8 dígitos exactos)
    texto = re.sub(r'\b\d{8}\b', '********', texto)

    # Capa 2: Regla explícita para el director de carrera
    patron_director = r'(?i)\b(?:profesor\s+|prof\.\s+)?(?:luis\s+)?flores\b'
    texto = re.sub(patron_director, '[DIRECTOR]', texto)

    # Capa 3: NER interactivo
    doc = nlp(texto)
    texto_anonimizado = texto
    
    # Iteramos en reversa para no romper los índices al modificar el string
    for ent in reversed(doc.ents):
        if ent.label_ == "PER":
            nombre_detectado = ent.text.strip()
            
            # Evitar falsos positivos con etiquetas previas
            if "[DIRECTOR]" in nombre_detectado.upper() or "[USUARIO_DISCORD]" in nombre_detectado:
                continue

            # Si es la primera vez que vemos esta palabra, preguntamos al usuario
            if nombre_detectado not in diccionario_memoria:
                # Extraemos contexto (40 caracteres antes y después)
                inicio_ctx = max(0, ent.start_char - 40)
                fin_ctx = min(len(texto), ent.end_char + 40)
                contexto = texto[inicio_ctx:fin_ctx]
                
                # Resaltamos la palabra en el contexto para que sea fácil de ver
                contexto_visual = contexto.replace(nombre_detectado, f"\033[93m>> {nombre_detectado} <<\033[0m")
                
                print(f"\n{'-'*60}")
                print(f"Entidad detectada: \033[96m{nombre_detectado}\033[0m")
                print(f"Contexto: ...{contexto_visual}...")
                
                # Bucle hasta que el usuario ingrese una opción válida
                while True:
                    opcion = input("Clasificar como -> [P]rofesor / [A]lumno / [D]escartar: ").strip().upper()
                    if opcion == 'P':
                        diccionario_memoria[nombre_detectado] = "[PROFESOR]"
                        break
                    elif opcion == 'A':
                        diccionario_memoria[nombre_detectado] = "[ALUMNO]"
                        break
                    elif opcion == 'D':
                        diccionario_memoria[nombre_detectado] = "DESCARTAR"
                        break
                    else:
                        print("Opción inválida. Usa P, A o D.")

            # Aplicar la decisión guardada en memoria
            etiqueta_asignada = diccionario_memoria[nombre_detectado]
            if etiqueta_asignada != "DESCARTAR":
                texto_anonimizado = (texto_anonimizado[:ent.start_char] + etiqueta_asignada + texto_anonimizado[ent.end_char:])
    
    return texto_anonimizado

def procesar_dataset(input_files, output_suffix="_anonimizado.json", dict_file="diccionario_anonimizacion.json"):
    # Cargar diccionario previo si existe (útil si pausas el script y continúas luego)
    diccionario_memoria = {}
    if os.path.exists(dict_file):
        with open(dict_file, 'r', encoding='utf-8') as f:
            diccionario_memoria = json.load(f)
            print(f"Diccionario cargado con {len(diccionario_memoria)} palabras previamente clasificadas.")

    # Procesar cada archivo de la lista
    for input_file in input_files:
        print(f"\nProcesando archivo: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data_procesada = []
        for i, item in enumerate(data):
            nuevo_item = {}
            for clave, valor in item.items():
                if clave in ["pregunta", "respuesta"]:
                    texto_limpio = limpiar_texto(valor)
                    nuevo_item[clave] = anonimizar_texto(texto_limpio, diccionario_memoria)
                else:
                    nuevo_item[clave] = valor
            data_procesada.append(nuevo_item)

        nombre_base = os.path.basename(input_file)
        # Guardar dataset anonimizado
        output_file = nombre_base.replace('.json', output_suffix)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_procesada, f, ensure_ascii=False, indent=4)
        print(f"Archivo guardado: {output_file}")

    # Al finalizar todo, guardar el diccionario actualizado
    with open(dict_file, 'w', encoding='utf-8') as f:
        json.dump(diccionario_memoria, f, ensure_ascii=False, indent=4)
    print(f"\nDiccionario de anonimización actualizado y guardado en: {dict_file}")
    print("Recuerda añadir este archivo a tu .gitignore")

# Lista de archivos a procesar
archivos = ['datasetOriginal/dataset_entrenamiento.json', 'datasetOriginal/dataset_psp.json']
procesar_dataset(archivos)