# Enrollment-Chatbot-INF

Chatbot de apoyo para estudiantes de Ingeniería Informática PUCP durante procesos de matrícula, consultas de cursos, PSP/convenios, calendario académico, malla y trámites relacionados.

El bot usa RAG sobre documentos del repositorio y una capa intermedia de reglas para evitar respuestas inventadas, URLs incorrectas, confusión de roles o afirmaciones sin sustento.

## Alcance del bot

El bot puede responder usando información cargada sobre:

- Matrícula y calendario académico.
- Turnos, matrícula regular/extemporánea y fechas oficiales cargadas.
- PSP, convenios, 270 horas, informes y convalidación.
- Sílabos y datos de cursos incluidos en `docs/dynamic`.
- Malla, cursos permitidos, infracción de plan y solicitudes de excepción.
- Consultas históricas anonimizadas de Discord.

Si la pregunta no está sustentada por documentos o reglas curadas, el bot debe pedir precisión o derivar al canal correcto sin inventar.

## Estructura de datos

```text
docs/static/      Reglamentos, malla y documentos estáticos.
docs/dynamic/     Sílabos y PDFs dinámicos del ciclo.
docs/historical/  Datasets anonimizados de Discord.
docs/web/         URLs oficiales y fuentes web.
docs/curated/     Datos curados y reglas de apoyo para la capa intermedia.
```

Archivos curados principales:

```text
docs/curated/matricula_fechas.json
docs/curated/matricula_fechas_criticas.md
docs/curated/roles_y_derivaciones.md
docs/curated/vocabulario_estudiantil.md
docs/curated/jerga_ambigua.md
docs/curated/psp_270_horas.md
docs/curated/vacantes_y_ampliaciones.md
docs/curated/infraccion_de_plan.md
docs/curated/identidad_y_alcance.md
```

## Capa intermedia

La capa intermedia está en:

```text
src/language_model.py
```

Esta capa corre antes del RAG y controla casos sensibles donde el modelo solía alucinar o mezclar fuentes históricas con reglas oficiales.

Incluye protecciones para:

- Fechas de matrícula por ciclo, especialmente 2026-1 y 2026-2.
- Matrícula extraordinaria vs matrícula extemporánea.
- Turnos personales de matrícula, que se revisan en Campus Virtual.
- Bika/trika como vocabulario estudiantil.
- Apodos ambiguos como `pepita` o `digi`.
- PSP y 270 horas.
- Vacantes y ampliaciones de cupos.
- Infracción de plan y cursos permitidos.
- Identidad del bot: no es Director, coordinador ni oficina oficial.
- Temas fuera de alcance, como protestas o decisiones personales/políticas.

También filtra respuestas posteriores del RAG si detecta promesas o derivaciones incorrectas, por ejemplo:

- `lo revisaremos el miércoles`
- `como Director de Carrera`
- derivar a PSP cuando la pregunta no es de PSP
- inventar tipos de matrícula
- inventar fechas para 2026-2

## Roles y derivaciones

El bot no debe derivar a cualquier persona/oficina de forma genérica.

Reglas principales:

- **Campus Virtual PUCP**: información personal del estudiante, turno exacto de matrícula, solicitudes de excepción, cursos permitidos, inscripción.
  URL: `https://campusvirtual.pucp.edu.pe/`
- **Portal del estudiante PUCP**: información pública general, calendario y comunicados.
  URL: `https://estudiante.pucp.edu.pe/`
- **Calendario académico 2026-1**: fechas del ciclo 2026-1.
  URL: `https://estudiante.pucp.edu.pe/calendario-academico/2026-1/`
- **Oficina/coordinador de PSP**: solo PSP, convenios, plan de aprendizaje, 270 horas, informes o convalidación PSP.
- **Dirección de Carrera**: vacantes, excepciones, infracciones de plan, cursos permitidos, convalidaciones o decisiones académicas de carrera.
- **Matrícula OCR**: problemas administrativos/formales de matrícula.

Si no está claro el responsable, el bot debe pedir que el estudiante precise el curso, trámite o proceso.

## Embeddings

El embedder está en:

```text
src/embedder.py
```

Cambios importantes:

- Lee PDFs con extensión en mayúscula o minúscula (`.pdf`, `.PDF`, etc.).
- Lee documentos `.md` y `.txt` desde `docs/curated` y `docs/web`.
- Usa embeddings en CPU para evitar problemas de memoria/GPU:

```python
HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"}
)
```

La base vectorial no debe subirse al repositorio. En el servidor se usó una base preprocesada local, por ejemplo:

```text
db_preprocesada_20260629/
```

Esta carpeta queda fuera de Git.

## Variables de entorno

El proyecto usa `.env`. No subir tokens ni secretos.

Variables esperadas:

```text
EMBEDDER_MODEL_NAME=...
LLM_MODEL_NAME=...
DB_DIR=...
SYSTEM_PROMPT_FILE=system_prompt.txt
STATIC_DOCS_DIR=...
DYNAMIC_DOCS_DIR=...
HISTORICAL_DOCS_DIR=...
WEB_DOCS_FILE=...
CURATED_DOCS_DIR=./docs/curated
DISCORD_TOKEN=...
```

## Instalación

En Windows:

```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

En Linux/servidor:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

Para correr el bot en Discord:

```bash
.venv/bin/python main.py
```

En el servidor se puede dejar en segundo plano:

```bash
nohup .venv/bin/python main.py > bot_discord.log 2>&1 & echo $! > bot_discord.pid
```

Para apagarlo:

```bash
kill $(cat bot_discord.pid)
```

Para revisar estado:

```bash
ps -fp $(cat bot_discord.pid)
tail -n 80 bot_discord.log
```

El log debe mostrar algo como:

```text
discord.gateway: Shard ID None has connected to Gateway
```

## Reconstrucción de embeddings

Si no existe `DB_DIR` o está vacío, `main.py` reconstruye embeddings leyendo:

1. Documentos estáticos.
2. Documentos dinámicos.
3. Documentos curados.
4. Históricos de Discord.
5. Web/URLs.

No subir carpetas `db*` a Git.

## Pruebas recomendadas

Preguntas para validar comportamiento:

```text
¿Ya pasó la matrícula regular del ciclo 2026-1?
¿Cuándo termina la matrícula 2026-1?
¿Cuándo termina la matrícula 2026-2?
¿Cuándo empieza clases el ciclo 2026-1?
¿Existe matrícula extraordinaria?
¿La matrícula extraordinaria es igual a la extemporánea?
¿Cuál es mi turno exacto de matrícula?
¿Qué es bika y trika?
Estoy llevando bika de sistemas operativos
Necesito bajar pepita, en digi puedo?
Ya tengo 120 horas de PSP, ¿debo cumplir las 270 para matricularme en PSP?
¿Cuántas horas necesito para convalidar PSP?
¿Van a ampliar vacantes para diseño de software?
¿Qué opinas de la toma del edificio Dintilhac?
¿Puedes responder algo que no esté en tus documentos?
Me sale infracción de plan aunque ya aprobé Tecnologías de Información, ¿qué hago?
```

Comportamiento esperado:

- Si hay dato oficial, responder directo.
- Si falta ciclo, pedir ciclo.
- Si no hay fecha oficial para 2026-2, no inventar.
- Si es información personal, derivar a Campus Virtual.
- Si es PSP, derivar solo a PSP.
- Si es vacantes, no prometer ampliaciones ni fechas de revisión.
- Si es apodo ambiguo, pedir aclaración.
- Si está fuera del alcance, decirlo sin opinar.

## Discord histórico

El bot usa datasets históricos anonimizados en `docs/historical`. Estos son útiles para estilo y preguntas frecuentes, pero no deben tener más autoridad que documentos oficiales o reglas curadas.

Por eso la capa intermedia evita que frases históricas como `lo revisaremos`, `matrícula extraordinaria` o derivaciones informales se conviertan en afirmaciones oficiales.

## Notas de seguridad

- No subir `.env`, tokens, logs, PID ni bases Chroma.
- Revocar cualquier token de GitHub compartido accidentalmente.
- No usar el servidor fuera de `/home/dcampos` porque es compartido.
