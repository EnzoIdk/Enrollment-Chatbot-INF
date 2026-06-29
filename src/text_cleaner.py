import re
import unicodedata

from langchain_core.documents.base import Document


"""
Este módulo contiene la clase 'TextCleaner', la cual se encarga de limpiar
y preprocesar el texto extraído de documentos PDF antes de que sea dividido 
en chunks para la base de datos vectorial del RAG.

Pipeline de limpieza:
    1. Reparar encoding (mojibake latin1 → UTF-8)
    2. Filtrar caracteres inválidos (solo español y ASCII)
    3. Normalizar whitespace
    4. Eliminar headers/footers repetidos de páginas
    5. Reestructurar tablas según el tipo de documento
    6. Enriquecer metadata con tipo de documento
    7. Filtrar chunks de baja calidad
"""


class TextCleaner:

    # Mapa de reemplazos para mojibake comunes (latin1 mal interpretado como UTF-8)
    # Se usan secuencias unicode escapadas para evitar problemas de encoding en el source
    _MOJIBAKE_REPLACEMENTS = {
        # Vocales minúsculas acentuadas
        "\u00c3\u00a1": "\u00e1",  # Ã¡ → á
        "\u00c3\u00a9": "\u00e9",  # Ã© → é
        "\u00c3\u00ad": "\u00ed",  # Ã­ → í
        "\u00c3\u00b3": "\u00f3",  # Ã³ → ó
        "\u00c3\u00ba": "\u00fa",  # Ãº → ú
        "\u00c3\u00b1": "\u00f1",  # Ã± → ñ
        "\u00c3\u00bc": "\u00fc",  # Ã¼ → ü
        # Vocales mayúsculas acentuadas
        "\u00c3\u0081": "\u00c1",  # Ã + \x81 → Á
        "\u00c3\u0089": "\u00c9",  # Ã + \x89 → É
        "\u00c3\u008d": "\u00cd",  # Ã + \x8d → Í
        "\u00c3\u0093": "\u00d3",  # Ã + \x93 → Ó
        "\u00c3\u009a": "\u00da",  # Ã + \x9a → Ú
        "\u00c3\u0091": "\u00d1",  # Ã + \x91 → Ñ
        "\u00c3\u009c": "\u00dc",  # Ã + \x9c → Ü
        # Comillas tipográficas
        "\u00e2\u0080\u009c": '"',   # " → "
        "\u00e2\u0080\u009d": '"',   # " → "
        "\u00e2\u0080\u0098": "'",   # ' → '
        "\u00e2\u0080\u0099": "'",   # ' → '
        # Guiones y puntos suspensivos
        "\u00e2\u0080\u0094": "\u2014",  # — (em dash)
        "\u00e2\u0080\u0093": "\u2013",  # – (en dash)
        "\u00e2\u0080\u00a6": "\u2026",  # … (ellipsis)
        # Símbolos comunes
        "\u00c2\u00b0": "\u00b0",  # ° (grado)
        "\u00c2\u00bf": "\u00bf",  # ¿
        "\u00c2\u00a1": "\u00a1",  # ¡
        "\u00c2\u00ab": "\u00ab",  # «
        "\u00c2\u00bb": "\u00bb",  # »
    }

    # Patrones de headers/footers comunes en PDFs universitarios
    _HEADER_FOOTER_PATTERNS = [
        # Header con espacios amplios entre letras (como el de la malla)
        re.compile(r"P\s+L\s+A\s+N\s+D\s+E\s+E\s+S\s+T\s+U\s+D\s+I\s+O\s+S.*", re.IGNORECASE),
        # Encabezado de columnas de la malla
        re.compile(r"^CI\s+CLAVE\s+C\s+U\s+R\s+S\s+O\s+CT.*$", re.MULTILINE),
        # Headers de facultad repetidos (con cualquier variante de encoding)
        re.compile(
            r"^FACULTAD\s+DE\s*\n?CIENCIAS\s+E\s*\n?INGENIER.A\s*$",
            re.MULTILINE
        ),
        re.compile(
            r"^PONTIFICIA\s+UNIVERSIDAD\s+CAT.LICA\s+DEL\s+PER.\s+FACULTAD.*$",
            re.MULTILINE
        ),
        # Headers de sílabos repetidos
        re.compile(r"^FACULTAD\s+DE\s+CIENCIAS\s+E\s+INGENIER.A\s*$", re.MULTILINE),
        # Números de página sueltos
        re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE),
    ]

    # Patrón para detectar líneas de la malla curricular
    # Formato: [CICLO?] CÓDIGO NOMBRE_CURSO CLASES [PA] [PB] REQUISITOS CRÉDITOS
    # Ejemplo: "5 1GES92 Taller de Habilidades Interpersonales 1 70 créditos aprobados * 1.00"
    # Ejemplo: "1INF25 Programación 2 4 2 (4q) INF144, [1INF27] 5.00"
    _MALLA_LINE_PATTERN = re.compile(
        r"^"
        r"(?:(\d{1,2})\s+)?"                           # Ciclo (opcional)
        r"(\d?[A-Z]{1,3}\d{2,3})\s+"                    # Código del curso (ej: 1INF25, INF238)
        r"(.+?)\s+"                                     # Nombre del curso (non-greedy)
        r"(\d+(?:\.\d+)?)\s*$"                          # Créditos (al final de la línea)
    )

    # Patrón para detectar códigos de curso como prerequisitos
    _COURSE_CODE_PATTERN = re.compile(r"\d?[A-Z]{1,3}\d{2,3}")

    # Patrones para detectar el tipo de documento a partir del nombre del archivo
    _DOC_TYPE_PATTERNS = {
        "silabo": re.compile(r"SILABO", re.IGNORECASE),
        "malla": re.compile(r"MALLA", re.IGNORECASE),
        "reglamento": re.compile(r"REGLAMENTO", re.IGNORECASE),
        "resolucion": re.compile(r"RESOLUCI[OÓ]N", re.IGNORECASE),
        "calendario": re.compile(r"CALENDARIO", re.IGNORECASE),
        "aviso": re.compile(r"AVISO", re.IGNORECASE),
        "indicaciones": re.compile(r"INDICACI[OÓ]N|INDICACIONES", re.IGNORECASE),
        "bienvenida": re.compile(r"BIENVENIDA|MENSAJE", re.IGNORECASE),
    }


    def __init__(self, min_chunk_length: int = 30):
        """
        Inicializa el TextCleaner.

        Args:
            min_chunk_length: Umbral mínimo de caracteres para considerar un chunk válido.
                              Chunks con menos caracteres serán descartados.
        """
        assert min_chunk_length >= 0, "min_chunk_length debe ser >= 0"
        self.min_chunk_length = min_chunk_length


    def clean(self, documents: list[Document]) -> list[Document]:
        """
        Aplica todo el pipeline de limpieza a los documentos extraídos de PDFs.

        Pipeline:
            1. Fix encoding (mojibake)
            2. Filtrar caracteres no válidos
            3. Normalizar whitespace
            4. Eliminar headers/footers
            5. Reestructurar tablas
            6. Enriquecer metadata
            7. Filtrar chunks de baja calidad

        Args:
            documents: Lista de Documents obtenidos de PyPDFDirectoryLoader.

        Returns:
            Lista de Documents limpios y enriquecidos, listos para chunking.
        """
        assert documents is not None, "La lista de documentos no puede ser None"

        cleaned = []
        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata.copy()

            # 1. Reparar encoding
            text = self._fix_encoding(text)

            # 2. Filtrar caracteres no válidos (solo español y ASCII)
            text = self._remove_invalid_characters(text)

            # 3. Normalizar whitespace
            text = self._normalize_whitespace(text)

            # 3. Eliminar headers/footers
            text = self._remove_headers_footers(text)

            # 4. Reestructurar tablas según tipo de documento
            doc_type = self._detect_document_type(metadata)
            text = self._restructure_tables(text, doc_type)

            # 5. Enriquecer metadata
            metadata = self._enrich_metadata(metadata, doc_type)

            # 6. Normalizar whitespace final (post-reestructuración)
            text = self._normalize_whitespace(text)

            cleaned.append(Document(page_content=text, metadata=metadata))

        # Filtrar chunks de baja calidad
        cleaned = self._filter_low_quality(cleaned)

        return cleaned


    def _fix_encoding(self, text: str) -> str:
        """
        Repara caracteres mojibake producidos por la mala interpretación
        de encoding latin1/cp1252 como UTF-8 por parte de PyPDF.
        
        También reemplaza el carácter de reemplazo Unicode (�) con heurísticas
        basadas en el contexto.
        """
        # Primero intentar reparar mojibake conocidos
        for bad, good in self._MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, good)

        # Reemplazar el carácter de reemplazo Unicode (U+FFFD = �)
        # con vocales acentuadas probables según contexto en español
        text = self._fix_replacement_chars(text)

        # Normalizar a NFC (forma canónica compuesta)
        text = unicodedata.normalize("NFC", text)

        return text


    def _remove_invalid_characters(self, text: str) -> str:
        """
        Elimina cualquier carácter que no sea ASCII imprimible, 
        caracteres especiales permitidos en español, o saltos de línea.
        """
        # Expresión regular con los caracteres permitidos:
        # \x20-\x7E : Rango ASCII imprimible (espacios, números, letras inglesas, signos comunes de puntuación)
        # \n, \t    : Saltos de línea y tabulaciones
        # áéíóúÁÉÍÓÚñÑüÜ : Vocales acentuadas, eñes, diéresis
        # ¿¡        : Signos de apertura de interrogación y exclamación
        allowed_pattern = re.compile(r"[^\x20-\x7E\n\táéíóúÁÉÍÓÚñÑüÜ¿¡]+")
        
        # Eliminar cualquier cosa que no coincida con los caracteres permitidos
        return allowed_pattern.sub("", text)


    def _fix_replacement_chars(self, text: str) -> str:
        """
        Intenta reparar caracteres '�' (U+FFFD) basándose en el contexto
        de palabras en español.
        
        Usa patrones comunes como:
            - "ci�n" → "ción"
            - "m�s" → "más"  
            - "�tica" → "ética"
            - etc.
        """
        # Patrones contextuales para español (patrón → reemplazo)
        context_patterns = [
            # Terminaciones comunes
            (r"ci\ufffd(?=n\b)", "ció"),           # "ción"
            (r"si\ufffd(?=n\b)", "sió"),           # "sión"
            (r"(?<=\b)m\ufffd(?=s\b)", "má"),      # "más"
            (r"\ufffd(?=rea\b)", "á"),              # "área"
            (r"\ufffd(?=tico)", "é"),               # "ético", "ética"
            (r"\ufffd(?=xito)", "é"),               # "éxito"
            (r"a\ufffd(?=o\b)", "añ"),              # "año"
            (r"espa\ufffd(?=ol)", "españ"),         # "español"
            (r"dise\ufffd(?=o)", "diseñ"),          # "diseño"
            (r"ense\ufffd(?=anza)", "enseñ"),       # "enseñanza"
            (r"compa\ufffd", "compañ"),             # "compañía", "compañero"
            (r"se\ufffd(?=al)", "señ"),             # "señal"
            # Vocales acentuadas en contextos comunes
            (r"(?<=inform\ufffd)t", "át"),          # "informática"  
            (r"(?<=matem\ufffd)t", "át"),           # "matemática"
            (r"(?<=estad\ufffd)st", "íst"),         # "estadística"
            (r"(?<=cr\ufffd)d", "éd"),              # "crédito"
            (r"(?<=acad\ufffd)m", "ém"),            # "académico"
            (r"(?<=l\ufffd)nea", "ínea"),           # "línea"
            (r"(?<=p\ufffd)gina", "ágina"),         # "página"
            (r"(?<=n\ufffd)mero", "úmero"),         # "número"
            (r"(?<=per\ufffd)odo", "íodo"),         # "período"
            (r"(?<=pr\ufffd)ctica", "áctica"),      # "práctica"
            (r"(?<=t\ufffd)cnica", "écnica"),       # "técnica"
            (r"(?<=t\ufffd)cni", "écni"),           # "técnico"
            (r"(?<=m\ufffd)todo", "étodo"),         # "método"
            (r"(?<=met\ufffd)di", "ódi"),           # "metódica"
            (r"(?<=algorit\ufffd)i", "mi"),         # no aplica realmente
            (r"(?<=electr\ufffd)ni", "óni"),        # "electrónica"
            (r"(?<=aut\ufffd)nom", "ónom"),         # "autónomo"
            (r"(?<=programaci\ufffd)n", "ón"),      # ya cubierto por "ción"
            (r"(?<=ingenier\ufffd)a", "ía"),        # "ingeniería"
            (r"(?<=econom\ufffd)a", "ía"),          # "economía"
            (r"(?<=tecnolog\ufffd)a", "ía"),        # "tecnología"
            (r"(?<=bibliograf\ufffd)a", "ía"),      # "bibliografía"
        ]

        for pattern, replacement in context_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Para los '�' restantes, reemplazar con cadena vacía para no
        # introducir ruido en el retriever
        text = text.replace("\ufffd", "")

        return text


    def _normalize_whitespace(self, text: str) -> str:
        """
        Normaliza espacios en blanco:
            - Colapsa múltiples saltos de línea en máximo 2 (un párrafo de separación)
            - Elimina espacios al final de cada línea (trailing whitespace)
            - Reemplaza tabulaciones por un espacio
            - Colapsa múltiples espacios horizontales en uno solo
        """
        # Reemplazar tabs por espacio
        text = text.replace("\t", " ")

        # Colapsar múltiples espacios horizontales en uno
        text = re.sub(r"[^\S\n]+", " ", text)

        # Eliminar espacios al inicio y final de cada línea
        text = re.sub(r"^ +| +$", "", text, flags=re.MULTILINE)

        # Colapsar 3+ saltos de línea en 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


    def _remove_headers_footers(self, text: str) -> str:
        """
        Elimina headers y footers repetidos que aparecen en cada página
        del PDF y no aportan información semántica útil al retriever.
        """
        for pattern in self._HEADER_FOOTER_PATTERNS:
            text = pattern.sub("", text)

        return text


    def _detect_document_type(self, metadata: dict) -> str:
        """
        Detecta el tipo de documento basándose en el nombre del archivo
        presente en la metadata ('source').

        Returns:
            Tipo de documento: 'silabo', 'malla', 'reglamento', 'resolucion',
            'calendario', 'aviso', 'indicaciones', 'bienvenida', o 'general'.
        """
        source = metadata.get("source", "")

        for doc_type, pattern in self._DOC_TYPE_PATTERNS.items():
            if pattern.search(source):
                return doc_type

        return "general"


    def _restructure_tables(self, text: str, doc_type: str) -> str:
        """
        Reestructura contenido tabular en texto semántico legible,
        aplicando un handler específico según el tipo de documento.
        """
        if doc_type == "malla":
            return self._restructure_malla(text)
        elif doc_type == "silabo":
            return self._restructure_silabo(text)
        else:
            # Para otros tipos, limpieza genérica de líneas tabulares
            return self._restructure_generic(text)


    def _restructure_malla(self, text: str) -> str:
        """
        Reestructura las filas de la malla curricular.
        
        Formato de entrada (8 cols, algunas vacías):
            [CICLO] CÓDIGO NOMBRE CT [Pa] [Pb] REQUISITOS CRÉDITOS
        
        Formato de salida (solo lo relevante):
            Curso CÓDIGO - NOMBRE. Créditos: X. Requisitos: Y.
        """
        lines = text.split("\n")
        result = []
        current_ciclo = ""
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                result.append("")
                i += 1
                continue

            # Detectar secciones especiales (ej: "ELECTIVOS DE LA ESPECIALIDAD")
            if line.isupper() and not self._COURSE_CODE_PATTERN.match(line.split()[0] if line.split() else ""):
                result.append(line)
                i += 1
                continue

            # Intentar parsear como línea de curso de la malla
            parsed = self._parse_malla_line(line, lines, i)

            if parsed:
                ciclo, codigo, nombre, requisitos, creditos, lines_consumed = parsed

                if ciclo:
                    current_ciclo = ciclo

                # Construir texto semántico
                parts = [f"Curso {codigo} - {nombre}"]
                if creditos:
                    parts.append(f"Créditos: {creditos}")
                if requisitos and requisitos != "Ninguno":
                    parts.append(f"Requisitos: {requisitos}")
                if current_ciclo:
                    parts.append(f"Ciclo: {current_ciclo}")

                result.append(". ".join(parts) + ".")
                i += lines_consumed
            else:
                result.append(line)
                i += 1

        return "\n".join(result)


    def _parse_malla_line(self, line: str, lines: list[str], current_index: int) -> tuple | None:
        """
        Intenta parsear una línea como entrada de la malla curricular.
        
        Maneja el caso donde el nombre del curso se divide en 2 líneas:
            "1INF55 Rediseño de interfaces gráficas en Sistemas de"
            "información empresarial 3 (2q) 1INF40 3.50"
        
        Returns:
            Tupla (ciclo, código, nombre, requisitos, créditos, líneas_consumidas)
            o None si no se pudo parsear.
        """
        tokens = line.split()
        if not tokens:
            return None

        idx = 0
        ciclo = ""

        # Verificar si empieza con un número de ciclo (1 o 2 dígitos solos)
        if tokens[0].isdigit() and len(tokens[0]) <= 2 and int(tokens[0]) <= 12:
            ciclo = tokens[0]
            idx = 1

        if idx >= len(tokens):
            return None

        # Verificar si el siguiente token es un código de curso
        if not self._COURSE_CODE_PATTERN.fullmatch(tokens[idx]):
            return None

        codigo = tokens[idx]
        idx += 1

        # Extraer el crédito (último número de la línea, formato X.XX)
        creditos = ""
        requisitos = "Ninguno"

        # Buscar el último token que parece crédito (ej: "3.50", "2.00", "1.00")
        credit_match = re.search(r"(\d+\.\d{2})\s*$", line)

        if credit_match:
            creditos = credit_match.group(1)
            # Todo entre el código y los créditos es nombre + datos intermedios
            remaining = line[:credit_match.start()].strip()
        else:
            # Podría ser un nombre que continúa en la siguiente línea
            if current_index + 1 < len(lines):
                next_line = lines[current_index + 1].strip()
                combined = line + " " + next_line
                credit_match = re.search(r"(\d+\.\d{2})\s*$", combined)
                if credit_match:
                    creditos = credit_match.group(1)
                    remaining = combined[:credit_match.start()].strip()
                    # Re-parsear el remaining para extraer después del código
                    tokens_combined = remaining.split()
                    # Saltar ciclo y código
                    skip = (1 if ciclo else 0) + 1  # ciclo + código
                    remaining_after_code = " ".join(tokens_combined[skip:])
                    nombre, requisitos = self._extract_nombre_y_requisitos(remaining_after_code)
                    return (ciclo, codigo, nombre, requisitos, creditos, 2)
                else:
                    return None
            else:
                return None

        # Extraer nombre y requisitos del remaining
        remaining_tokens = remaining.split()
        # Saltar ciclo y código
        skip = (1 if ciclo else 0) + 1
        remaining_after_code = " ".join(remaining_tokens[skip:])

        nombre, requisitos = self._extract_nombre_y_requisitos(remaining_after_code)

        return (ciclo, codigo, nombre, requisitos, creditos, 1)


    def _extract_nombre_y_requisitos(self, text: str) -> tuple[str, str]:
        """
        Dado el texto entre el código del curso y los créditos,
        separa el nombre del curso de los datos intermedios (CT, Pa, Pb)
        y los requisitos.
        
        El nombre es texto alfabético, los datos intermedios son números
        y paréntesis, y los requisitos incluyen códigos de cursos o 
        frases como "70 créditos aprobados".
        """
        if not text:
            return ("", "Ninguno")

        # Buscar la primera secuencia de: número solo, o "(Xq)"
        # que indica el inicio de las columnas numéricas (CT, Pa, Pb)
        break_match = re.search(r"\s(\d+(?:\.\d+)?)\s", text)

        if break_match:
            nombre = text[:break_match.start()].strip()
            rest = text[break_match.start():].strip()
        else:
            # Todo es nombre
            return (text.strip(), "Ninguno")

        # Extraer requisitos: buscar códigos de curso o frases de créditos
        requisitos_parts = []

        # Buscar códigos de curso (formato: letras + dígitos, con posibles [], (), {})
        codigos = re.findall(r"[\[\({]?([A-Z]{1,3}\d{2,3})[\]\)}]?", rest)
        requisitos_parts.extend(codigos)

        # Buscar requisitos de créditos aprobados
        creditos_req = re.search(r"(\d+\s*cr[eé]ditos?\s*aprobados?\s*\*?)", rest, re.IGNORECASE)
        if creditos_req:
            requisitos_parts.append(creditos_req.group(1).strip())

        if requisitos_parts:
            return (nombre, ", ".join(requisitos_parts))
        else:
            return (nombre, "Ninguno")


    def _restructure_silabo(self, text: str) -> str:
        """
        Reestructura tablas comunes en sílabos:
            - Tabla de información general (CURSO, CLAVE, CRÉDITOS, etc.)
            - Tabla de planes curriculares (ESPECIALIDAD, ETAPA, etc.)
            - Tabla de evaluación (Código, Tipo, Pesos, etc.)
            - Tabla de competencias
        """
        # Reestructurar tabla de evaluación
        text = self._restructure_evaluation_table(text)

        # Reestructurar tabla de info general del curso
        text = self._restructure_course_info(text)

        return text


    def _restructure_evaluation_table(self, text: str) -> str:
        """
        Detecta y reestructura la tabla de evaluación de sílabos.
        
        Entrada típica:
            N° Codigo Tipo de Evaluación Cant. Eval. ... Pesos ...
            1 Pb Práctica tipo B 7 Por Promedio Pb=2 0
            2 Ta Tarea académica 1 Por Promedio Ta=3 0
            3 Ex Examen 2 Por Evaluación Ex1=2 Ex2=3
        
        Salida:
            Sistema de evaluación: Práctica tipo B (Pb), peso=2. 
            Tarea académica (Ta), peso=3. Examen (Ex), peso Ex1=2, Ex2=3.
        """
        # Buscar la sección de evaluación
        eval_header = re.search(
            r"(?:Sistema\s+de\s+evaluaci[oó]n|N[°º]\s+Codigo\s+Tipo)",
            text, re.IGNORECASE
        )
        if not eval_header:
            return text

        # Buscar la fórmula de evaluación (viene después de la tabla)
        formula_match = re.search(
            r"[Ff][oó]rmula\s+para\s+el\s+c[aá]lculo\s+de\s+la\s+nota\s+final\s*\n\s*(.+)",
            text
        )

        if formula_match:
            formula = formula_match.group(1).strip()
            # Limpiar la fórmula
            formula = re.sub(r"\s+", " ", formula)
            # Insertar texto descriptivo de la fórmula
            text = text[:formula_match.start()] + \
                   f"\nFórmula de nota final: {formula}\n" + \
                   text[formula_match.end():]

        return text


    def _restructure_course_info(self, text: str) -> str:
        """
        Reestructura campos clave-valor que aparecen en la info general
        del sílabo, limpiando espacios excesivos entre campos.
        
        Ejemplo:
            "CURSO PROCESAMIENTO DE LENGUAJE NATURAL"
            "CLAVE 1INF59"
            "CRÉDITOS 3.5"
        """
        # Limpiar campos clave-valor con espacios excesivos
        kv_patterns = [
            (r"CURSO\s+", "Curso: "),
            (r"CLAVE\s+", "Clave: "),
            (r"CR[ÉE�]DITOS\s+", "Créditos: "),
            (r"HORAS\s+DE\s+DICTADO\s+", "Horas de dictado: "),
            (r"PROFESORES?\s+", "Profesores: "),
        ]

        for pattern, replacement in kv_patterns:
            text = re.sub(pattern, replacement, text)

        return text


    def _restructure_generic(self, text: str) -> str:
        """
        Limpieza genérica para documentos sin handler específico.
        Detecta y limpia patrones tabulares obvios.
        """
        # Unir líneas que son continuación de la anterior
        # (línea que no empieza con mayúscula, número, ni bullet y la anterior no termina en punto)
        lines = text.split("\n")
        result = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                result.append("")
                continue

            # Si la línea anterior no terminó en un delimitador y esta
            # empieza en minúscula, unirlas
            if (result and result[-1] and
                not result[-1].endswith((".", ":", ";", ",")) and
                stripped[0].islower()):
                result[-1] = result[-1] + " " + stripped
            else:
                result.append(stripped)

        return "\n".join(result)


    def _filter_low_quality(self, documents: list[Document]) -> list[Document]:
        """
        Filtra documentos de baja calidad:
            - Muy cortos (< min_chunk_length caracteres)
            - Mayormente numéricos/puntuación (> 95%)
        """
        filtered = []

        for doc in documents:
            text = doc.page_content.strip()

            # Descartar si es muy corto
            if len(text) < self.min_chunk_length:
                continue

            # Descartar si es > 95% numérico/puntuación/whitespace
            if text:
                alpha_count = sum(1 for c in text if c.isalpha())
                alpha_ratio = alpha_count / len(text)
                if alpha_ratio < 0.05:
                    continue

            filtered.append(doc)

        return filtered


    def _enrich_metadata(self, metadata: dict, doc_type: str) -> dict:
        """
        Agrega campos adicionales al metadata del Document:
            - document_type: tipo de documento detectado
            - cleaned: flag que indica que pasó por el pipeline de limpieza
        """
        metadata["document_type"] = doc_type
        metadata["cleaned"] = True
        return metadata
