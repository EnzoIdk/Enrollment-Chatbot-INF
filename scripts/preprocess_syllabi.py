import json
import re
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader

CURRICULUM_PATH = Path("docs/static/processed/malla_informatica.json")
DYNAMIC_DOCS_DIR = Path("docs/dynamic")
OUTPUT_JSON = Path("docs/static/processed/silabos_resumen.json")
OUTPUT_MD = Path("docs/static/processed/silabos_resumen.md")

COURSE_NAME_FALLBACKS = {
    "1INF01": "Fundamentos de Programación",
    "INF134": "Estructuras Discretas",
    "INF144": "Técnicas de Programación",
}


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


ROMAN_SECTION_NAMES = {
    "I": "informacion_general",
    "II": "fundamentacion",
    "III": "sumilla",
    "IV": "sumilla",
    "V": "objetivos",
    "VI": "programa_analitico",
    "VII": "metodologia",
    "VIII": "evaluacion",
    "IX": "bibliografia",
    "X": "politica_contra_plagio",
}


def clean_pdf_noise(text: str, course_code: str = "", course_name: str = "") -> str:
    text = text or ""
    text = text.replace("\u00ad", "")
    text = text.replace("¿", "'") if "A learner¿s" in text else text
    text = re.sub(r"FACULTAD DE CIENCIAS E INGENIER[ÍI]A", " ", text, flags=re.IGNORECASE)
    if course_code:
        text = re.sub(rf"\b{re.escape(course_code)}\s*-\s*[^\n]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    text = re.sub(r"\$00[0-9a-fA-F]+", " ", text)
    text = re.sub(r"\bSD_ILS\b[^\s]*", " ", text)
    text = re.sub(r"\b(?:one|ne)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_inline_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([.;:])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" .")


def split_numbered_sections(text: str, course_code: str = "", course_name: str = "") -> dict[str, str]:
    cleaned_text = clean_pdf_noise(text, course_code, course_name)
    heading_pattern = re.compile(
        r"(?m)^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9 ,;:()\-/]+)\s*$"
    )
    matches = list(heading_pattern.finditer(cleaned_text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        roman = match.group(1).upper()
        title = clean_inline_text(match.group(2)).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned_text)
        raw_body = cleaned_text[start:end]
        key = ROMAN_SECTION_NAMES.get(roman, f"seccion_{roman.lower()}")
        if "bibliograf" in title:
            key = "bibliografia"
        elif "evaluaci" in title:
            key = "evaluacion"
        elif "metodolog" in title:
            key = "metodologia"
        elif "objetivo" in title:
            key = "objetivos"
        elif "programa" in title or "contenido" in title:
            key = "programa_analitico"
        elif "sumilla" in title:
            key = "sumilla"
        body = clean_section_body(raw_body)
        if body:
            sections[key] = body
    return sections


def clean_section_body(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        line = line.strip(" -\t")
        if not line:
            continue
        if re.match(r"^P[áa]gina\s+\d+", line, flags=re.IGNORECASE):
            continue
        if re.match(r"^\d[A-Z]{3}\d{2}\s*-", line):
            continue
        if line.upper() == "FACULTAD DE CIENCIAS E INGENIERÍA":
            continue
        lines.append(line)
    joined = "\n".join(lines)
    joined = re.sub(r"([a-záéíóúñ])\n([a-záéíóúñ])", r"\1 \2", joined)
    joined = re.sub(r"\n(?=[,.;:])", "", joined)
    joined = re.sub(r"\n{2,}", "\n", joined)
    joined = re.sub(r"[ \t]+", " ", joined)
    return joined.strip(" .\n")


def clean_section_for_embedding(text: str) -> str:
    text = clean_section_body(text)
    text = re.sub(r"\n", " ", text)
    return clean_inline_text(text)


def load_curriculum_courses() -> dict[str, dict[str, Any]]:
    if not CURRICULUM_PATH.exists():
        return {}
    data = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))
    return data.get("cursos", {}) if isinstance(data, dict) else {}


def extract_between(text: str, start_pattern: str, end_pattern: str) -> str:
    start = re.search(start_pattern, text, flags=re.IGNORECASE)
    if not start:
        return ""
    section = text[start.end():]
    end = re.search(end_pattern, section, flags=re.IGNORECASE)
    if end:
        section = section[:end.start()]
    return normalize_spaces(section)


def clean_sumilla(text: str) -> str:
    section = extract_between(text, r"IV\.\s*SUMILLA", r"\n\s*V\.")
    if not section:
        section = extract_between(text, r"\bSUMILLA\b", r"\n\s*(?:(?:III|IV|V)\.\s*)?(?:RESULTADOS\s+DE\s+APRENDIZAJE|CONTENIDO\s+TEM[ÁA]TICO|PROGRAMA\s+ANAL[ÍI]TICO|BIBLIOGRAF[ÍI]A)\b")
    section = re.sub(r"FACULTAD DE CIENCIAS E INGENIER[ÍI]A", " ", section, flags=re.IGNORECASE)
    section = re.sub(r"\s+", " ", section)
    return section.strip()



def clean_syllabus_section(text: str) -> str:
    return clean_section_for_embedding(clean_pdf_noise(text))


def extract_objectives(text: str, sections: dict[str, str] | None = None) -> list[str]:
    section = (sections or {}).get("objetivos", "")
    if not section:
        section = extract_between(text, r"V\.\s*OBJETIVOS", r"\n\s*VI\.")
    if not section:
        return []
    section = clean_syllabus_section(section)
    results = []
    for match in re.finditer(r"\b(RA\d+)\s*:\s*(.*?)(?=\s+RA\d+\s*:|$)", section, flags=re.IGNORECASE):
        item = f"{match.group(1).upper()}: {match.group(2).strip(' .')}"
        results.append(item)
    if results:
        return results
    return [section] if section else []


def extract_methodology(text: str, sections: dict[str, str] | None = None) -> str:
    section = (sections or {}).get("metodologia", "")
    if not section:
        section = extract_between(text, r"VII\.\s*METODOLOG[ÍI]A", r"\n\s*VIII\.")
    return clean_syllabus_section(section)


def extract_evaluation_section(text: str, sections: dict[str, str] | None = None) -> str:
    section = (sections or {}).get("evaluacion", "")
    if not section:
        section = extract_between(text, r"VIII\.\s*EVALUACI[ÓO]N", r"\n\s*IX\.")
    return clean_syllabus_section(section)


def extract_final_grade_formula(evaluation_section: str) -> str:
    match = re.search(
        r"F[óo]rmula\s+para\s+el\s+c[áa]lculo\s+de\s+la\s+nota\s+final\s*(.*?)(?:Aproximaci[óo]n|Consideraciones|$)",
        evaluation_section,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" .")


def extract_evaluation_components(evaluation_section: str) -> list[str]:
    compact = re.sub(r"\s+", " ", evaluation_section or " ").strip()
    components = []
    practice = re.search(r"\bPb\s+Pr[áa]ctica\s+tipo\s+B\s+(\d+)\s+Por\s+Promedio\s+Pb\s*=\s*(\d+)(?:\s+(\d+))?", compact, flags=re.IGNORECASE)
    if practice:
        eliminable = practice.group(3)
        item = f"Práctica tipo B (Pb): {practice.group(1)} evaluaciones; peso Pb={practice.group(2)}"
        if eliminable is not None:
            item += f"; evaluaciones eliminables: {eliminable}"
        components.append(item + ".")
    task = re.search(r"\bTa\s+Tarea\s+acad[ée]mica\s+(\d+)\s+Por\s+Promedio\s+Ta\s*=\s*(\d+)(?:\s+(\d+))?", compact, flags=re.IGNORECASE)
    if task:
        item = f"Tarea académica (Ta): {task.group(1)} evaluación; peso Ta={task.group(2)}"
        if task.group(3) is not None:
            item += f"; evaluaciones eliminables: {task.group(3)}"
        components.append(item + ".")
    exam = re.search(r"\bEx\s+Examen\s+(\d+)\s+Por\s+Evaluaci[óo]n\s+Ex1\s*=\s*(\d+)\s+Ex2\s*=\s*(\d+)", compact, flags=re.IGNORECASE)
    if exam:
        components.append(f"Examen (Ex): {exam.group(1)} evaluaciones; pesos Ex1={exam.group(2)} y Ex2={exam.group(3)}.")
    return components


def extract_ai_policy(evaluation_section: str) -> str:
    match = re.search(r"Lineamientos\s+sobre\s+el\s+uso\s+de\s+la\s+IA\.?(.*)$", evaluation_section, flags=re.IGNORECASE)
    if not match:
        return ""
    return clean_syllabus_section(match.group(1))


def extract_bibliography(text: str, sections: dict[str, str] | None = None) -> list[str]:
    section = (sections or {}).get("bibliografia", "")
    if not section:
        section = extract_between(text, r"IX\.\s*BIBLIOGRAF[ÍI]A", r"\n\s*X\.")
    if not section:
        return []
    section = clean_section_body(section)
    section = re.sub(r"FACULTAD DE CIENCIAS E INGENIER[ÍI]A", " ", section, flags=re.IGNORECASE)
    section = re.sub(r"https?://\S+", " ", section)

    if re.search(r"\n\s*-\s*Libro\s*\n", section, flags=re.IGNORECASE):
        blocks = re.split(r"\n\s*-\s*Libro\s*\n", section, flags=re.IGNORECASE)
    else:
        blocks = re.split(r"(?:^|\n|\s)Libro\s+", section, flags=re.IGNORECASE)

    items = []
    for block in blocks[1:]:
        block = re.split(r"(?:Referencia\s+complementaria|X\.\s*)", block, maxsplit=1, flags=re.IGNORECASE)[0]
        if "Referencia obligatoria" in block:
            block = block.split("Referencia obligatoria", 1)[-1]
        lines = [re.sub(r"\s+", " ", line).strip(" .") for line in block.splitlines()]
        if len(lines) <= 1:
            lines = [part.strip(" .") for part in re.split(r"(?<=\.)\s+", clean_inline_text(block)) if part.strip(" .")]
        lines = [line for line in lines if line and line.lower() not in {"one", "ne", "referencia obligatoria"} and "SD_ILS" not in line]
        if not lines:
            continue
        item = ". ".join(lines[:5]).strip(" .") + "."
        item = clean_inline_text(item) + "."
        if item not in items:
            items.append(item)
        if len(items) >= 8:
            break
    return items


def extract_syllabus_details(text: str, sections: dict[str, str] | None = None) -> dict[str, Any]:
    sections = sections or {}
    evaluation = extract_evaluation_section(text, sections)
    return {
        "objetivos": extract_objectives(text, sections),
        "metodologia": extract_methodology(text, sections),
        "evaluacion": {
            "componentes": extract_evaluation_components(evaluation),
            "formula_nota_final": extract_final_grade_formula(evaluation),
            "lineamientos_ia": extract_ai_policy(evaluation),
            "texto": evaluation,
        },
        "bibliografia": extract_bibliography(text, sections),
    }

def format_topic(title: str, details: list[str]) -> str:
    title = re.sub(r"\s+", " ", title).strip(" .")
    title = re.sub(r"CAP[ÍI]TULO\s+(\d+)\s+CAP[ÍI]TULO\s+\1\s*:?", r"Capítulo \1:", title, flags=re.IGNORECASE)
    title = re.sub(r"CAP[ÍI]TULO\s+(\d+)\s*[.:]?", r"Capítulo \1:", title, flags=re.IGNORECASE)
    title = re.sub(r"UNIDAD\s+(\d+)\s*:?", r"Unidad \1:", title, flags=re.IGNORECASE)
    title = re.sub(r"SESI[ÓO]N\s+(\d+)\s+SESI[ÓO]N\s+\1\s*:?", r"Sesión \1:", title, flags=re.IGNORECASE)
    title = re.sub(r"SESI[ÓO]N\s+(\d+)\s*:?", r"Sesión \1:", title, flags=re.IGNORECASE)
    title = re.sub(r":\s*:", ":", title)
    title = re.sub(r":\s*\.", ":", title)
    title = re.sub(r":(?=\S)", ": ", title)

    useful_details = []
    for detail in details:
        detail = re.sub(r"\s+", " ", detail).strip(" .")
        if not detail:
            continue
        useful_details.append(detail)
        if len(useful_details) >= 3:
            break
    if useful_details:
        return f"{title}: {'; '.join(useful_details)}."
    return f"{title}."


def topic_sort_key(topic: str) -> tuple[int, str]:
    match = re.match(r"(?:Capítulo|Unidad|Sesión)\s+(\d+)", topic, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), topic
    return 10_000, topic


def extract_program_topics(text: str) -> list[str]:
    title_pattern = r"(?:CAP[ÍI]TULO|UNIDAD|SESI[ÓO]N)\s+\d+"
    section = extract_between(text, r"VI\.\s*PROGRAMA ANAL[ÍI]TICO", r"\n\s*VII\.")
    if not section:
        section = extract_between(text, r"\bCONTENIDO\s+TEM[ÁA]TICO\b", r"\n\s*(?:DESCRIPCI[ÓO]N|BIBLIOGRAF[ÍI]A)\b")
    if not section and re.search(title_pattern, text, flags=re.IGNORECASE):
        section = text
    if not section:
        return []

    section = re.sub(r"FACULTAD DE CIENCIAS E INGENIER[ÍI]A", " ", section, flags=re.IGNORECASE)
    section = re.sub(r"[ \t]+", " ", section)
    lines = [line.strip(" -") for line in section.splitlines() if line.strip(" -")]

    topics = []
    current_title = ""
    current_details = []
    for line in lines:
        if re.match(title_pattern, line, flags=re.IGNORECASE):
            if current_title:
                topics.append(format_topic(current_title, current_details))
            current_title = line
            current_details = []
            continue
        if not current_title:
            continue
        title_incomplete = not re.search(r"\(\d+\s*horas?\)", current_title, flags=re.IGNORECASE)
        starts_detail = re.match(r"(?:objetivo|contenido):", line, flags=re.IGNORECASE)
        if title_incomplete and not starts_detail:
            current_title = f"{current_title} {line}"
            continue
        current_details.append(line)
    if current_title:
        topics.append(format_topic(current_title, current_details))
    return sorted((topic for topic in topics if topic), key=topic_sort_key)


def read_pdf_text(pdf_path: Path) -> str:
    pages = PyPDFLoader(str(pdf_path)).load()
    return "\n".join(page.page_content for page in pages)


def build_records() -> list[dict[str, Any]]:
    courses = load_curriculum_courses()
    records = []
    for pdf_path in sorted(DYNAMIC_DOCS_DIR.glob("*.PDF")) + sorted(DYNAMIC_DOCS_DIR.glob("*.pdf")):
        code_match = re.search(r"([A-Z]{0,3}\d{3}|\d[A-Z]{3}\d{2})", pdf_path.name.upper())
        if not code_match:
            continue
        code = code_match.group(1)
        course = courses.get(code, {})
        name = course.get("nombre", "") or COURSE_NAME_FALLBACKS.get(code, "")
        try:
            text = read_pdf_text(pdf_path)
        except Exception as exc:
            print(f"No se pudo procesar {pdf_path}: {exc}")
            continue

        if not name:
            title_match = re.search(rf"{re.escape(code)}\s*-\s*(.+)", text)
            if title_match:
                name = title_match.group(1).strip().title()
        if not name:
            colon_title_match = re.search(rf":\s*([^\n:]+?)\s*\n\s*:\s*{re.escape(code)}\b", text, flags=re.IGNORECASE)
            if colon_title_match:
                name = colon_title_match.group(1).strip().title()
        if not name:
            first_title_match = re.search(r"^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ]{6,})$", text, flags=re.MULTILINE)
            if first_title_match:
                name = first_title_match.group(1).strip().title()
        if not name:
            name = code

        sections = split_numbered_sections(text, code, name)
        clean_sections = {key: clean_section_for_embedding(value) for key, value in sections.items() if clean_section_for_embedding(value)}
        sumilla = clean_sections.get("sumilla") or clean_sumilla(text)
        topics = extract_program_topics(sections.get("programa_analitico", "") or text)
        if not sumilla and not topics:
            continue

        records.append({
            "codigo": code,
            "nombre": name,
            "fuente": str(pdf_path),
            "sumilla": sumilla,
            "programa_analitico": topics,
            "secciones_limpias": clean_sections,
            "detalles_silabo": extract_syllabus_details(text, clean_sections),
        })
    return records


def write_outputs(records: list[dict[str, Any]]) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Resumen procesado de sílabos 2026-1",
        "",
        "Archivo generado por `scripts/preprocess_syllabi.py` desde los PDFs en `docs/dynamic`.",
        "",
    ]
    for record in records:
        lines.append(f"## {record['nombre']} ({record['codigo']})")
        lines.append(f"Fuente: {record['fuente']}")
        if record.get("sumilla"):
            lines.append(f"Sumilla: {record['sumilla']}")
        if record.get("programa_analitico"):
            lines.append("Programa analítico:")
            for topic in record["programa_analitico"]:
                lines.append(f"- {topic}")
        sections = record.get("secciones_limpias") or {}
        if sections:
            lines.append("Secciones limpias disponibles:")
            for section_name in sections:
                lines.append(f"- {section_name}")
        details = record.get("detalles_silabo") or {}
        if details.get("objetivos"):
            lines.append("Objetivos / resultados de aprendizaje:")
            for objective in details["objetivos"]:
                lines.append(f"- {objective}")
        evaluation = details.get("evaluacion") or {}
        if evaluation.get("componentes") or evaluation.get("formula_nota_final"):
            lines.append("Evaluación:")
            for component in evaluation.get("componentes") or []:
                lines.append(f"- {component}")
            if evaluation.get("formula_nota_final"):
                lines.append(f"- Fórmula de nota final: {evaluation['formula_nota_final']}")
        if details.get("bibliografia"):
            lines.append("Bibliografía:")
            for item in details["bibliografia"][:5]:
                lines.append(f"- {item}")
        lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = build_records()
    write_outputs(records)
    print(f"Sílabos procesados: {len(records)}")
    print(f"Salida JSON: {OUTPUT_JSON}")
    print(f"Salida Markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
