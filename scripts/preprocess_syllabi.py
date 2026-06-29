import json
import re
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader

CURRICULUM_PATH = Path("docs/static/processed/malla_informatica.json")
DYNAMIC_DOCS_DIR = Path("docs/dynamic")
OUTPUT_JSON = Path("docs/static/processed/silabos_resumen.json")
OUTPUT_MD = Path("docs/static/processed/silabos_resumen.md")


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    section = re.sub(r"FACULTAD DE CIENCIAS E INGENIER[ÍI]A", " ", section, flags=re.IGNORECASE)
    section = re.sub(r"\s+", " ", section)
    return section.strip()


def format_topic(title: str, details: list[str]) -> str:
    title = re.sub(r"\s+", " ", title).strip(" .")
    title = re.sub(r"CAP[ÍI]TULO\s+(\d+)\s+CAP[ÍI]TULO\s+\1\s*:?", r"Capítulo \1:", title, flags=re.IGNORECASE)
    title = re.sub(r"CAP[ÍI]TULO\s+(\d+)\s*:?", r"Capítulo \1:", title, flags=re.IGNORECASE)
    title = re.sub(r"UNIDAD\s+(\d+)\s*:?", r"Unidad \1:", title, flags=re.IGNORECASE)
    title = re.sub(r":\s*:", ":", title)
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


def extract_program_topics(text: str) -> list[str]:
    section = extract_between(text, r"VI\.\s*PROGRAMA ANAL[ÍI]TICO", r"\n\s*VII\.")
    if not section:
        return []

    section = re.sub(r"FACULTAD DE CIENCIAS E INGENIER[ÍI]A", " ", section, flags=re.IGNORECASE)
    section = re.sub(r"[ \t]+", " ", section)
    lines = [line.strip(" -") for line in section.splitlines() if line.strip(" -")]

    topics = []
    current_title = ""
    current_details = []
    title_pattern = r"(?:CAP[ÍI]TULO|UNIDAD)\s+\d+"
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
    return [topic for topic in topics if topic]


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
        name = course.get("nombre", "")
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
            name = code

        sumilla = clean_sumilla(text)
        topics = extract_program_topics(text)
        if not sumilla and not topics:
            continue

        records.append({
            "codigo": code,
            "nombre": name,
            "fuente": str(pdf_path),
            "sumilla": sumilla,
            "programa_analitico": topics,
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
