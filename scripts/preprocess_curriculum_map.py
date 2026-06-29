from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

MAIN_CYCLE_RANGE = set(range(5, 11))
COURSE_CODE_PATTERN = re.compile(r"\b(?:\d[A-Z]{3}\d{2}|[A-Z]{3}\d{3})\b")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def title_course_name(name: str) -> str:
    small_words = {"de", "del", "la", "las", "los", "y", "en", "para", "con", "e"}
    words = normalize_spaces(name).lower().split()
    titled = []
    for i, word in enumerate(words):
        titled.append(word if i > 0 and word in small_words else word.capitalize())
    return " ".join(titled)


def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return normalize_spaces(text)


def extract_pdf_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout or ""


def load_syllabus_courses(dynamic_dir: Path) -> dict[str, dict[str, Any]]:
    courses: dict[str, dict[str, Any]] = {}
    for pdf_path in sorted(dynamic_dir.glob("**/*.[pP][dD][fF]")):
        text = extract_pdf_text(pdf_path)
        course_match = re.search(r"^CURSO\s+(.+)$", text, flags=re.MULTILINE)
        code_match = re.search(r"^CLAVE\s+(.+)$", text, flags=re.MULTILINE)
        credits_match = re.search(r"^CRÉDITOS\s+(.+)$", text, flags=re.MULTILINE)
        if not course_match or not code_match:
            continue
        code = normalize_spaces(code_match.group(1)).upper()
        courses[code] = {
            "codigo": code,
            "nombre": title_course_name(course_match.group(1)),
            "creditos_silabo": normalize_spaces(credits_match.group(1)) if credits_match else "",
            "fuente_silabo": str(pdf_path),
        }
    return courses


def strip_course_name_tail(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*$", "", text).strip()
    text = re.sub(r"^(?:\d+(?:\.\d+)?(?:\s*\(\d+q\))?\s*)+", "", text).strip()
    return text


def parse_course_line(line: str, current_cycle: int | None, elective: bool) -> tuple[int | None, dict[str, Any] | None]:
    match = re.search(r"^(?P<prefix>\s*(?P<cycle>\d{1,2})?\s*)(?P<code>(?:\d[A-Z]{3}\d{2}|[A-Z]{3}\d{3}))\s+(?P<rest>.+)$", line)
    if not match:
        return current_cycle, None

    cycle_text = match.group("cycle")
    if cycle_text and int(cycle_text) in MAIN_CYCLE_RANGE:
        current_cycle = int(cycle_text)

    code = match.group("code").upper()
    rest = match.group("rest").rstrip()
    credits_match = re.search(r"(\d+\.\d+)\s*$", rest)
    credits = credits_match.group(1) if credits_match else ""
    before_credits = rest[: credits_match.start()].rstrip() if credits_match else rest

    req_text = ""
    name_part = before_credits
    req_markers = [
        r"\d+\s*\(\d+q\)",
        r"\d+(?:\.\d+)?\s+créditos aprobados\s*\*?",
        r"Acreditar capacidad de lectura",
        COURSE_CODE_PATTERN.pattern,
        r"Mínimo",
    ]
    marker_positions = []
    for pattern in req_markers:
        found = re.search(pattern, before_credits)
        if found:
            position = found.start()
            if pattern == COURSE_CODE_PATTERN.pattern and position > 0 and before_credits[position - 1] in "[({":
                position -= 1
            marker_positions.append(position)
    if marker_positions:
        split_at = min(marker_positions)
        name_part = before_credits[:split_at].rstrip()
        req_text = before_credits[split_at:].strip()

    return current_cycle, {
        "codigo": code,
        "nombre_extraido": strip_course_name_tail(name_part),
        "ciclo": current_cycle,
        "categoria": "electivo" if elective else "obligatorio",
        "creditos": credits,
        "requisitos_texto": normalize_spaces(req_text),
        "requisitos_codigos": COURSE_CODE_PATTERN.findall(req_text),
        "linea_fuente": normalize_spaces(line),
    }


def attach_wrapped_names(lines: list[str], entries: list[dict[str, Any]]) -> None:
    index_by_code = {entry["codigo"]: entry for entry in entries}
    for i, line in enumerate(lines):
        code_match = COURSE_CODE_PATTERN.search(line)
        if not code_match:
            continue
        code = code_match.group(0)
        entry = index_by_code.get(code)
        if not entry or entry.get("nombre_extraido"):
            continue

        fragments = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(lines) and not COURSE_CODE_PATTERN.search(lines[j]):
                fragment = normalize_spaces(lines[j])
                if fragment and not fragment.startswith(("CI ", "CT ", "ME ", "(", "[", "*", "**")):
                    fragments.append(fragment)
        if fragments:
            entry["nombre_extraido"] = normalize_spaces(" ".join(fragments))


def parse_curriculum_pdf(malla_pdf: Path, dynamic_dir: Path) -> dict[str, Any]:
    text = extract_pdf_text(malla_pdf)
    lines = text.splitlines()
    syllabus_courses = load_syllabus_courses(dynamic_dir)
    entries: list[dict[str, Any]] = []
    current_cycle: int | None = None
    elective = False

    for line in lines:
        if "ELECTIVOS DE LA ESPECIALIDAD" in line:
            elective = True
            current_cycle = None
            continue
        current_cycle, entry = parse_course_line(line, current_cycle, elective)
        if entry:
            entries.append(entry)

    attach_wrapped_names(lines, entries)

    by_code: dict[str, dict[str, Any]] = {}
    for entry in entries:
        code = entry["codigo"]
        syllabus = syllabus_courses.get(code, {})
        name = syllabus.get("nombre") or title_course_name(entry.get("nombre_extraido", ""))
        if code in by_code:
            continue
        by_code[code] = {
            "codigo": code,
            "nombre": name,
            "ciclo": entry.get("ciclo"),
            "categoria": entry.get("categoria"),
            "creditos": entry.get("creditos") or syllabus.get("creditos_silabo", ""),
            "requisitos_texto": entry.get("requisitos_texto", ""),
            "requisitos_codigos": entry.get("requisitos_codigos", []),
            "fuente": str(malla_pdf),
            "fuente_silabo": syllabus.get("fuente_silabo", ""),
        }

    cycles: dict[str, list[str]] = {}
    for course in by_code.values():
        cycle = course.get("ciclo")
        if cycle is not None:
            cycles.setdefault(str(cycle), []).append(course["codigo"])

    return {
        "fuente": str(malla_pdf),
        "descripcion_requisitos": {
            "parentesis": "( ) Haber cursado con nota 08 o más",
            "corchetes": "[ ] Haber cursado o cursar simultáneamente",
            "llaves": "{ } Haber aprobado o cursar simultáneamente, según sílabos donde aparece como tipo 05",
        },
        "ciclos": {cycle: sorted(codes) for cycle, codes in sorted(cycles.items(), key=lambda item: int(item[0]))},
        "cursos": dict(sorted(by_code.items())),
        "prerrequisito_de": build_prerequisite_index(by_code),
    }


def build_prerequisite_index(courses: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for course in courses.values():
        cycle = course.get("ciclo")
        if cycle not in MAIN_CYCLE_RANGE:
            continue
        for req_code in course.get("requisitos_codigos", []):
            index.setdefault(req_code, []).append(course["codigo"])
    return {code: sorted(set(dependent_codes)) for code, dependent_codes in sorted(index.items())}


def write_outputs(curriculum: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "malla_informatica.json"
    json_path.write_text(json.dumps(curriculum, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    courses = curriculum["cursos"]
    lines = ["# Malla curricular de Ingeniería Informática", "", f"Fuente: `{curriculum['fuente']}`", ""]
    for cycle, codes in curriculum["ciclos"].items():
        lines.append(f"## Ciclo {cycle}")
        for code in codes:
            course = courses[code]
            req = course.get("requisitos_texto") or "sin requisitos en la malla"
            lines.append(f"- {course['nombre']} ({code}) - {course.get('creditos', '')} créditos. Requisitos: {req}.")
        lines.append("")
    lines.append("## Electivos")
    for course in courses.values():
        if course.get("categoria") == "electivo":
            req = course.get("requisitos_texto") or "sin requisitos en la malla"
            lines.append(f"- {course['nombre']} ({course['codigo']}) - {course.get('creditos', '')} créditos. Requisitos: {req}.")
    lines.append("")

    lines.append("## Índice de prerrequisitos")
    lines.append("Cursos que registran cada código como requisito en la malla procesada:")
    for req_code, dependent_codes in curriculum.get("prerrequisito_de", {}).items():
        formatted = ", ".join(f"{courses[code]['nombre']} ({code})" for code in dependent_codes if code in courses)
        if formatted:
            lines.append(f"- {req_code}: {formatted}.")
    lines.append("")
    lines.append("Si una pregunta solicita ciclo, código, créditos o requisitos, usar esta malla antes que el historial de Discord.")
    (output_dir / "malla_informatica.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Procesa la malla curricular de Ingeniería Informática PUCP.")
    parser.add_argument("--malla-pdf", default="docs/static/MALLA_INFORMATICA_FCI_2026-1.pdf")
    parser.add_argument("--dynamic-dir", default="docs/dynamic")
    parser.add_argument("--output-dir", default="docs/static/processed")
    args = parser.parse_args()

    curriculum = parse_curriculum_pdf(Path(args.malla_pdf), Path(args.dynamic_dir))
    write_outputs(curriculum, Path(args.output_dir))
    print(f"Cursos procesados: {len(curriculum['cursos'])}")
    print(f"Ciclos obligatorios: {', '.join(curriculum['ciclos'].keys())}")


if __name__ == "__main__":
    main()
