from __future__ import annotations

import argparse
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

AMBIGUOUS_ALIAS_KEYS = {"diseno"}

DEFAULT_ALIAS_SEEDS = [
    {"alias": "tp", "nombre": "Técnicas de Programación", "codigo": "INF144", "tipo": "curso"},
    {"alias": "tecnicas de programacion", "nombre": "Técnicas de Programación", "codigo": "INF144", "tipo": "curso"},
    {"alias": "funpro", "nombre": "Fundamentos de Programación", "codigo": "1INF01", "tipo": "curso"},
    {"alias": "fpro", "nombre": "Fundamentos de Programación", "codigo": "1INF01", "tipo": "curso"},
    {"alias": "fundamentos de programacion", "nombre": "Fundamentos de Programación", "codigo": "1INF01", "tipo": "curso"},
    {"alias": "ed", "nombre": "Estructuras Discretas", "codigo": "INF134", "tipo": "curso"},
    {"alias": "estructuras discretas", "nombre": "Estructuras Discretas", "codigo": "INF134", "tipo": "curso"},
    {"alias": "adso", "nombre": "Administración de Sistemas Operativos", "codigo": "1INF35", "tipo": "curso"},
    {"alias": "aso", "nombre": "Administración de Sistemas Operativos", "codigo": "1INF35", "tipo": "curso"},
    {"alias": "so", "nombre": "Sistemas Operativos", "codigo": "1INF29", "tipo": "curso"},
    {"alias": "sisops", "nombre": "Sistemas Operativos", "codigo": "1INF29", "tipo": "curso"},
    {"alias": "sisop", "nombre": "Sistemas Operativos", "codigo": "1INF29", "tipo": "curso"},
    {"alias": "p3", "nombre": "Programación 3", "codigo": "1INF30", "tipo": "curso"},
    {"alias": "progra3", "nombre": "Programación 3", "codigo": "1INF30", "tipo": "curso"},
    {"alias": "prog3", "nombre": "Programación 3", "codigo": "1INF30", "tipo": "curso"},
    {"alias": "dp1", "nombre": "Proyecto de Fin de Carrera 1", "codigo": "1INF42", "tipo": "curso", "fuente": "vocabulario estudiantil validado"},
    {"alias": "formulacion", "nombre": "Formulación de Proyecto de Fin de Carrera", "codigo": "1INF26", "tipo": "curso"},
    {"alias": "dp2", "nombre": "Proyecto de Implementación de Software", "codigo": "1INF47", "tipo": "curso"},
    {"alias": "ingesoft", "nombre": "Ingeniería de Software", "codigo": "1INF37", "tipo": "curso"},
    {"alias": "ingsoft", "nombre": "Ingeniería de Software", "codigo": "1INF37", "tipo": "curso"},
    {"alias": "psp", "nombre": "Práctica Supervisada Preprofesional", "codigo": "INF008", "tipo": "curso/trámite"},
    {"alias": "bd", "nombre": "Base de Datos", "codigo": "1INF33", "tipo": "curso"},
    {"alias": "arquicom", "nombre": "Arquitectura de Computadoras", "codigo": "1ELE01", "tipo": "curso"},
    {"alias": "ia", "nombre": "Inteligencia Artificial", "codigo": "1INF24", "tipo": "curso"},
    {"alias": "pei", "nombre": "Planeamiento Estratégico en Informática", "codigo": "", "tipo": "curso"},
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def alias_pattern(alias: str) -> re.Pattern[str]:
    normalized_alias = normalize_text(alias)
    return re.compile(rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])")


def extract_pdf_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout or ""


def clean_syllabus_field(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"^:+\s*", "", value).strip()


def extract_official_courses(dynamic_dir: Path) -> list[dict[str, str]]:
    courses = []
    for pdf_path in sorted(dynamic_dir.glob("**/*.[pP][dD][fF]")):
        text = extract_pdf_text(pdf_path)
        course_match = re.search(r"^CURSO\s+(.+)$", text, flags=re.MULTILINE)
        code_match = re.search(r"^CLAVE\s+(.+)$", text, flags=re.MULTILINE)
        if not course_match or not code_match:
            continue
        name = clean_syllabus_field(course_match.group(1)).title()
        code = clean_syllabus_field(code_match.group(1)).upper()
        courses.append({
            "alias": normalize_text(name),
            "nombre": name,
            "codigo": code,
            "tipo": "curso",
            "fuente": f"sílabo {pdf_path.name}",
        })
    return courses


def read_historical_records(historical_dir: Path) -> list[dict[str, Any]]:
    records = []
    for json_path in sorted(historical_dir.glob("**/*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                item = dict(item)
                item["_source"] = str(json_path)
                records.append(item)
    return records


def build_vocabulary(dynamic_dir: Path, historical_dir: Path, existing_vocab: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official_courses = extract_official_courses(dynamic_dir)
    by_alias: dict[str, dict[str, Any]] = {}

    if existing_vocab and existing_vocab.exists():
        for record in json.loads(existing_vocab.read_text(encoding="utf-8")):
            alias = normalize_text(str(record.get("alias", "")))
            if alias and alias not in AMBIGUOUS_ALIAS_KEYS:
                record["alias"] = alias
                by_alias[alias] = record

    for record in official_courses + DEFAULT_ALIAS_SEEDS:
        alias = normalize_text(record["alias"])
        merged = dict(by_alias.get(alias, {}))
        merged.update(record)
        merged["alias"] = alias
        by_alias[alias] = merged

    records = read_historical_records(historical_dir)
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, str] = {}
    normalized_docs = []
    for record in records:
        text = " ".join(str(record.get(field, "")) for field in ("pregunta", "respuesta"))
        normalized = normalize_text(text)
        normalized_docs.append((record, normalized))

    for alias, info in by_alias.items():
        pattern = alias_pattern(alias)
        official_name = normalize_text(str(info.get("nombre", "")))
        official_pattern = alias_pattern(official_name) if official_name else None
        for record, normalized in normalized_docs:
            if pattern.search(normalized) or (official_pattern and official_pattern.search(normalized)):
                counts[alias] += 1
                if alias not in examples:
                    examples[alias] = str(record.get("pregunta") or record.get("respuesta") or "")[:240]

    vocabulary = []
    for alias, info in sorted(by_alias.items(), key=lambda item: (str(item[1].get("codigo", "")), item[0])):
        item = dict(info)
        item["alias"] = alias
        if not item.get("fuente"):
            item["fuente"] = "vocabulario estudiantil validado"
        item["menciones_discord"] = counts.get(alias, 0)
        if alias in examples:
            item["ejemplo_discord"] = examples[alias]
        vocabulary.append(item)

    detected = [item for item in vocabulary if item.get("menciones_discord", 0) > 0]
    return vocabulary, detected


def write_outputs(vocabulary: list[dict[str, Any]], detected: list[dict[str, Any]], curated_dir: Path) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    (curated_dir / "vocabulario_cursos.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Vocabulario de cursos y abreviaturas",
        "",
        "Archivo generado por `scripts/preprocess_discord_courses.py` a partir de sílabos y del histórico de Discord cargado.",
        "",
    ]
    for item in vocabulary:
        code = f" ({item['codigo']})" if item.get("codigo") else ""
        mentions = item.get("menciones_discord", 0)
        source = item.get("fuente", "")
        lines.append(f"- {item['alias']}: {item['nombre']}{code}. Menciones DC: {mentions}. Fuente: {source}.")
    lines.append("")
    (curated_dir / "vocabulario_cursos.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    detected_lines = ["# Cursos detectados en el histórico de Discord", ""]
    for item in sorted(detected, key=lambda record: (-record.get("menciones_discord", 0), record.get("alias", ""))):
        code = f" ({item['codigo']})" if item.get("codigo") else ""
        detected_lines.append(f"- {item['alias']}: {item['nombre']}{code} - {item.get('menciones_discord', 0)} menciones")
    (curated_dir / "cursos_detectados_discord.md").write_text("\n".join(detected_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae vocabulario de cursos desde sílabos e histórico de Discord.")
    parser.add_argument("--dynamic-dir", default="docs/dynamic")
    parser.add_argument("--historical-dir", default="docs/historical")
    parser.add_argument("--curated-dir", default="docs/curated")
    args = parser.parse_args()

    curated_dir = Path(args.curated_dir)
    vocabulary, detected = build_vocabulary(
        dynamic_dir=Path(args.dynamic_dir),
        historical_dir=Path(args.historical_dir),
        existing_vocab=curated_dir / "vocabulario_cursos.json",
    )
    write_outputs(vocabulary, detected, curated_dir)
    print(f"Vocabulario generado: {len(vocabulary)} aliases/cursos")
    print(f"Cursos/aliases detectados en Discord: {len(detected)}")


if __name__ == "__main__":
    main()
