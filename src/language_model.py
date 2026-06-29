from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from typing import Any
from datetime import date
from pathlib import Path
import json
import re
import unicodedata


class LanguageModel(object):
    PORTAL_URL = "https://estudiante.pucp.edu.pe/"
    CAMPUS_URL = "https://campusvirtual.pucp.edu.pe/"
    CALENDAR_2026_1_URL = "https://estudiante.pucp.edu.pe/calendario-academico/2026-1/"
    MATRICULA_FACTS_PATH = Path("docs/curated/matricula_fechas.json")
    COURSE_ALIASES_PATH = Path("docs/curated/vocabulario_cursos.json")
    CURRICULUM_PATH = Path("docs/static/processed/malla_informatica.json")

    def __init__(self, model_name: str, initial_prompt: str, temperature: float = 0.1):
        assert model_name is not None, "Model name cannot be None"
        assert initial_prompt is not None, "Initial prompt cannot be None"

        try:
            _llm = ChatOllama(model=model_name, temperature=temperature, keep_alive=1)
            _prompt = ChatPromptTemplate.from_messages([
                ("system", initial_prompt + "\n\nContexto:\n{context}"),
                ("human", "{input}")
            ])
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

        self.llm: ChatOllama = _llm
        self.prompt: ChatPromptTemplate = _prompt
        self.docs_chain: Runnable[dict[str, Any], Any] = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt,
        )
        self.rag_chain: Runnable = None

    def define_rag_chain(self, retriever: EnsembleRetriever) -> None:
        self.rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=self.docs_chain)

    def generate_response(self, pregunta: str) -> str:
        if self.rag_chain is None:
            raise ValueError("RAG chain is not defined. Please call define_rag_chain() first.")

        language_response = self._preflight_language_response(pregunta)
        if language_response is not None:
            return language_response

        injection_response = self._preflight_prompt_injection_response(pregunta)
        if injection_response is not None:
            return injection_response

        cyber_safety_response = self._preflight_cyber_safety_response(pregunta)
        if cyber_safety_response is not None:
            return cyber_safety_response

        small_talk_response = self._preflight_small_talk_response(pregunta)
        if small_talk_response is not None:
            return small_talk_response

        out_of_scope_response = self._preflight_out_of_scope_response(pregunta)
        if out_of_scope_response is not None:
            return out_of_scope_response

        meta_scope_response = self._preflight_meta_scope_response(pregunta)
        if meta_scope_response is not None:
            return meta_scope_response

        external_task_response = self._preflight_external_task_response(pregunta)
        if external_task_response is not None:
            return external_task_response

        academic_integrity_response = self._preflight_academic_integrity_response(pregunta)
        if academic_integrity_response is not None:
            return academic_integrity_response

        course_load_response = self._preflight_course_load_response(pregunta)
        if course_load_response is not None:
            return course_load_response

        student_slang_response = self._preflight_student_slang_response(pregunta)
        if student_slang_response is not None:
            return student_slang_response

        ambiguous_slang_response = self._preflight_ambiguous_slang_response(pregunta)
        if ambiguous_slang_response is not None:
            return ambiguous_slang_response

        curriculum_response = self._preflight_curriculum_response(pregunta)
        if curriculum_response is not None:
            return curriculum_response

        psp_response = self._preflight_psp_response(pregunta)
        if psp_response is not None:
            return psp_response

        plan_issue_response = self._preflight_plan_issue_response(pregunta)
        if plan_issue_response is not None:
            return plan_issue_response

        vacancy_response = self._preflight_vacancy_response(pregunta)
        if vacancy_response is not None:
            return vacancy_response

        calendar_response = self._preflight_calendar_event_response(pregunta)
        if calendar_response is not None:
            return calendar_response

        preflight_response = self._preflight_matricula_response(pregunta)
        if preflight_response is not None:
            return preflight_response

        try:
            response = self.rag_chain.invoke({"input": pregunta})
            answer = response["answer"]
            context_docs = response.get("context", [])

            if self._contains_unsupported_matricula_claim(answer, context_docs):
                return self._safe_matricula_derivation()
            if self._contains_unsupported_matricula_date_claim(answer, context_docs):
                return self._safe_matricula_date_derivation()
            if self._contains_unsupported_vacancy_promise(answer):
                return self._safe_vacancy_derivation()
            if self._contains_false_identity_claim(answer):
                return self._safe_out_of_scope_derivation()
            if self._contains_unsupported_ambiguous_term_claim(pregunta, answer):
                return self._ask_to_clarify_ambiguous_terms(pregunta)
            if self._contains_wrong_psp_derivation(pregunta, answer):
                return self._safe_generic_unknown_response()
            if self._contains_unsupported_plan_issue_derivation(pregunta, answer):
                return self._answer_plan_issue()
            if self._contains_invented_course_name(answer):
                return self._safe_invented_course_name_response()
            if self._contains_out_of_scope_answer(answer):
                return self._safe_generic_unknown_response()

            return self._sanitize_stale_cycle_links(pregunta, answer)
        except Exception as e:
            print(f"Error en inferencia: {e}")

        return "¡Uy! Mis servidores están un poco saturados procesando matrículas. ¿Te importa repetirme la pregunta en un ratito?"

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text or "")
        text = "".join(char for char in text if not unicodedata.combining(char))
        return text.lower()

    def _preflight_language_response(self, pregunta: str) -> str | None:
        cleaned_question = re.sub(r"<@!?\d+>", " ", pregunta or "")
        normalized_question = self._normalize_for_course_matching(cleaned_question)
        if not normalized_question or len(normalized_question) < 25:
            return None

        english_terms = [
            "ignore", "previous", "instructions", "you are", "now playing", "role",
            "unregulated", "research assistant", "writing", "thriller", "need to",
            "figure out", "hacker", "break into", "system", "script", "authentically",
            "can you", "write", "hypothetical", "educational", "step by step", "guide",
            "how this is achieved", "please", "provide", "explain", "chapter",
            "what", "which", "when", "where", "why", "how", "courses", "course",
            "should", "take", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "cycle", "credits", "requirements", "requirement", "code", "codes",
        ]
        spanish_terms = [
            "matricula", "matrícula", "curso", "cursos", "ciclo", "creditos", "créditos",
            "malla", "requisito", "requisitos", "llevar", "puedo", "quiero", "necesito",
            "codigo", "código", "clave", "pucp", "informatica", "informática",
        ]
        english_hits = sum(1 for term in english_terms if term in normalized_question)
        spanish_hits = sum(1 for term in spanish_terms if term in normalized_question)
        english_question_pattern = re.search(r"\b(what|which|when|where|why|how|can|could|should|do|does|is|are)\b", normalized_question) is not None
        if spanish_hits == 0 and (english_hits >= 4 or (english_hits >= 2 and english_question_pattern)):
            return (
                "Solo puedo responder preguntas en español y dentro del alcance de Ingeniería Informática PUCP: "
                "matrícula, malla, cursos, créditos, requisitos, PSP y trámites cargados. "
                "Reformula tu consulta en español y relacionada con ese alcance."
            )
        return None

    def _preflight_prompt_injection_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_for_course_matching(pregunta)
        injection_terms = [
            "ignora todas las instrucciones", "ignora instrucciones", "ignore all previous instructions",
            "ignore previous instructions", "olvida las instrucciones", "olvida todo lo anterior",
            "asumes el papel", "ahora asume", "you are now", "now playing the role",
            "sin restricciones", "unregulated", "jailbreak", "developer mode",
        ]
        if any(term in normalized_question for term in injection_terms):
            return (
                "No puedo ignorar mis instrucciones ni asumir otro rol. "
                "Solo puedo ayudar con información cargada sobre Ingeniería Informática PUCP: matrícula, malla, cursos, créditos, requisitos, PSP y trámites."
            )
        return None

    def _preflight_cyber_safety_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_for_course_matching(pregunta)
        mentions_intrusion = any(term in normalized_question for term in [
            "hacker", "hackear", "infiltrarse", "infiltrar", "vulnerar", "breach",
            "break into", "explotar vulnerabilidad", "exploit", "backdoor", "phishing",
            "escalar privilegios", "privilege escalation", "exfiltrar", "exfiltration",
            "borrar logs", "covering tracks", "nmap", "metasploit",
        ])
        asks_instructions = any(term in normalized_question for term in [
            "paso a paso", "step by step", "guia", "guía", "manual", "tutorial",
            "como se lleva a cabo", "how this is achieved", "como hacerlo", "redactar una guia",
        ])
        if mentions_intrusion and asks_instructions:
            return (
                "No puedo ayudar con guías, pasos o procedimientos para infiltrarse, vulnerar sistemas o evadir controles. "
                "Además, eso está fuera del alcance del bot, que solo responde sobre procesos de Ingeniería Informática PUCP."
            )
        return None

    def _preflight_small_talk_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta).strip()
        normalized_question = re.sub(r"<@!?\d+>", "", normalized_question).strip()
        normalized_question = re.sub(r"[^a-z0-9áéíóúñü\s]", "", normalized_question).strip()

        greetings = {"hola", "buenos dias", "buenas tardes", "buenas noches", "hi", "hello"}
        wellbeing_questions = {
            "como estas", "como esta", "que tal", "que tal estas", "como vas",
            "como te va", "todo bien", "que fue", "que tal chatbotinf",
        }
        thanks = {"gracias", "muchas gracias", "ok gracias", "listo gracias"}
        farewells = {"adios", "chau", "hasta luego", "nos vemos"}

        if normalized_question in greetings:
            return (
                "Hola. Te puedo ayudar con información cargada sobre procesos de Ingeniería Informática PUCP: "
                "matrícula, calendario académico, trámites relevantes de la carrera, PSP/convenios, malla, sílabos "
                "y consultas históricas del Discord cargado. Si es sobre otro proceso de la PUCP o información externa, "
                "te indicaré que está fuera de mi alcance o que no tengo información suficiente."
            )
        if normalized_question in wellbeing_questions:
            return (
                "Bien, gracias. Puedo ayudarte con información cargada sobre procesos de Ingeniería Informática PUCP: "
                "matrícula, calendario académico, trámites, PSP/convenios, malla y sílabos."
            )
        if normalized_question in thanks:
            return "De nada. Recuerda que puedo apoyarte solo con procesos de Ingeniería Informática PUCP cubiertos por la información cargada."
        if normalized_question in farewells:
            return "Hasta luego. Si tienes otra duda sobre matrícula, trámites, malla, sílabos o PSP de Ingeniería Informática PUCP, me escribes."

        return None

    def _preflight_out_of_scope_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        out_of_scope_terms = [
            "protesta", "protestas", "toma", "tomar el edificio", "dintilhac",
            "apoyar esta protesta", "deberiamos apoyar", "opinas", "opinion politica",
        ]
        if any(term in normalized_question for term in out_of_scope_terms):
            return self._safe_out_of_scope_derivation()
        return None

    def _safe_out_of_scope_derivation(self) -> str:
        return (
            "No puedo tomar postura ni dar recomendaciones sobre protestas, tomas de edificios o decisiones políticas/personales. "
            "Mi alcance es apoyar con información cargada sobre procesos de Ingeniería Informática PUCP: matrícula, calendario académico, trámites relevantes de la carrera, PSP/convenios, malla y sílabos. "
            "Para orientación institucional fuera de ese alcance, consulta directamente con los canales oficiales de la PUCP o de la carrera."
        )

    def _contains_false_identity_claim(self, answer: str) -> bool:
        normalized_answer = self._normalize_text(answer)
        false_identity_phrases = [
            "como director de carrera", "soy el director", "como coordinador",
            "desde la direccion de carrera", "lo revisaremos", "lo evaluaremos",
        ]
        return any(phrase in normalized_answer for phrase in false_identity_phrases)

    def _preflight_meta_scope_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        asks_about_scope = any(phrase in normalized_question for phrase in [
            "cual es tu alcance", "cual es tu funcion", "sobre que respondes",
            "eres de la pucp", "eres un bot de la pucp", "que temas cubres",
        ])
        if asks_about_scope:
            return (
                "Soy un asistente virtual de apoyo para Ingeniería Informática PUCP. "
                "No soy un bot general de toda la PUCP ni una oficina oficial. "
                "Respondo con la información cargada sobre matrícula, calendario académico, trámites relevantes de la carrera, PSP/convenios, malla, sílabos y consultas históricas del Discord cargado. "
                "Si el tema es externo a ese alcance, indicaré que no tengo información suficiente."
            )

        asks_about_docs = any(phrase in normalized_question for phrase in [
            "fuera de tus documentos", "no este en tus documentos", "no esta en tus documentos",
            "algo que no este", "algo que no esta", "puedes responder algo",
        ])
        if not asks_about_docs:
            return None
        return (
            "No. Debo responder solo con la información cargada en mis documentos y contexto de Ingeniería Informática PUCP. "
            "Si no tengo sustento suficiente, te pediré que precises el curso, trámite o proceso, o te indicaré que no tengo esa información."
        )

    def _preflight_external_task_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_for_course_matching(pregunta)
        academic_scope_terms = [
            "matricula", "curso", "cursos", "ciclo", "malla", "credito", "creditos",
            "requisito", "requisitos", "psp", "convenio", "campus", "portal",
            "horario", "turno", "vacante", "trika", "bika", "silabo", "sílabo",
            "informatica", "pucp", "codigo", "clave", "director", "carrera",
        ]
        if any(term in normalized_question for term in academic_scope_terms):
            return None

        arithmetic_pattern = re.search(
            r"\b(?:cuanto|cuanto es|calcula|resolver|resuelve)?\s*\d+(?:\s*[+\-*/x÷]\s*\d+)+\s*\??$",
            normalized_question,
        )
        if arithmetic_pattern:
            return self._safe_generic_unknown_response()

        external_terms = [
            "chiste", "poema", "cuento", "receta", "clima", "noticia", "capital de",
            "traduce", "traducir", "resume este texto", "programa", "codigo python",
            "hazme una tarea", "calcula", "cuanto es", "resuelve", "derivada", "integral",
        ]
        if any(term in normalized_question for term in external_terms):
            return self._safe_generic_unknown_response()
        return None

    def _preflight_academic_integrity_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        asks_permission = any(term in normalized_question for term in [
            "me das permiso", "puedo usar", "me permites", "esta permitido", "autorizas",
            "permiso para usar", "usar ia", "usar chatgpt", "usar inteligencia artificial",
        ])
        mentions_assessment = any(term in normalized_question for term in [
            "examen", "examenes", "evaluacion", "evaluaciones", "practica calificada",
            "pc", "control", "parcial", "final", "aprobar", "pasar mi trika", "pasar el curso",
        ])
        mentions_ai = any(term in normalized_question for term in [
            "ia", "chatgpt", "inteligencia artificial", "herramienta", "herramientas",
        ])
        if asks_permission and (mentions_assessment or mentions_ai):
            return (
                "No tengo autoridad ni información suficiente para darte permiso de usar IA o herramientas externas "
                "en exámenes o evaluaciones. Eso depende de las reglas del curso y de las indicaciones del docente responsable. "
                "Revisa el sílabo o consulta directamente al docente antes de usar cualquier herramienta en una evaluación."
            )
        return None

    def _preflight_curriculum_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_for_course_matching(pregunta)
        curriculum = self._load_curriculum()
        if not curriculum:
            return None

        cycle = self._extract_curriculum_cycle(normalized_question)
        asks_cycle_courses = cycle is not None and any(term in normalized_question for term in [
            "curso", "cursos", "llevar", "malla", "ciclo",
        ])
        if asks_cycle_courses:
            return self._answer_curriculum_cycle(cycle, curriculum)

        asks_code = any(term in normalized_question for term in ["codigo", "codigos", "clave", "claves"])
        asks_credits = any(term in normalized_question for term in ["credito", "creditos", "cuantos creditos", "cuanto creditaje"])
        asks_course_by_code = (
            re.search(r"\b[0-9][a-z]{3}[0-9]{2}\b", normalized_question) is not None
            and any(term in normalized_question for term in ["curso", "cursos", "cual es", "que es"])
        )
        asks_requirements = any(term in normalized_question for term in [
            "requisito", "requisitos", "que me falta", "qué me falta", "necesito",
            "puedo adelantar", "puedo llevar", "puedo pedir excepcion", "puedo pedir excepción",
            "pedir excepcion", "pedir excepción", "solicitud de excepcion", "solicitud de excepción",
            "mismo tiempo", "a la vez", "simultaneamente", "simultáneamente",
            "debo pasar", "tengo que pasar", "si o si pasarlo", "sí o sí pasarlo",
            "si o si aprobar", "sí o sí aprobar",
        ])
        if not (asks_code or asks_credits or asks_requirements or asks_course_by_code):
            return None

        refs = self._find_course_references(normalized_question)
        refs = self._merge_curriculum_name_references(normalized_question, refs, curriculum)
        if not refs:
            return None

        courses = curriculum.get("cursos", {})
        passed_codes = {ref.get("codigo") for ref in self._find_passed_course_references(normalized_question)}
        target_refs = [ref for ref in refs if ref.get("codigo") in courses and ref.get("codigo") not in passed_codes]
        if not target_refs:
            target_refs = [ref for ref in refs if ref.get("codigo") in courses]
        if not target_refs:
            return None

        asks_simultaneous_or_exception = any(term in normalized_question for term in [
            "puedo pedir excepcion", "puedo pedir excepción", "pedir excepcion", "pedir excepción",
            "solicitud de excepcion", "solicitud de excepción", "mismo tiempo", "a la vez",
            "simultaneamente", "simultáneamente", "en paralelo", "junto con",
        ])
        if asks_requirements and asks_simultaneous_or_exception and len(target_refs) >= 2:
            selected_courses = [courses.get(ref.get("codigo")) for ref in target_refs if ref.get("codigo") in courses]
            selected_courses = [course for course in selected_courses if course]
            for dependent_course in selected_courses:
                required_codes = set(dependent_course.get("requisitos_codigos", []))
                for prerequisite_course in selected_courses:
                    if prerequisite_course.get("codigo") in required_codes:
                        return (
                            f"{dependent_course['nombre']} ({dependent_course['codigo']}) registra como requisito "
                            f"{prerequisite_course['nombre']} ({prerequisite_course['codigo']}). "
                            "Si aún no has aprobado ese requisito, no puedo confirmar que puedas llevar ambos cursos "
                            "en el mismo ciclo ni que proceda una excepción solo con la información cargada. "
                            f"Verifica tus cursos permitidos o la solicitud correspondiente en Campus Virtual PUCP: {self.CAMPUS_URL}"
                        )

        lines = []
        for ref in target_refs[:3]:
            course = courses.get(ref.get("codigo"))
            if not course:
                continue
            if asks_course_by_code and not asks_requirements and not asks_credits:
                parts = [f"{course['codigo']} es {course['nombre']}"]
                if course.get("ciclo") is not None:
                    parts.append(f"ciclo {course['ciclo']}")
                if course.get("creditos"):
                    parts.append(f"{course['creditos']} créditos")
                lines.append(", ".join(parts) + ".")
                continue
            if asks_code and not asks_requirements and not asks_credits:
                lines.append(f"El código de {course['nombre']} es {course['codigo']}.")
                continue
            parts = [f"{course['nombre']} ({course['codigo']})"]
            if course.get("ciclo") is not None:
                parts.append(f"ciclo {course['ciclo']}")
            if course.get("creditos"):
                parts.append(f"{course['creditos']} créditos")
            lines.append("- " + ", ".join(parts) + ".")
            if asks_requirements:
                lines.append("  Requisitos: " + self._format_curriculum_requirements(course, courses))

        if asks_requirements:
            lines.append(f"Valida tu caso personal y cursos permitidos en Campus Virtual PUCP: {self.CAMPUS_URL}")
        return "\n".join(lines) if lines else None

    def _load_curriculum(self) -> dict[str, Any]:
        if not self.CURRICULUM_PATH.exists():
            return {}
        try:
            return json.loads(self.CURRICULUM_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _extract_curriculum_cycle(self, normalized_question: str) -> str | None:
        word_cycles = {
            "quinto": "5", "quinta": "5", "5to": "5", "5": "5",
            "sexto": "6", "sexta": "6", "6to": "6", "6": "6",
            "septimo": "7", "setimo": "7", "septima": "7", "7mo": "7", "7": "7",
            "octavo": "8", "octava": "8", "8vo": "8", "8": "8",
            "noveno": "9", "novena": "9", "9no": "9", "9": "9",
            "decimo": "10", "decima": "10", "10mo": "10", "10": "10",
        }
        match = re.search(r"\b(?:ciclo\s*)?(5|6|7|8|9|10)(?:to|mo|no|vo)?\b", normalized_question)
        if match:
            return match.group(1)
        for term, cycle in word_cycles.items():
            if re.search(rf"\b{term}\b", normalized_question):
                return cycle
        return None

    def _answer_curriculum_cycle(self, cycle: str, curriculum: dict[str, Any]) -> str | None:
        codes = curriculum.get("ciclos", {}).get(str(cycle), [])
        courses = curriculum.get("cursos", {})
        if not codes:
            return None
        lines = [f"Cursos del ciclo {cycle} en la malla de Ingeniería Informática PUCP:"]
        for code in codes:
            course = courses.get(code, {})
            credit_text = f", {course.get('creditos')} créditos" if course.get("creditos") else ""
            lines.append(f"- {course.get('nombre', code)} ({code}){credit_text}.")
        return "\n".join(lines)

    def _merge_curriculum_name_references(self, normalized_question: str, refs: list[dict[str, Any]], curriculum: dict[str, Any]) -> list[dict[str, Any]]:
        merged = list(refs)
        seen = {ref.get("codigo") for ref in merged}
        for course in curriculum.get("cursos", {}).values():
            code = course.get("codigo")
            if code in seen:
                continue
            if self._contains_course_term(normalized_question, course.get("nombre", "")) or self._contains_course_term(normalized_question, code or ""):
                merged.append(course)
                seen.add(code)
        return merged

    def _format_curriculum_requirements(self, course: dict[str, Any], courses: dict[str, Any]) -> str:
        req_text = course.get("requisitos_texto") or "sin requisitos registrados en la malla"
        for code in sorted(set(course.get("requisitos_codigos", [])), key=len, reverse=True):
            req_course = courses.get(code)
            if req_course:
                req_text = re.sub(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", f"{req_course['nombre']} ({code})", req_text)
        return req_text + "."

    def _preflight_course_load_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        mentions_retake_slang = re.search(r"\b(bika|bica|trika|trica)\b", normalized_question) is not None
        mentions_third_time = re.search(r"\b(trika|trica)\b", normalized_question) is not None or "tercera vez" in normalized_question
        asks_credit_limit = any(phrase in normalized_question for phrase in [
            "cuantos creditos", "cuanto es el maximo", "maximo numero de creditos",
            "maximo de creditos", "maxima cantidad de creditos",
        ])
        asks_course_count = any(phrase in normalized_question for phrase in [
            "cuantos cursos", "cantidad de cursos", "numero de cursos",
        ])
        asks_can_take_course = any(phrase in normalized_question for phrase in [
            "puedo llevar", "puedo matricularme",
        ])
        if not (asks_credit_limit or asks_course_count or asks_can_take_course):
            return None

        if mentions_third_time:
            course = self._extract_course_after_slang(normalized_question)
            requested_course = self._extract_requested_course_in_slang_question(normalized_question)
            course_display = self._format_course_alias(course) if course else None
            requested_course_display = self._format_course_alias(requested_course) if requested_course else None

            if asks_credit_limit and not asks_course_count:
                return "Si te matriculas en un curso por tercera vez, puedes llevar como máximo 15 créditos en el semestre."
            if asks_course_count:
                return (
                    "Si te matriculas en un curso por tercera vez, no puedes matricularte en más de un curso por tercera vez. "
                    "Además, solo puedes llevar un curso por segunda vez y el máximo total es 15 créditos en el semestre. "
                    "La cantidad exacta de cursos depende de los créditos de cada curso."
                )

            if asks_can_take_course and course_display and requested_course_display:
                return (
                    f"Si estás llevando {course_display} por tercera vez y quieres llevar {requested_course_display}, "
                    "considera que en ese semestre puedes llevar como máximo 15 créditos, no puedes matricularte en más "
                    "de un curso por tercera vez y solo puedes llevar un curso por segunda vez. "
                    "No puedo confirmar si ese curso te aparecerá permitido solo con la pregunta; verifícalo en "
                    f"Campus Virtual PUCP: {self.CAMPUS_URL}"
                )

            if asks_can_take_course and requested_course_display:
                return (
                    f"Si llevas un curso por tercera vez y quieres llevar {requested_course_display}, considera estas restricciones: "
                    "máximo total de 15 créditos en el semestre, no más de un curso por tercera vez y solo un curso por segunda vez. "
                    f"Verifica si el curso específico te aparece permitido en Campus Virtual PUCP: {self.CAMPUS_URL}"
                )

            return (
                "Si llevas un curso por tercera vez, debes considerar estas restricciones: máximo total de 15 créditos en el semestre, "
                "no más de un curso por tercera vez y solo un curso por segunda vez. Verifica si el curso específico te aparece permitido en "
                f"Campus Virtual PUCP: {self.CAMPUS_URL}"
            )

        if mentions_retake_slang:
            return (
                "No tengo cargada una regla exacta de carga académica para bika. "
                "Verifica tu caso concreto, cursos permitidos y posibles restricciones en "
                f"Campus Virtual PUCP: {self.CAMPUS_URL}"
            )

        if asks_credit_limit:
            return "Para Ingeniería Informática PUCP, el máximo cargado es 28 créditos en un ciclo."

        return None

    def _preflight_student_slang_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        mentions_bika = re.search(r"\b(bika|bica)\b", normalized_question) is not None
        mentions_trika = re.search(r"\b(trika|trica)\b", normalized_question) is not None
        if not mentions_bika and not mentions_trika:
            return None

        if mentions_bika and mentions_trika:
            return (
                "Bika y trika son vocabulario estudiantil. "
                "Bika significa llevar un mismo curso por segunda vez; "
                "trika significa llevarlo por tercera vez. "
                "No se refieren a evaluaciones tipo TA ni a nombres de cursos."
            )

        term = "bika" if mentions_bika else "trika"
        attempt = "segunda" if mentions_bika else "tercera"
        course = self._extract_course_after_slang(normalized_question)
        requested_course = self._extract_requested_course_in_slang_question(normalized_question)
        course_display = self._format_course_alias(course) if course else None
        requested_course_display = self._format_course_alias(requested_course) if requested_course else None
        if course_display and requested_course_display:
            return (
                f"Entiendo que estás llevando por {attempt} vez '{course_display}'. "
                f"No puedo confirmar si puedes llevar '{requested_course_display}' en el mismo ciclo solo con esa información; "
                "eso depende de requisitos, cursos permitidos, posibles infracciones de plan y reglas de matrícula. "
                f"Revísalo en Campus Virtual PUCP: {self.CAMPUS_URL}. "
                "Si el sistema no te permite matricularte o te marca infracción de plan, consulta con la Dirección de Carrera o con matrícula-ocr@pucp.edu.pe."
            )

        if course_display:
            return (
                f"Si dices que llevas {term} de '{course_display}', entiendo que estás llevando por {attempt} vez "
                f"ese curso. {term.capitalize()} no es parte del nombre del curso."
            )

        return (
            f"{term.capitalize()} es vocabulario estudiantil: significa llevar un mismo curso por {attempt} vez. "
            "No se refiere a evaluaciones tipo TA ni a un nombre de curso."
        )

    def _extract_course_after_slang(self, normalized_question: str) -> str | None:
        match = re.search(r"\b(?:bika|bica|trika|trica)\s+de\s+(.+)$", normalized_question)
        if not match:
            return None
        course = re.sub(r"<@!?\d+>", "", match.group(1))
        course = re.split(
            r"[,.;?]|\b(?:puedo|podria|debo|quisiera|quiero|es posible|tambien|igualmente)\b",
            course,
            maxsplit=1,
        )[0]
        course = re.sub(r"[^a-z0-9\s]", " ", course)
        course = re.sub(r"\s+", " ", course).strip()
        stop_phrases = ["que significa", "que entiendes", "sabes", "cuando te digo"]
        if not course or any(phrase in course for phrase in stop_phrases):
            return None
        return course

    def _extract_requested_course_in_slang_question(self, normalized_question: str) -> str | None:
        match = re.search(
            r"\b(?:puedo|podria|quisiera|quiero|debo)\s+(?:llevar|matricularme\s+en|meterme\s+a|inscribirme\s+en)\s+(.+)$",
            normalized_question,
        )
        if not match:
            return None
        requested_course = re.split(
            r"[,.;?]|\b(?:igualmente|tambien|este ciclo|en el ciclo|a la vez|en paralelo|junto con)\b",
            match.group(1),
            maxsplit=1,
        )[0]
        requested_course = re.sub(r"[^a-z0-9\s]", " ", requested_course)
        requested_course = re.sub(r"\s+", " ", requested_course).strip()
        return requested_course or None

    def _load_course_alias_records_by_alias(self) -> dict[str, dict[str, Any]]:
        if not self.COURSE_ALIASES_PATH.exists():
            return {}
        try:
            records = json.loads(self.COURSE_ALIASES_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        aliases = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            alias = self._normalize_for_course_matching(str(record.get("alias", "")))
            name = str(record.get("nombre", "")).strip()
            if alias and name:
                aliases[alias] = record
        return aliases

    def _find_course_references(self, question: str) -> list[dict[str, Any]]:
        normalized_question = self._normalize_for_course_matching(question)
        aliases = self._load_course_alias_records_by_alias()
        matches = []
        seen_codes = set()
        for alias, record in aliases.items():
            name = self._normalize_for_course_matching(str(record.get("nombre", "")))
            code = self._normalize_for_course_matching(str(record.get("codigo", "")))
            terms = {alias, name, code}
            if not any(self._contains_course_term(normalized_question, term) for term in terms):
                continue
            code = record.get("codigo", "")
            key = code or name or alias
            if key in seen_codes:
                continue
            seen_codes.add(key)
            matches.append(record)
        return matches

    def _find_passed_course_references(self, question: str) -> list[dict[str, Any]]:
        normalized_question = self._normalize_for_course_matching(question)
        passed_refs = []
        for ref in self._find_course_references(question):
            terms = [ref.get("alias", ""), ref.get("nombre", "")]
            for term in terms:
                normalized_term = self._normalize_for_course_matching(str(term))
                if not normalized_term:
                    continue
                pattern = rf"\b(?:pase|pasee|pasado|aprobe|aprobado|acabo de pasar|ya pase|ya aprobe)\b(?:\s+\w+){{0,4}}\s+{re.escape(normalized_term)}\b"
                if re.search(pattern, normalized_question):
                    passed_refs.append(ref)
                    break
        return passed_refs

    def _normalize_for_course_matching(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", normalized.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _contains_course_term(self, normalized_text: str, term: str) -> bool:
        normalized_term = self._normalize_for_course_matching(term)
        if not normalized_term:
            return False
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", normalized_text) is not None

    def _load_course_aliases(self) -> dict[str, str]:
        if not self.COURSE_ALIASES_PATH.exists():
            return {}
        try:
            records = json.loads(self.COURSE_ALIASES_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        aliases = {}
        for record in records:
            alias = self._normalize_text(str(record.get("alias", "")))
            name = str(record.get("nombre", "")).strip()
            if alias and name:
                aliases[alias] = name
        return aliases

    def _format_course_alias(self, course: str | None) -> str | None:
        if not course:
            return None
        aliases = self._load_course_alias_records_by_alias()
        normalized_course = self._normalize_for_course_matching(course)
        record = aliases.get(normalized_course)
        if record:
            name = str(record.get("nombre", "")).strip()
            code = str(record.get("codigo", "")).strip()
            return f"{name} ({code})" if code else name
        return course

    def _preflight_ambiguous_slang_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        normalized_course_question = self._normalize_for_course_matching(pregunta)
        ambiguous_terms = ["pepita", "digi"]
        if any(re.search(rf"\b{re.escape(term)}\b", normalized_question) for term in ambiguous_terms):
            return self._ask_to_clarify_ambiguous_terms(pregunta)
        mentions_design_alone = re.search(r"\bdiseno\b", normalized_course_question) is not None
        mentions_design_full_name = any(term in normalized_course_question for term in [
            "diseno de software", "proyecto de diseno y desarrollo de software",
        ])
        if mentions_design_alone and not mentions_design_full_name:
            return (
                "Cuando dices 'diseño', necesito que precises el nombre oficial del curso: "
                "Diseño de Software o Proyecto de Diseño y Desarrollo de Software. Así evito asignar un código incorrecto."
            )
        return None

    def _ask_to_clarify_ambiguous_terms(self, pregunta: str) -> str:
        normalized_question = self._normalize_text(pregunta)
        found_terms = [term for term in ["pepita", "digi"] if re.search(rf"\b{re.escape(term)}\b", normalized_question)]
        if found_terms:
            terms = ", ".join(found_terms)
            return (
                f"No tengo claro a qué te refieres con '{terms}' en la información cargada. "
                "¿Puedes escribir el nombre oficial del curso, trámite o sistema? Así evito darte una respuesta inventada."
            )
        return (
            "No tengo suficiente contexto para interpretar esa abreviatura o apodo. "
            "¿Puedes escribir el nombre oficial del curso, trámite o sistema?"
        )

    def _contains_wrong_psp_derivation(self, pregunta: str, answer: str) -> bool:
        normalized_question = self._normalize_text(pregunta)
        normalized_answer = self._normalize_text(answer)
        mentions_psp_derivation = any(term in normalized_answer for term in [
            "coordinador de psp", "oficina de psp", "profesor aguilera",
        ])
        if not mentions_psp_derivation:
            return False
        question_is_psp = any(term in normalized_question for term in [
            "psp", "practica", "practicas", "convenio", "modalidades formativas",
        ])
        return not question_is_psp

    def _contains_out_of_scope_answer(self, answer: str) -> bool:
        normalized_answer = self._normalize_for_course_matching(answer)
        out_of_scope_patterns = [
            r"\b2\s*\+\s*2\s*(?:es|=|igual a)\s*4\b",
            r"si necesitas ayuda con calculos",
            r"usa una calculadora",
            r"puedo ayudarte con matematicas",
        ]
        return any(re.search(pattern, normalized_answer) for pattern in out_of_scope_patterns)

    def _safe_generic_unknown_response(self) -> str:
        return (
            "No tengo esa información en los documentos cargados. "
            "¿Puedes precisar el curso, trámite o proceso para orientarte mejor?"
        )

    def _contains_unsupported_ambiguous_term_claim(self, pregunta: str, answer: str) -> bool:
        normalized_question = self._normalize_text(pregunta)
        normalized_answer = self._normalize_text(answer)
        if not any(re.search(rf"\b{term}\b", normalized_question) for term in ["pepita", "digi"]):
            return False
        risky_answer_terms = [
            "plan de estudios", "bajar un curso", "portal del estudiante", "pepipa",
            "debes hacerlo", "puedes hacerlo", "inicia sesion",
        ]
        return any(term in normalized_answer for term in risky_answer_terms)

    def _contains_invented_course_name(self, answer: str) -> bool:
        normalized_answer = self._normalize_text(answer)
        invented_names = [
            "diseno de producto", "diseño de producto", "diseno de producto 1", "diseño de producto 1",
        ]
        return any(name in normalized_answer for name in invented_names)

    def _safe_invented_course_name_response(self) -> str:
        return (
            "No puedo confirmar ese nombre de curso con los sílabos cargados. "
            "Para evitar darte un código incorrecto, escribe el nombre oficial del curso o revísalo en Campus Virtual PUCP: "
            f"{self.CAMPUS_URL}"
        )

    def _preflight_psp_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        mentions_psp = "psp" in normalized_question or "practica supervisada preprofesional" in normalized_question
        if not mentions_psp:
            return None

        mentions_convenio = "convenio" in normalized_question or "modalidades formativas" in normalized_question
        if mentions_convenio:
            return (
                "Para consultas sobre convenios de PSP o modalidades formativas, corresponde revisarlo con la Oficina de PSP de la FCI "
                "o con el coordinador de PSP de tu especialidad."
            )

        mentions_hours = "270" in normalized_question or "hora" in normalized_question or "horas" in normalized_question
        mentions_enrollment = any(term in normalized_question for term in [
            "matricular", "matricula", "inscribir", "inscribirme", "curso de psp",
        ])
        asks_required_hours = any(term in normalized_question for term in [
            "cuantas", "cuantos", "necesito", "requiere", "requisito", "convalidar", "validar",
        ])

        if mentions_hours and asks_required_hours and not mentions_enrollment:
            return "Para convalidar o validar PSP se requieren 270 horas de práctica."

        if not (mentions_hours and mentions_enrollment):
            return None

        return (
            "No necesariamente debes tener las 270 horas completas antes de matricularte en PSP. "
            "Las 270 horas son necesarias para convalidar la práctica y pueden completarse hasta antes de la entrega del informe. "
            "Lo recomendable es matricularte cuando estés por terminar la práctica. Si tu caso es particular, consúltalo con la Oficina de PSP o el coordinador de PSP de tu especialidad."
        )

    def _preflight_plan_issue_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        mentions_plan_issue = any(term in normalized_question for term in [
            "infraccion de plan", "infractor al plan", "infraccion del plan",
            "no me permite matricular", "no me deja matricular", "no puedo matricular",
            "cursos permitidos", "lista de cursos permitidos",
        ])
        mentions_prereq_or_grade = any(term in normalized_question for term in [
            "requisito", "requisitos", "prerrequisito", "previo", "nota", "aprobe", "aprobado",
        ])
        if mentions_plan_issue or (mentions_prereq_or_grade and "matricul" in normalized_question):
            return self._answer_plan_issue()
        return None

    def _answer_plan_issue(self) -> str:
        return (
            "Si el sistema te marca infracción de plan o no te permite matricularte pese a cumplir requisitos, "
            "primero verifica si corresponde registrar una solicitud de excepción en Campus Virtual PUCP: "
            f"{self.CAMPUS_URL}. "
            "Si el problema continúa, consulta con la Dirección de Carrera o con matrícula-ocr@pucp.edu.pe. "
            "No puedo confirmar tu caso sin revisar tu situación académica personal."
        )

    def _contains_unsupported_plan_issue_derivation(self, pregunta: str, answer: str) -> bool:
        normalized_question = self._normalize_text(pregunta)
        normalized_answer = self._normalize_text(answer)
        is_plan_issue = any(term in normalized_question for term in [
            "infraccion de plan", "infractor al plan", "requisito", "prerrequisito",
            "no me permite matricular", "no me deja matricular", "cursos permitidos",
        ])
        if not is_plan_issue:
            return False
        unsupported_terms = ["coordinador", "coordinador de carrera", "portal del estudiante"]
        return any(term in normalized_answer for term in unsupported_terms)

    def _preflight_vacancy_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        mentions_vacancy = any(term in normalized_question for term in [
            "vacante", "vacantes", "cupo", "cupos", "puesto", "lista de espera",
        ])
        asks_expansion = any(term in normalized_question for term in [
            "ampliar", "ampliaran", "amplian", "abrir", "abriran", "aumentar", "aumenten",
            "quedarme fuera", "no quiero quedarme fuera", "alcanzar", "entro",
        ])
        if not (mentions_vacancy and asks_expansion):
            return None

        return self._safe_vacancy_derivation()

    def _safe_vacancy_derivation(self) -> str:
        return (
            "No puedo confirmar si ampliarán vacantes ni comprometer una revisión en una fecha específica. "
            "Eso lo comunica la Dirección de Carrera. "
            "Mantente atento a los anuncios oficiales del curso y, si necesitas evaluar tu caso, consulta directamente con la Dirección de Carrera."
        )

    def _contains_unsupported_vacancy_promise(self, answer: str) -> bool:
        normalized_answer = self._normalize_text(answer)
        mentions_vacancy = any(term in normalized_answer for term in [
            "vacante", "vacantes", "cupo", "cupos", "horario", "horarios",
        ])
        if not mentions_vacancy:
            return False
        risky_phrases = [
            "lo revisaremos", "lo evaluaremos", "vamos a revisar", "vamos a evaluar",
            "se revisara", "se evaluara", "el dia lunes", "el dia martes", "el dia miercoles",
            "el dia jueves", "el dia viernes", "ampliaremos", "abriremos otro horario",
        ]
        return any(phrase in normalized_answer for phrase in risky_phrases)

    def _preflight_calendar_event_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        if "matricula" in normalized_question:
            return None

        event_key = self._detect_calendar_event(normalized_question)
        if event_key is None:
            return None

        facts = self._load_matricula_facts()
        cycle = self._extract_cycle(normalized_question)
        if cycle is None:
            return self._ask_for_cycle_with_known_options(facts)

        cycle_data = facts.get("cycles", {}).get(cycle)
        if not cycle_data or cycle_data.get("status") != "official":
            return (
                f"No tengo una fecha oficial confirmada para ese evento del ciclo {cycle} en el contexto cargado. "
                f"Revisa el Portal del estudiante PUCP: {self.PORTAL_URL}"
            )

        event = self._event_by_key(cycle_data, event_key)
        if not event:
            return (
                f"No tengo ese evento en el calendario cargado para el ciclo {cycle}. "
                f"Fuente disponible: {cycle_data.get('source_url', self.PORTAL_URL)}"
            )

        return self._answer_calendar_event_status(cycle, cycle_data, event)

    def _detect_calendar_event(self, normalized_question: str) -> str | None:
        if "clase" in normalized_question and any(term in normalized_question for term in ["empieza", "inicia", "inicio", "comienza"]):
            return "classes_start"
        if any(term in normalized_question for term in ["fin del semestre", "termina el semestre", "culmina el semestre", "acaba el semestre"]):
            return "semester_end"
        if "turno" in normalized_question and "matricula" in normalized_question:
            return "publication_turns"
        if "horario" in normalized_question and any(term in normalized_question for term in ["publica", "salen", "sale", "cuando"]):
            return "publication_schedules"
        return None

    def _answer_calendar_event_status(self, cycle: str, cycle_data: dict[str, Any], event: dict[str, Any]) -> str:
        event_date_text = self._format_event_date(event)
        event_date = date.fromisoformat(event.get("start_date") or event.get("end_date"))
        today = date.today()
        connector = "" if event_date_text.startswith("del ") else "el "
        event_key = event.get("key")

        if event_key == "classes_start":
            verb = "empezaron" if today > event_date else "empiezan"
            return f"Las clases del ciclo {cycle} {verb} {connector}{event_date_text}."

        if event_key == "semester_end":
            verb = "terminó" if today > event_date else "termina"
            return f"El semestre {cycle} {verb} {connector}{event_date_text}."

        if today > event_date:
            verb = "fue"
        elif today == event_date:
            verb = "es"
        else:
            verb = "será"
        return f"Para el ciclo {cycle}, {event['label']} {verb} {connector}{event_date_text}."

    def _preflight_matricula_response(self, pregunta: str) -> str | None:
        normalized_question = self._normalize_text(pregunta)
        if "matricula" not in normalized_question:
            return None

        if "extraordinaria" in normalized_question or "extraordinarias" in normalized_question:
            return self._answer_extraordinary_enrollment_question(normalized_question)

        if self._asks_personal_enrollment_turn(normalized_question):
            return self._answer_personal_enrollment_turn()

        asks_date = any(term in normalized_question for term in [
            "cuando", "fecha", "fechas", "termina", "culmina", "finaliza", "concluye", "vence", "cierra",
            "hasta cuando", "calendario", "paso", "ya paso", "sigue abierta",
        ])
        if not asks_date:
            return None

        facts = self._load_matricula_facts()
        cycle = self._extract_cycle(normalized_question)
        asks_end = any(term in normalized_question for term in ["termina", "culmina", "finaliza", "concluye", "vence", "cierra", "hasta cuando"])
        asks_if_passed = any(term in normalized_question for term in ["paso", "ya paso", "sigue abierta"])

        if cycle is None:
            return self._ask_for_cycle_with_known_options(facts)

        cycle_data = facts.get("cycles", {}).get(cycle)
        if not cycle_data:
            return (
                f"No tengo información oficial cargada para el ciclo {cycle}. "
                f"Revisa el Portal del estudiante PUCP: {self.PORTAL_URL} "
                "o consulta a matricula-ocr@pucp.edu.pe."
            )

        if cycle_data.get("status") != "official":
            return (
                f"No tengo una fecha oficial confirmada de matrícula para el ciclo {cycle}. "
                f"Revisa el Portal del estudiante PUCP: {cycle_data.get('source_url', self.PORTAL_URL)} "
                "o consulta a matricula-ocr@pucp.edu.pe."
            )

        specific_event_key = self._detect_matricula_event(normalized_question)
        if specific_event_key:
            return self._answer_matricula_event(cycle, cycle_data, specific_event_key)

        if asks_if_passed:
            return self._answer_if_enrollment_passed(cycle, cycle_data)
        if asks_end:
            return self._answer_enrollment_end(cycle, cycle_data)
        return self._answer_cycle_calendar(cycle, cycle_data)

    def _asks_personal_enrollment_turn(self, normalized_question: str) -> bool:
        mentions_turn = "turno" in normalized_question and "matricula" in normalized_question
        personal_terms = ["mi", "mis", "exacto", "asignado", "asignados", "me toca", "tengo"]
        return mentions_turn and any(term in normalized_question for term in personal_terms)

    def _answer_personal_enrollment_turn(self) -> str:
        return (
            "Tu turno exacto de matrícula es información personal. Revísalo en Campus Virtual PUCP: "
            f"{self.CAMPUS_URL}"
        )

    def _detect_matricula_event(self, normalized_question: str) -> str | None:
        if "extemporanea" in normalized_question or "extemporaneas" in normalized_question:
            return "extemporaneous_enrollment"
        if "regular" in normalized_question or "campus virtual" in normalized_question:
            return "regular_enrollment"
        if "turno" in normalized_question:
            return "publication_turns"
        if "horario" in normalized_question:
            return "publication_schedules"
        return None

    def _answer_matricula_event(self, cycle: str, cycle_data: dict[str, Any], event_key: str) -> str:
        event = self._event_by_key(cycle_data, event_key)
        if not event:
            return (
                f"No tengo esa fecha de matrícula para el ciclo {cycle} en el contexto cargado. "
                f"Revisa el Portal del estudiante PUCP: {cycle_data.get('source_url', self.PORTAL_URL)}"
            )

        event_date = date.fromisoformat(event.get("end_date") or event.get("start_date"))
        today = date.today()
        if today > event_date:
            verb = "fue"
        elif today == event_date:
            verb = "es"
        else:
            verb = "será"
        return f"{event['label']} del ciclo {cycle} {verb} {self._format_event_date(event)}."

    def _answer_extraordinary_enrollment_question(self, normalized_question: str) -> str:
        facts = self._load_matricula_facts()
        cycle = self._extract_cycle(normalized_question)
        cycles = facts.get("cycles", {})
        cycle_data = cycles.get(cycle) if cycle else None

        if not cycle_data:
            official_cycles = sorted(cycle_id for cycle_id, data in cycles.items() if data.get("status") == "official")
            if official_cycles:
                cycle = official_cycles[-1]
                cycle_data = cycles.get(cycle)

        message = (
            "No puedo confirmar que 'matrícula extraordinaria' sea un proceso oficial "
            "ni tratarla como sinónimo de matrícula extemporánea con la información cargada."
        )

        if cycle_data and cycle_data.get("status") == "official":
            extemporaneous = self._event_by_key(cycle_data, "extemporaneous_enrollment")
            if extemporaneous:
                message += (
                    f" Para el ciclo {cycle}, el calendario cargado sí menciona "
                    f"{extemporaneous['label']}: {self._format_event_date(extemporaneous)}."
                )

        return (
            message
            + f" Revisa el calendario académico 2026-1: {self.CALENDAR_2026_1_URL} "
            + "o consulta directamente a matricula-ocr@pucp.edu.pe si tu caso es excepcional."
        )

    def _load_matricula_facts(self) -> dict[str, Any]:
        if not self.MATRICULA_FACTS_PATH.exists():
            return {"cycles": {}}
        with open(self.MATRICULA_FACTS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    def _ask_for_cycle_with_known_options(self, facts: dict[str, Any]) -> str:
        cycles = facts.get("cycles", {})
        official_cycles = sorted(cycle for cycle, data in cycles.items() if data.get("status") == "official")
        unavailable_cycles = sorted(cycle for cycle, data in cycles.items() if data.get("status") != "official")
        lines = ["¿A qué ciclo te refieres?"]
        if official_cycles:
            lines.append("Tengo fechas oficiales cargadas para: " + ", ".join(official_cycles) + ".")
        if unavailable_cycles:
            lines.append("Para " + ", ".join(unavailable_cycles) + " no tengo fechas oficiales confirmadas en el contexto cargado.")
        lines.append("Indícame el ciclo, por ejemplo 2026-1 o 2026-2.")
        return "\n".join(lines)

    def _event_by_key(self, cycle_data: dict[str, Any], key: str) -> dict[str, Any] | None:
        for event in cycle_data.get("events", []):
            if event.get("key") == key:
                return event
        return None

    def _format_event_date(self, event: dict[str, Any], include_start: bool = True) -> str:
        if not event:
            return "fecha no disponible"
        start = event.get("start_date")
        end = event.get("end_date") or start
        start_time = event.get("start_time")
        end_time = event.get("end_time")
        if start and end and start != end and include_start:
            text = f"del {self._format_date(start)} al {self._format_date(end)}"
        else:
            text = self._format_date(end or start)
        if start_time and include_start:
            text += f" desde las {start_time}"
        if end_time:
            text += f", hasta las {end_time}"
        return text

    def _format_date(self, iso_date: str) -> str:
        parsed = date.fromisoformat(iso_date)
        months = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ]
        return f"{parsed.day} de {months[parsed.month - 1]} de {parsed.year}"

    def _answer_if_enrollment_passed(self, cycle: str, cycle_data: dict[str, Any]) -> str:
        regular = self._event_by_key(cycle_data, "regular_enrollment")
        extemporaneous = self._event_by_key(cycle_data, "extemporaneous_enrollment")
        source = cycle_data.get("source_url", self.PORTAL_URL)
        if not regular or not regular.get("end_date"):
            return (
                f"No tengo fecha de cierre de matrícula regular para {cycle} en el contexto cargado. "
                f"Fuente: {source}"
            )

        today = date.today()
        regular_end = date.fromisoformat(regular["end_date"])
        if today > regular_end:
            response = (
                f"Sí, la matrícula regular del ciclo {cycle} ya pasó.\n"
                f"{regular['label']} fue {self._format_event_date(regular)}."
            )
            if extemporaneous:
                response += f"\n{extemporaneous['label']} fue {self._format_event_date(extemporaneous)}."
            return response

        return (
            f"No, la matrícula regular del ciclo {cycle} todavía no ha terminado.\n"
            f"{regular['label']} termina {self._format_event_date(regular, include_start=False)}."
        )

    def _answer_enrollment_end(self, cycle: str, cycle_data: dict[str, Any]) -> str:
        regular = self._event_by_key(cycle_data, "regular_enrollment")
        extemporaneous = self._event_by_key(cycle_data, "extemporaneous_enrollment")
        lines = [f"Para el ciclo {cycle}:"]
        if regular:
            lines.append(f"- {regular['label']}: {self._format_event_date(regular)}.")
        if extemporaneous:
            lines.append(f"- {extemporaneous['label']}: {self._format_event_date(extemporaneous)}.")
        return "\n".join(lines)

    def _answer_cycle_calendar(self, cycle: str, cycle_data: dict[str, Any]) -> str:
        lines = [f"Para el ciclo {cycle}:"]
        for event in cycle_data.get("events", []):
            lines.append(f"- {event['label']}: {self._format_event_date(event)}.")
        return "\n".join(lines)

    def _extract_cycle(self, normalized_question: str) -> str | None:
        match = re.search(r"\b(20\d{2})\s*[-.]\s*([12])\b", normalized_question)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        if "2026 1" in normalized_question or "20261" in normalized_question:
            return "2026-1"
        if "2026 2" in normalized_question or "20262" in normalized_question:
            return "2026-2"
        return None

    def _context_text(self, context_docs: list[Any]) -> str:
        parts = []
        for doc in context_docs or []:
            page_content = getattr(doc, "page_content", "")
            metadata = getattr(doc, "metadata", {}) or {}
            parts.append(page_content)
            parts.extend(str(value) for value in metadata.values())
        return self._normalize_text("\n".join(parts))

    def _sanitize_stale_cycle_links(self, pregunta: str, answer: str) -> str:
        current_year = date.today().year
        normalized_question = self._normalize_text(pregunta)
        requested_cycle = self._extract_cycle(normalized_question)
        requested_years = {int(year) for year in re.findall(r"\b20\d{2}\b", normalized_question)}
        removed_stale_link = False

        def is_stale_url(url: str) -> bool:
            normalized_url = self._normalize_text(url)
            url_cycles = {
                f"{match.group(1)}-{match.group(2)}"
                for match in re.finditer(r"\b(20\d{2})[-_/ ]?([12])\b", normalized_url)
            }
            if requested_cycle and url_cycles:
                return any(cycle != requested_cycle for cycle in url_cycles)

            url_years = {int(year) for year in re.findall(r"\b20\d{2}\b", normalized_url)}
            for year in url_years:
                if year in requested_years:
                    continue
                if year < current_year:
                    return True
            return False

        def has_stale_year_text(text_value: str) -> bool:
            text_cycles = {
                f"{match.group(1)}-{match.group(2)}"
                for match in re.finditer(r"\b(20\d{2})[-_/ ]?([12])\b", self._normalize_text(text_value))
            }
            if requested_cycle and text_cycles:
                return any(cycle != requested_cycle for cycle in text_cycles)
            for year in {int(year) for year in re.findall(r"\b20\d{2}\b", text_value)}:
                if year not in requested_years and year < current_year:
                    return True
            return False

        def replace_markdown_link(match: re.Match) -> str:
            nonlocal removed_stale_link
            label = match.group(1).strip()
            url = match.group(2).strip()
            if is_stale_url(url):
                removed_stale_link = True
                return "" if has_stale_year_text(label) else label
            return match.group(0)

        sanitized = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", replace_markdown_link, answer)

        def replace_plain_url(match: re.Match) -> str:
            nonlocal removed_stale_link
            full_url = match.group(0)
            url = full_url.rstrip(".,;)])")
            suffix = full_url[len(url):]
            if is_stale_url(url):
                removed_stale_link = True
                return suffix.lstrip(".,; ")
            return full_url

        sanitized = re.sub(r"https?://[^\s<>)\]]+", replace_plain_url, sanitized)
        sanitized = re.sub(r"[ \t]+", " ", sanitized)
        sanitized = re.sub(r" *\n *", "\n", sanitized).strip()
        sanitized = re.sub(r"\s+([,.;:])", r"\1", sanitized)
        sanitized = re.sub(r"(?:\s+y)?\s*(?:link|enlace)?\s*(?:viejo|antiguo|anterior)\s*$", "", sanitized, flags=re.IGNORECASE).strip()
        if removed_stale_link and self._normalize_text(sanitized) in {"revisa", "revisa.", "consulta", "consulta.", "completa el formulario psp", "completa el formulario psp:"}:
            sanitized = "No tengo un enlace vigente cargado para ese formulario."

        if removed_stale_link and not re.search(r"https?://", sanitized):
            if "psp" in normalized_question or "practica" in normalized_question:
                campus_message = f"Para formatos o formularios vigentes, revisa Campus Virtual PUCP: {self.CAMPUS_URL}"
                sanitized = f"{sanitized}\n{campus_message}" if sanitized else campus_message
            elif not sanitized:
                sanitized = "No tengo un enlace vigente cargado para ese ciclo."
        return sanitized

    def _contains_unsupported_matricula_claim(self, answer: str, context_docs: list[Any]) -> bool:
        normalized_answer = self._normalize_text(answer)
        normalized_context = self._context_text(context_docs)

        sensitive_terms = [
            "modificacion de matricula",
            "matricula extraordinaria",
            "matricula extemporanea",
            "matriculas extraordinarias",
            "matriculas extemporaneas",
        ]

        for term in sensitive_terms:
            if term in normalized_answer and term not in normalized_context:
                return True

        enumerates_matricula_types = (
            re.search(r"\b(existen|hay|son|tenemos)\b.{0,50}\b(tipos|modalidades|clases)\b.{0,40}\bmatricula", normalized_answer)
            or re.search(r"\b[0-9]+\s+(tipos|modalidades|clases)\s+de\s+matricula", normalized_answer)
            or "tipos de matricula" in normalized_answer
            or "modalidades de matricula" in normalized_answer
        )
        if enumerates_matricula_types and not (
            "tipos de matricula" in normalized_context
            or "modalidades de matricula" in normalized_context
            or "clases de matricula" in normalized_context
        ):
            return True

        return False

    def _contains_unsupported_matricula_date_claim(self, answer: str, context_docs: list[Any]) -> bool:
        normalized_answer = self._normalize_text(answer)
        normalized_context = self._context_text(context_docs)

        mentions_matricula = "matricula" in normalized_answer
        if not mentions_matricula:
            return False

        unsupported_relative_claim = any(phrase in normalized_answer for phrase in [
            "ha concluido hoy",
            "concluyo hoy",
            "termina hoy",
            "vence hoy",
            "ya concluyo",
            "ya finalizo",
            "ya termino",
            "ha finalizado",
            "ha terminado",
        ])
        if unsupported_relative_claim and not any(phrase in normalized_context for phrase in [
            "ha concluido hoy",
            "concluyo hoy",
            "termina hoy",
            "vence hoy",
        ]):
            return True

        mentions_2026_2 = "2026-2" in normalized_answer or "2026 2" in normalized_answer
        gives_end_date = re.search(
            r"\b(termina|culmina|finaliza|concluye|vence|cierra)\b.{0,80}\b(\d{1,2}\s+de\s+[a-z]+|lunes|martes|miercoles|jueves|viernes|sabado|domingo|hoy|manana)",
            normalized_answer,
        )
        if mentions_2026_2 and gives_end_date and not (
            "matricula 2026-2" in normalized_context
            and re.search(r"\b(termina|culmina|finaliza|concluye|vence|cierra)\b", normalized_context)
        ):
            return True

        return False

    def _safe_matricula_derivation(self) -> str:
        return (
            "Con la información oficial que tengo a la mano, no puedo afirmar que existan esas "
            "modalidades o tipos de matrícula. Para evitar darte un dato incorrecto, consulta "
            "directamente con la Oficina Central de Registro en matricula-ocr@pucp.edu.pe."
        )

    def _safe_matricula_date_derivation(self) -> str:
        return (
            "Con la información oficial que tengo a la mano, no puedo confirmar una fecha exacta "
            "de cierre para esa matrícula. Para evitar darte una fecha incorrecta, indícame el ciclo "
            "que quieres consultar, por ejemplo 2026-1 o 2026-2. "
            f"Portal del estudiante PUCP: {self.PORTAL_URL}"
        )
