from pathlib import Path
import re

# Skills que vamos a detectar en el MVP
SKILLS = ["Python", "Machine Learning", "SQL", "Java", "Git"]
REQUIRED_SKILLS = ["Python", "Machine Learning", "SQL"]
JOB_TITLE = "Data Analyst Junior"
SKILL_POINTS = 2
EXPERIENCE_POINTS = 1
MIN_EXPERIENCE_YEARS = 2
JOB_DESCRIPTION = """
Buscamos un Data Analyst Junior con conocimientos en Python, SQL y Machine Learning.
Se valoran buenas prácticas con Git y experiencia trabajando con datos.
"""


# Ruta a la carpeta data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def read_cv(file_path):
    """Lee el contenido de un CV en formato txt."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_skills(text):
    """Detecta skills dentro del texto del CV."""
    found_skills = []

    text_lower = text.lower()

    for skill in SKILLS:
        pattern = rf"\b{re.escape(skill.lower())}\b"

        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills


def extract_experience(text):
    """Detecta años de experiencia dentro del texto del CV."""
    text_lower = text.lower()

    pattern = r"(\d+)\s+años"
    match = re.search(pattern, text_lower)

    if match:
        return int(match.group(1))

    return 0


def calculate_score(skills, experience, required_skills):
    """Calcula un puntaje simple para cada candidato."""
    score = 0
    reasons = []

    for skill in required_skills:
        if skill in skills:
            score += SKILL_POINTS
            reasons.append(f"Tiene la skill requerida: {skill} (+{SKILL_POINTS})")
        else:
            reasons.append(f"No tiene la skill requerida: {skill} (+0)")

    if experience > MIN_EXPERIENCE_YEARS:
        score += EXPERIENCE_POINTS
        reasons.append(
            f"Tiene más de {MIN_EXPERIENCE_YEARS} años de experiencia (+{EXPERIENCE_POINTS})"
    )

    return score, reasons


def calculate_max_score(required_skills):
    """Calcula el puntaje máximo posible según las reglas actuales."""
    skill_points = len(required_skills) * SKILL_POINTS
    experience_points = EXPERIENCE_POINTS

    return skill_points + experience_points


def analyze_cv(cv_file, required_skills):
    """Analiza un CV y devuelve la información del candidato."""
    text = read_cv(cv_file)
    skills = extract_skills(text)
    experience = extract_experience(text)
    score, reasons = calculate_score(skills, experience, required_skills)

    return {
        "file": cv_file.name,
        "skills": skills,
        "experience": experience,
        "score": score,
        "reasons": reasons
    }


def print_candidate(candidate, position, required_skills):
    """Muestra en pantalla la información de un candidato."""
    max_score = calculate_max_score(required_skills)

    print(f"\n{position}. {candidate['file']}")
    print(f"Score: {candidate['score']}/{max_score}")
    print(f"Skills: {candidate['skills']}")
    print(f"Años de experiencia: {candidate['experience']}")
    print("Motivos:")
    for reason in candidate["reasons"]:
        print(f"- {reason}")


def rank_candidates(cv_files, required_skills):
    """Analiza CVs y devuelve los candidatos ordenados por puntaje."""
    candidates = []

    for cv_file in cv_files:
        candidate = analyze_cv(cv_file, required_skills)
        candidates.append(candidate)

    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def extract_required_skills(job_description):
    """Extrae skills requeridas desde la descripción del puesto."""
    return extract_skills(job_description)


def main():
    cv_files = list(DATA_DIR.glob("*.txt"))
    required_skills = extract_required_skills(JOB_DESCRIPTION)
    candidates = rank_candidates(cv_files,  required_skills)

    print(f"Puesto evaluado: {JOB_TITLE}")
    print("Skills requeridas para el puesto:")

    for skill in required_skills:
        print(f"- {skill}")
    
    print("\n=== RANKING DE CANDIDATOS ===")
    for i, candidate in enumerate(candidates, start=1):
        print_candidate(candidate, i, required_skills)

if __name__ == "__main__":
    main()
