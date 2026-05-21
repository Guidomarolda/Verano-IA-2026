from pathlib import Path
import re

# Skills que vamos a detectar en el MVP
SKILLS = ["Python", "Machine Learning", "SQL", "Java", "Git"]
REQUIRED_SKILLS = ["Python", "Machine Learning", "SQL"]
JOB_TITLE = "Data Analyst Junior"
SKILL_POINTS = 2
EXPERIENCE_POINTS = 1
MIN_EXPERIENCE_YEARS = 2

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


def calculate_score(skills, experience):
    """Calcula un puntaje simple para cada candidato."""
    score = 0
    reasons = []

    for skill in REQUIRED_SKILLS:
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


def calculate_max_score():
    """Calcula el puntaje máximo posible según las reglas actuales."""
    skill_points = len(REQUIRED_SKILLS) * SKILL_POINTS
    experience_points = EXPERIENCE_POINTS

    return skill_points + experience_points


def analyze_cv(cv_file):
    """Analiza un CV y devuelve la información del candidato."""
    text = read_cv(cv_file)
    skills = extract_skills(text)
    experience = extract_experience(text)
    score, reasons = calculate_score(skills, experience)

    return {
        "file": cv_file.name,
        "skills": skills,
        "experience": experience,
        "score": score,
        "reasons": reasons
    }


def print_candidate(candidate, position):
    """Muestra en pantalla la información de un candidato."""
    max_score = calculate_max_score()

    print(f"\n{position}. {candidate['file']}")
    print(f"Score: {candidate['score']}/{max_score}")
    print(f"Skills: {candidate['skills']}")
    print(f"Años de experiencia: {candidate['experience']}")
    print("Motivos:")
    for reason in candidate["reasons"]:
        print(f"- {reason}")


def rank_candidates(cv_files):
    """Analiza CVs y devuelve los candidatos ordenados por puntaje."""
    candidates = []

    for cv_file in cv_files:
        candidate = analyze_cv(cv_file)
        candidates.append(candidate)

    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def main():
    cv_files = list(DATA_DIR.glob("*.txt"))
    candidates = rank_candidates(cv_files)

    print(f"Puesto evaluado: {JOB_TITLE}")
    print("Skills requeridas para el puesto:")

    for skill in REQUIRED_SKILLS:
        print(f"- {skill}")
    
    print("\n=== RANKING DE CANDIDATOS ===")
    for i, candidate in enumerate(candidates, start=1):
        print_candidate(candidate, i)

if __name__ == "__main__":
    main()
