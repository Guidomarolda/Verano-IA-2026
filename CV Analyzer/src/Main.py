from pathlib import Path
import re

# Skills que vamos a detectar en el MVP
SKILLS = ["Python", "Machine Learning", "SQL", "Java", "Git"]
REQUIRED_SKILLS = ["Python", "Machine Learning", "SQL"]
JOB_TITLE = "Data Analyst Junior"

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
        if skill.lower() in text_lower:
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
            score += 2
            reasons.append(f"Tiene la skill requerida: {skill} (+2)")
        else:
            reasons.append(f"No tiene la skill requerida: {skill} (+0)")

    if experience > 2:
        score += 1
        reasons.append("Tiene más de 2 años de experiencia (+1)")

    return score, reasons


def main():
    cv_files = list(DATA_DIR.glob("*.txt"))
    candidates = []
    

    for cv_file in cv_files:
        text = read_cv(cv_file)
        skills = extract_skills(text)
        experience = extract_experience(text)
        score, reasons = calculate_score(skills, experience)
        candidate = {
            "file": cv_file.name,
            "skills": skills,
            "experience": experience,
            "score": score,
            "reasons": reasons
        }

        candidates.append(candidate)

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    
    print(f"Puesto evaluado: {JOB_TITLE}")
    print("Skills requeridas para el puesto:")
    for skill in REQUIRED_SKILLS:
        print(f"- {skill}")
    
    print("\n=== RANKING DE CANDIDATOS ===")

    for i, candidate in enumerate(candidates, start=1):
        print(f"\n{i}. {candidate['file']}")
        print(f"Score: {candidate['score']}/8")
        print(f"Skills: {candidate['skills']}")            
        print(f"Años de experiencia: {candidate['experience']}")
        print("Motivos:")
        for reason in candidate["reasons"]:
              print(f"- {reason}")

if __name__ == "__main__":
    main()