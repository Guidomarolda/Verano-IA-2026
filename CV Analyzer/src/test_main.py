from Main import calculate_score, extract_experience, extract_skills


def test_calculate_score_with_all_requirements():
    skills = ["Python", "Machine Learning", "SQL"]
    experience = 3

    score, reasons = calculate_score(skills, experience)

    assert score == 7
    assert len(reasons) == 4


def test_calculate_score_without_experience_bonus():
    skills = ["Python", "SQL"]
    experience = 1

    score, reasons = calculate_score(skills, experience)

    assert score == 4
    assert len(reasons) == 3


def test_extract_experience_from_text():
    text = "Desarrollador backend con 3 años de experiencia."

    experience = extract_experience(text)

    assert experience == 3


def test_extract_experience_returns_zero_when_missing():
    text = "Perfil orientado a datos con experiencia en Python y SQL."

    experience = extract_experience(text)

    assert experience == 0


def test_extract_skills_from_text():
    text = "Tengo experiencia en Python, SQL y Git."

    skills = extract_skills(text)

    assert skills == ["Python", "SQL", "Git"]


def test_extract_skills_is_case_insensitive():
    text = "Tengo experiencia en python, sql y git."

    skills = extract_skills(text)

    assert skills == ["Python", "SQL", "Git"]

def test_extract_skills_does_not_match_partial_words():
    text = "Tengo experiencia usando NoSQL en proyectos de datos."

    skills = extract_skills(text)

    assert skills == []


if __name__ == "__main__":
    test_calculate_score_with_all_requirements()
    test_calculate_score_without_experience_bonus()
    test_extract_experience_from_text()
    test_extract_experience_returns_zero_when_missing()
    test_extract_skills_from_text()
    test_extract_skills_is_case_insensitive()
    test_extract_skills_does_not_match_partial_words()
    print("Tests OK")