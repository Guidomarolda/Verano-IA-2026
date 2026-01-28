# RAG mínimo local (sin embeddings reales)
# Idea: "vectorizar" textos -> buscar similares -> meterlos como contexto

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1) Fuente de info (tu "base de conocimiento")
docs = [
    "Jarvis Núcleo es el copiloto cognitivo: refleja principios y detecta incoherencias.",
    "Jarvis Operativo ejecuta tareas y automatiza procesos siguiendo criterios del Núcleo.",
    "Un embedding es un vector numérico que representa significado.",
    "RAG recupera fragmentos relevantes y los inyecta al prompt para responder mejor.",
    "Data leakage es usar información del test en entrenamiento (directa o indirectamente).",
]

# 2) Vectorización (simula embeddings)
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs)

def retrieve(query: str, k: int = 2):
    # 3) Recuperación (search)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_vectors)[0]  # similaridad con cada doc
    top_idx = sims.argsort()[::-1][:k]
    return [(docs[i], sims[i]) for i in top_idx]

# 4) Respuesta con contexto inyectado (sin LLM, solo demo)
def rag_answer_fake(query: str, k: int = 3):
    retrieved = retrieve(query, k=k)

    # armamos el contexto en texto (por si lo querés imprimir)
    context_lines = [text for text, _score in retrieved]

    # respuesta "fake": resume lo recuperado (sin LLM)
    answer = (
        "Jarvis necesita memoria externa y RAG para no depender del historial completo.\n"
        "En vez de guardar o leer TODO, recupera solo fragmentos relevantes (top-k) según la consulta.\n"
        "Esos fragmentos se inyectan como contexto para responder mejor, más rápido y con menos ruido.\n"
        "Esto permite continuidad (recordar cosas útiles) sin saturar el sistema."
    )

    return answer, retrieved, context_lines

# 4) Respuesta con contexto inyectado (prompt final para LLM)
def rag_answer(query: str, k: int = 2):
    retrieved = retrieve(query, k=k)

    # 4) Inyección en contexto
    context = "\n".join([f"- {text}" for text, score in retrieved])

    # Esto sería el prompt final que le darías al LLM
    prompt = f"""
Sos un asistente. Respondé usando SOLO el contexto.

Contexto:
{context}

Pregunta:
{query}

Respuesta:
""".strip()

    return prompt, retrieved

if __name__ == "__main__":
    q = "¿Por qué Jarvis necesita memoria externa y RAG?"

    answer, retrieved, context_lines = rag_answer_fake(q, k=3)

    print("\n=== TOP-K RECUPERADO ===")
    for text, score in retrieved:
        print(f"score={score:.3f} | {text}")

    print("\n=== RESPUESTA (FAKE) ===")
    print(answer)
