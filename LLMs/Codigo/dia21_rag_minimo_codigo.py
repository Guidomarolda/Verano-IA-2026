import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Documentos de ejemplo
docs = [
    "Overfitting ocurre cuando un modelo memoriza los datos de entrenamiento.",
    "La regularización reduce el overfitting penalizando la complejidad.",
    "Un embedding representa significado como un vector numérico.",
    "RAG combina recuperación semántica con un LLM.",
    "Los embeddings permiten búsqueda por significado y no por palabras."
]

# Convertir documentos a vectores TF-IDF
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(docs)

# Función para recuperar los documentos más similares a una consulta
def retrieve(query, k=2):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors)[0]
    top_k_idx = np.argsort(similarities)[::-1][:k]
    return [(docs[i], similarities[i]) for i in top_k_idx]

# Función para construir el prompt para el LLM
def build_prompt(query, retrieved_docs):
    context = "\n".join([f"- {doc}" for doc, _ in retrieved_docs])
    prompt = f"""
Sos un profesor de Machine Learning.
Usá solo el contexto provisto.

Contexto:
{context}

Pregunta:
{query}

Explicá de forma simple y sin fórmulas.
"""
    return prompt

# "Respuesta" FAKE usando RAG
def rag_answer(query):
    retrieved = retrieve(query, k=2)
    prompt = build_prompt(query, retrieved)

    print("=== CONTEXTO RECUPERADO ===")
    for doc, score in retrieved:
        print(f"{score:.2f} | {doc}")

    print("\n=== PROMPT FINAL ===")
    print(prompt)

# "Respuesta" FAKE usando RAG
def rag_answer(query):
    retrieved = retrieve(query, k=2)
    prompt = build_prompt(query, retrieved)

    print("=== CONTEXTO RECUPERADO ===")
    for doc, score in retrieved:
        print(f"{score:.2f} | {doc}")

    print("\n=== PROMPT FINAL ===")
    print(prompt)

    # Respuesta "fake" (simula que el LLM respondió usando el contexto)
    print("\n=== RESPUESTA (FAKE) ===")
    # juntamos solo el texto recuperado (sin scores)
    context_text = " ".join([doc for doc, _ in retrieved])
    answer = (
        "Generaliza mal porque está aprendiendo de memoria patrones específicos del entrenamiento "
        "en vez de aprender una regla que funcione en casos nuevos. "
        f"Según el contexto: {context_text}"
    )
    print(answer)

# Respuesta "fake" (simula que el LLM respondió usando el contexto)
if __name__ == "__main__":
    q = "¿Por qué un modelo que memoriza datos generaliza mal?"
    rag_answer(q)

