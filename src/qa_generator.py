import google.generativeai as genai

genai.configure(api_key=API_KEY)

MODEL = genai.GenerativeModel("gemini-1.5-flash")

def generate_answer(question, context_chunks):

    if not context_chunks:
        return {
            "answer": "No relevant scheme information found for your query.",
            "sources": [],
            "chunks": [],
        }

    prompt = _build_prompt(question, context_chunks)

    try:
        response = MODEL.generate_content(prompt)
        answer_text = response.text.strip()
    except Exception as e:
        answer_text = f"[Error calling Gemini API: {e}]"

    sources = list(dict.fromkeys(
        chunk.get("source", "unknown") for chunk in context_chunks
    ))

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks": context_chunks,
    }