"""
qa_generator.py
---------------
RAG pipeline: takes retrieved context chunks + user question,
sends them to Google Gemini API, and returns a grounded answer with citations.

Requires environment variable:
    GOOGLE_API_KEY=your_key_here
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from google import genai

# ── Setup ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY environment variable is not set.\n"
        "Get a free key at https://aistudio.google.com/app/apikey"
    )

_client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"   # latest stable free-tier model


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Construct a RAG prompt that instructs Gemini to answer ONLY from context.

    Args:
        question:       User's question.
        context_chunks: Retrieved chunks with "text" and "source" fields.

    Returns:
        Full prompt string.
    """
    context_text = ""
    for i, chunk in enumerate(context_chunks, start=1):
        source = chunk.get("source", "unknown")
        text   = chunk.get("text", "")
        context_text += f"\n--- Chunk {i} (Source: {source}) ---\n{text}\n"

    prompt = f"""You are Bharat Sahayak AI, a helpful assistant for Indian citizens seeking 
information about government welfare schemes.

Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information about this in my database."
Always mention which scheme/document your answer comes from.
Keep the answer concise, clear, and in simple language.

CONTEXT:
{context_text}

USER QUESTION:
{question}

ANSWER:"""
    return prompt


# ── Generation ────────────────────────────────────────────────────────────────

import time

def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:

    if not context_chunks:
        return {
            "answer": "No relevant scheme information found for your query.",
            "sources": [],
            "chunks": [],
        }

    prompt = _build_prompt(question, context_chunks)

    retries = 3
    for attempt in range(retries):
        try:
            response = _client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "temperature": 0.2
                }
            )

            answer_text = response.text.strip() if response.text else ""
            break

        except Exception as e:

            if attempt < retries - 1:
                time.sleep(5)
            else:
                answer_text = f"[Gemini API Error: {e}]"

    sources = list(dict.fromkeys(
        chunk.get("source", "unknown") for chunk in context_chunks
    ))

    return {
        "answer": answer_text,
        "sources": sources,
        "chunks": context_chunks,
    }