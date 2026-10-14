"""Vercel serverless endpoint for meeting-decision extraction."""
from http.server import BaseHTTPRequestHandler
import json
import math
import os

from google import genai

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "gemini-embedding-001"
MAX_BODY_BYTES = 1_500_000
MAX_TEXT_CHARS = 120_000


def json_response(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def chunk_text(text, chunk_size=1800, step=1500):
    text = text.strip()
    if not text:
        return []
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def vector_values(embedding):
    values = getattr(embedding, "values", None)
    if values is None and isinstance(embedding, dict):
        values = embedding.get("values")
    return list(values or [])


def cosine_similarity(left, right):
    if not left or not right or len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return -1.0
    return dot / (left_norm * right_norm)


def retrieve_relevant_chunks(client, document_text, query):
    chunks = chunk_text(document_text)
    if not chunks:
        return document_text[:6000]

    # Keep embedding work bounded on serverless deployments.
    chunks = chunks[:60]
    document_embeddings = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=chunks,
    )
    query_embedding = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
    )
    vectors = [vector_values(item) for item in (document_embeddings.embeddings or [])]
    query_vectors = [vector_values(item) for item in (query_embedding.embeddings or [])]
    if not vectors or not query_vectors:
        raise ValueError("The embedding service returned no vectors.")

    ranked = []
    for index, vector in enumerate(vectors):
        ranked.append((cosine_similarity(vector, query_vectors[0]), index))
    ranked.sort(reverse=True)
    selected = [chunks[index] for _, index in ranked[:5]]
    return "\n\n--- Relevant meeting excerpt ---\n\n".join(selected)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                json_response(self, 413, {"error": "Request is empty or too large. Please upload a smaller .txt file."})
                return

            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            document_text = str(payload.get("text", "")).strip()
            query = str(payload.get("query", "")).strip()
            mode = str(payload.get("mode", "prompt")).lower()

            if not document_text:
                json_response(self, 400, {"error": "Upload a .txt meeting-minutes file first."})
                return
            if len(document_text) > MAX_TEXT_CHARS:
                json_response(self, 413, {"error": "The text is too long. Please keep it below 120,000 characters."})
                return
            if not query:
                json_response(self, 400, {"error": "Enter a question about the meeting minutes."})
                return
            if mode not in ("prompt", "rag"):
                json_response(self, 400, {"error": "Choose Prompt Mode or RAG Mode."})
                return

            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                json_response(self, 500, {"error": "GEMINI_API_KEY is missing. Add it in Vercel → Project Settings → Environment Variables, then redeploy."})
                return

            client = genai.Client(api_key=api_key)
            if mode == "rag":
                context = retrieve_relevant_chunks(client, document_text, query)
                prompt = (
                    "You are an assistant that extracts approved decisions from board meeting minutes.\n"
                    "Answer the user's question using only the relevant meeting excerpts below.\n"
                    "Return concise bullet points. Do not invent decisions. If the excerpts do not "
                    "contain the answer, say so clearly.\n\n"
                    f"User question:\n{query}\n\nRelevant meeting excerpts:\n{context}"
                )
            else:
                prompt = (
                    "Extract approved decisions from these board meeting minutes.\n"
                    "Rules: Return concise bullet points only; do not hallucinate; distinguish approved "
                    "decisions from discussion or proposals; say when no approved decision is found.\n\n"
                    f"User question:\n{query}\n\nMeeting minutes:\n{document_text[:80000]}"
                )

            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            answer = (response.text or "").strip()
            if not answer:
                json_response(self, 502, {"error": "Gemini returned an empty response. Please try again."})
                return
            json_response(self, 200, {"result": answer, "mode": mode, "model": MODEL_NAME})
        except Exception as exc:
            print("Meeting analysis failed:", repr(exc))
            json_response(self, 502, {"error": "AI processing failed. Check the Gemini API key, model access, and Vercel function logs, then try again."})

    def do_GET(self):
        json_response(self, 200, {"status": "ok", "service": "Board Meeting Decisions Tracker API"})

    def log_message(self, format, *args):
        print("API:", format % args)
