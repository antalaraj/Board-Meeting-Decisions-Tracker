import os
import io
import logging
import streamlit as st
from google import genai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import chromadb
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Meeting Analyzer", layout="wide", page_icon="🧠")


# ---------------- CONFIG ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        # A missing local secrets.toml should produce the friendly message below.
        GEMINI_API_KEY = ""

if not GEMINI_API_KEY:
    st.error("Gemini API key is not configured. Add GEMINI_API_KEY in Streamlit Community Cloud secrets or set it in your environment.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Use documented stable Gemini API model IDs as fallbacks.
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]


def generate_text(prompt):
    """Call Gemini with a fallback model for transient/model-specific failures."""
    models_to_try = list(dict.fromkeys([MODEL_NAME, *FALLBACK_MODELS]))
    last_error = None

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            text = (response.text or "").strip()
            if not text:
                st.warning("Gemini returned an empty response. Please try again.")
            return text
        except Exception as exc:
            last_error = exc
            status_code = (
                getattr(exc, "code", None)
                or getattr(exc, "status_code", None)
            )
            logger.warning(
                "Gemini request failed for model %s (status=%s): %s",
                model_name,
                status_code,
                type(exc).__name__,
            )

            # Authentication and quota errors won't be fixed by changing models.
            # A 400 can also mean that this particular model ID is unavailable,
            # so continue to the next model for model-specific errors.
            error_message = str(exc).lower()
            model_unavailable = any(
                phrase in error_message
                for phrase in (
                    "model not found",
                    "not found for api version",
                    "is not supported",
                    "unsupported model",
                    "does not exist",
                    "unknown model",
                )
            )
            if status_code in (401, 403, 429):
                break
            if status_code == 400 and not model_unavailable:
                break

    status_code = (
        getattr(last_error, "code", None)
        or getattr(last_error, "status_code", None)
    )
    logger.error("All configured Gemini model attempts failed (status=%s, error_type=%s)", status_code, type(last_error).__name__ if last_error else "UnknownError")
    if status_code == 429:
        st.error(
            "Gemini API quota or rate limit reached. Check your API usage "
            "and billing/quota settings, then try again."
        )
    elif status_code in (401, 403):
        st.error(
            "Gemini rejected the API key or access permissions. Check "
            "GEMINI_API_KEY and the API project's access settings."
        )
    elif status_code == 400:
        st.error(
            "Gemini rejected the request. Check the prompt and model settings "
            "in the Streamlit Cloud logs."
        )
    else:
        st.error(
            "Gemini is temporarily unavailable or the request failed for all "
            "available models. Please retry shortly. If it continues, check "
            "Manage app → Logs for the detailed error."
        )
    return ""


# ---------------- CACHED MODELS ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embedder()


# ---------------- RAG DB ----------------
chroma_client = chromadb.PersistentClient(path="rag_storage")
collection = chroma_client.get_or_create_collection(name="meeting_docs")


# ---------------- HELPERS ----------------
def embed_text(text):
    return embed_model.encode(text).tolist()


def store_in_rag(filename, text):
    stored = collection.get()
    files = [m.get("filename") for m in stored.get("metadatas", [])]

    if filename not in files:
        collection.add(
            documents=[text],
            ids=[filename],
            embeddings=[embed_text(text)],
            metadatas=[{"filename": filename}]
        )


@st.cache_data
def cached_rag_query(query):
    result = collection.query(query_embeddings=[embed_text(query)], n_results=1)
    docs = result.get("documents", [["None"]])

    if isinstance(docs[0], list):
        docs = docs[0]
    text = "\n\n".join(docs)

    return text[:3000] + " ...(trimmed)" if len(text) > 3000 else text


def extract_using_rag(query):
    context = cached_rag_query(query)

    prompt = f"""
You are a careful board-meeting secretary. Answer the user's question using ONLY the relevant stored meeting context below.

USER QUESTION:
{query}

MEETING CONTEXT:
{context}

DECISION-EXTRACTION RULES:
- Identify a decision when the minutes clearly say it was approved, accepted, agreed, resolved, adopted, or confirmed. Equivalent wording is allowed.
- Treat an action item as an approved decision only when the minutes show that the action was agreed or assigned, not merely suggested.
- Keep proposals, recommendations, requests, open questions, and discussion points separate from confirmed decisions.
- Do not infer approval from phrases such as "proposed", "recommended", "suggested", "should consider", or "will discuss".
- For each confirmed decision, state the decision and include the responsible person/team, deadline, or conditions only when explicitly present.
- If no confirmed decisions are present in the retrieved context, say: "No explicitly approved decisions found in the retrieved meeting context." Do not claim the full meeting had no decisions if the context may be incomplete.
- Do not invent details. Be concise and use clear bullet points. If useful, add a separate "Pending proposals / follow-ups" section.
"""
    return generate_text(prompt)


def build_prompt(minutes, query):
    return f"""
You are an accurate board-meeting secretary. Answer the user's question using ONLY the meeting minutes provided below.

USER QUESTION:
{query}

MEETING MINUTES:
{minutes}

HOW TO EXTRACT DECISIONS:
1. Extract confirmed decisions, not just sentences that sound important.
2. A decision is confirmed when the minutes explicitly indicate that participants approved, agreed to, resolved, adopted, authorized, or confirmed something. Equivalent clear wording is acceptable.
3. Include clearly agreed action items when the minutes assign or commit someone to do something. Include the owner and deadline only if stated.
4. Do NOT label a proposal, recommendation, request, suggestion, option, discussion point, or future consideration as an approved decision unless the minutes also confirm acceptance or approval.
5. Read the surrounding sentences for context. Approval may be expressed in a later sentence or in indirect but unambiguous wording.
6. Do not require the exact word "approved"; recognize clear equivalents such as "the board agreed", "it was decided", "the committee resolved", or "the team confirmed".
7. Do not invent names, dates, outcomes, votes, or approvals. Preserve important qualifications and conditions.
8. If there are confirmed decisions, return them as concise bullet points. For each, include the decision, owner, deadline, and relevant condition when explicitly available.
9. If there are no confirmed decisions, write exactly: "No explicitly approved decisions found in the provided meeting minutes."
10. If the minutes contain proposals but no confirmed decisions, add a separate "Pending proposals / follow-ups" section with those items clearly marked as NOT approved. Do not confuse this section with decisions.

Prefer accuracy over quantity. Do not return an empty result when the minutes contain a clearly agreed or confirmed decision.
"""


def extract_using_prompt(minutes, query):
    return generate_text(build_prompt(minutes, query))


def create_pdf(text):
    from datetime import datetime

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header Section
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 50, "Meeting Decisions Report")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, height - 70, f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

    # Horizontal line
    pdf.setLineWidth(1)
    pdf.line(40, height - 80, width - 40, height - 80)

    # Content Section
    y_position = height - 110
    pdf.setFont("Helvetica", 11)
    line_height = 16
    margin = 40

    for line in text.split("\n"):
        if not line.strip():
            y_position -= line_height / 2
            continue

        words = line.split()
        current_line = ""

        for word in words:
            test_line = current_line + " " + word if current_line else word

            if len(test_line) > 85:
                if current_line:
                    if y_position < 50:
                        pdf.showPage()
                        pdf.setFont("Helvetica", 11)
                        y_position = height - 50

                    pdf.drawString(margin, y_position, current_line)
                    y_position -= line_height
                    current_line = word
                else:
                    if y_position < 50:
                        pdf.showPage()
                        pdf.setFont("Helvetica", 11)
                        y_position = height - 50

                    pdf.drawString(margin, y_position, test_line)
                    y_position -= line_height
                    current_line = ""
            else:
                current_line = test_line

        if current_line:
            if y_position < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 11)
                y_position = height - 50

            pdf.drawString(margin, y_position, current_line)
            y_position -= line_height

    # Footer
    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawString(margin, 30, "Generated by AI Meeting Decisions Extractor")
    pdf.drawRightString(width - margin, 30, f"Page {pdf.getPageNumber()}")

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- UI DESIGN ----------------

# Custom CSS for square buttons
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        height: 120px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        border-color: #4CAF50;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton > button:active {
        background-color: #4CAF50 !important;
        color: white !important;
        border-color: #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>📄 AI Meeting Decisions Extractor</h1>", unsafe_allow_html=True)
st.write("Upload a meeting file → type your question → choose AI mode → output auto generates.")


uploaded = st.file_uploader("📁 Upload Minutes (.txt)", type=["txt"])
query = st.text_input("🔍 What do you want from this document?", "What decisions were approved?")


# ---------- MODE WITH SQUARE BUTTONS ----------
mode = None
result = None

if uploaded and query.strip():
    st.markdown("<br><b>⚙️ Choose AI Mode to Run:</b><br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⚡ Prompt Mode (Fast)\n\nDirect AI Processing", use_container_width=True):
            mode = "Prompt Mode (Fast)"

    with col2:
        if st.button("🔍 RAG Mode\n\nSearch Knowledge Base", use_container_width=True):
            mode = "RAG Mode (Search Knowledge Base)"

    minutes_text = uploaded.getvalue().decode("utf-8")

    if mode:
        with st.spinner("⏳ Generating response..."):
            if mode == "Prompt Mode (Fast)":
                result = extract_using_prompt(minutes_text, query)
            elif mode == "RAG Mode (Search Knowledge Base)":
                store_in_rag(uploaded.name, minutes_text)
                result = extract_using_rag(query)


# ---------------- RESULT ----------------
if result:
    st.success("✔ Successfully Processed!")
    st.subheader("🧠 Extracted Decisions")
    st.write(result)

    pdf = create_pdf(result)

    st.download_button("⬇ Download as PDF", pdf, "Meeting-Decisions.pdf", mime="application/pdf")
