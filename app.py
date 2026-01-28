import os
import io
import hashlib
import streamlit as st
from google import genai
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import chromadb
from sentence_transformers import SentenceTransformer


# ---------------- CONFIG ----------------
GEMINI_API_KEY = "PUT YOU API KEY HERE"
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash-lite"


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

    if isinstance(docs[0], list): docs = docs[0]
    text = "\n\n".join(docs)

    return text[:3000] + " ...(trimmed)" if len(text) > 3000 else text


def extract_using_rag(query):
    context = cached_rag_query(query)
    
    prompt = f"""
Analyze stored meeting records and extract approved decisions.

User Query: {query}

Relevant Stored Context:
\"\"\"{context}\"\"\"

Return only bullet points and avoid hallucination.
"""
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text.strip()


def build_prompt(minutes, query):
    return f"""
Extract approved decisions from these meeting minutes.

Rules:
- Bullet points only
- No hallucinated items
- Skip discussions

Query: {query}

Minutes:
\"\"\"{minutes}\"\"\"
"""

def extract_using_prompt(minutes, query):
    response = client.models.generate_content(
        model=MODEL_NAME, 
        contents=build_prompt(minutes, query)
    )
    return response.text.strip()


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
    max_width = width - 2 * margin
    
    for line in text.split("\n"):
        # Handle empty lines
        if not line.strip():
            y_position -= line_height / 2
            continue
            
        # Word wrap for long lines
        words = line.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            
            # Check if line is too long (approximate)
            if len(test_line) > 85:
                if current_line:
                    # Check if we need a new page
                    if y_position < 50:
                        pdf.showPage()
                        pdf.setFont("Helvetica", 11)
                        y_position = height - 50
                    
                    pdf.drawString(margin, y_position, current_line)
                    y_position -= line_height
                    current_line = word
                else:
                    # Single word is too long, just print it
                    if y_position < 50:
                        pdf.showPage()
                        pdf.setFont("Helvetica", 11)
                        y_position = height - 50
                    
                    pdf.drawString(margin, y_position, test_line)
                    y_position -= line_height
                    current_line = ""
            else:
                current_line = test_line
        
        # Print remaining text in current_line
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
st.set_page_config(page_title="AI Meeting Analyzer", layout="wide", page_icon="🧠")

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

    minutes_text = uploaded.read().decode("utf-8")

    # Auto-run when mode selected
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
