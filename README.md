# Board Meeting Decisions Tracker (AI Meeting Analyzer)

The Board Meeting Decisions Tracker is an AI-powered application that analyzes board meeting minutes and automatically extracts important decisions, insights, and summaries. The system uses Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and Google Gemini AI to provide accurate, context-aware answers strictly based on the uploaded meeting document. This application helps organizations reduce manual review effort, improve clarity, and ensure effective tracking of board-level decisions.

## Problem Statement
Manual review of meeting minutes is time-consuming, error-prone, and inefficient. Keyword-based searches fail to understand context, synonyms, and implicit decisions. There is a need for an intelligent system that can understand meeting context and deliver accurate, structured results.

## Features
- Upload meeting minutes in .txt format
- Ask any question related to the meeting content
- Context-aware question answering using RAG
- Structured bullet-point responses
- AI-powered decision and insight extraction
- Download results as a professionally formatted PDF
- Simple and user-friendly Streamlit interface

## Tech Stack
- Frontend: Streamlit
- Backend: Python
- AI & NLP: Google Gemini AI, LangChain, Sentence Transformers
- Vector Database: ChromaDB
- PDF Generation: ReportLab

## System Architecture
User uploads meeting file → Text extraction and chunking → Embedding generation → Vector storage (ChromaDB) → Retrieval-Augmented Generation (RAG) → Gemini AI response → Structured output → PDF export

## Installation and Setup
1. Clone the repository  
git clone https://github.com/your-username/Board-Meeting-Decisions-Tracker.git  
cd Board-Meeting-Decisions-Tracker  

2. Create and activate virtual environment  
python -m venv .venv  
Windows: .venv\Scripts\activate  
Linux/macOS: source .venv/bin/activate  

3. Install dependencies  
pip install -r requirements.txt  

4. Run the application  
streamlit run app.py  

## Applications
Corporate board meetings, management review meetings, governance and compliance tracking, educational and institutional meetings, and project documentation analysis.

## Limitations
Supports only .txt files, requires internet connectivity for AI inference, supports English language only, and does not include real-time transcription.

## Future Enhancements
Support for PDF and DOCX files, speech-to-text transcription, multilingual support, automated task reminders, analytics dashboard, and role-based access control.

## 👨‍💻 Author
Raj Antala  
🎓 PGDM Student in AI and Data Science  
🏫 Adani Institute of Digital Technology Management (AIDTM)  
📍 Gandhinagar, India  
💡 Passionate about turning data into meaningful insights and building intelligent systems  
📧 antalaraj214@gmail.com  
🔗 www.linkedin.com/in/antalaraj
