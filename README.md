# Board Meeting Decisions Tracker (AI Meeting Analyzer)

An AI-powered Streamlit application that analyzes board meeting minutes and answers questions about decisions, action items, meeting summaries, departmental challenges, and pending proposals. It uses Google Gemini AI and Retrieval-Augmented Generation (RAG) to answer questions grounded in the uploaded meeting text.

**Live App:** [https://board-meeting-decisions-tracker.streamlit.app/](https://board-meeting-decisions-tracker.streamlit.app/)

## Features

- Upload meeting minutes in `.txt` format
- Ask natural-language questions about the uploaded meeting
- Prompt mode for direct analysis of the current meeting minutes
- RAG mode backed by ChromaDB and Sentence Transformers
- Question-specific responses, including meeting summaries, action items with owners and deadlines, approved decisions, pending proposals, and departmental challenges
- Uses the current uploaded minutes as the primary source in RAG mode, with retrieved knowledge-base context as secondary information
- Updates an existing knowledge-base record when a file with the same filename is uploaded again
- Download responses as a PDF report

## Tech Stack

- Python and Streamlit
- Google Gemini API
- ChromaDB and Sentence Transformers
- ReportLab for PDF export

## Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/antalaraj/Board-Meeting-Decisions-Tracker.git
   cd Board-Meeting-Decisions-Tracker
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Set your Gemini API key. You can create a local file named `.streamlit/secrets.toml` (do not commit it) with:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

   Or set the `GEMINI_API_KEY` environment variable.

5. Run the Streamlit interface:

   ```bash
   streamlit run streamlit_app.py
   ```

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Select **Create app** and choose this repository.
3. Choose branch `main`.
4. Set **Main file path** to `streamlit_app.py`.
5. Open **Advanced settings → Secrets** and add:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

6. Deploy the app. The first build may take longer because Sentence Transformers downloads an embedding model.
7. If the app fails to start, open the app's **Manage app / Logs** view and check the first dependency or runtime error.

Keep API keys out of GitHub and never commit a real `.streamlit/secrets.toml` file. The app expects the secret name `GEMINI_API_KEY`.

## Limitations

- Meeting input currently supports `.txt` files only.
- Gemini API access and internet connectivity are required.
- RAG mode uses a local ChromaDB store; files stored on an ephemeral hosted filesystem may not persist across restarts or redeployments.
- English-language meetings are the primary target.
- AI responses should be checked against the original minutes, especially dates, owners, and whether a proposal was approved or remains pending.

## Future Enhancements

PDF and DOCX input, speech-to-text, multilingual support, task reminders, analytics, and role-based access control.

## Author

**Raj Antala**  
PGDM Student in AI and Data Science  
Adani Institute of Digital Technology Management (AIDTM)  
LinkedIn: [antalaraj](https://www.linkedin.com/in/antalaraj)
