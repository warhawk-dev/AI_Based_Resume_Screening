# TalentAlign AI — Resume Screening System

An AI-powered resume screening and ranking web application built with Streamlit. It semantically compares uploaded PDF resumes against a job description using vector embeddings, and ranks candidates by match score.

---

## Features

- Upload multiple PDF resumes at once
- Paste any job description for comparison
- Semantic similarity scoring using sentence-transformers (not just keyword matching)
- Filter results by a configurable minimum match score
- Ranked results table with a bar chart visualization
- Clean two-tab UI: Upload & Configure → View Results

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit, streamlit-option-menu |
| Embeddings | sentence-transformers (`all-mpnet-base-v2`) |
| PDF Parsing | PyPDF2 |
| Visualization | Altair, Pandas |
| Config | python-dotenv |

---

## Project Structure

```
├── myvenv/               # Virtual environment (not committed)
├── .env                  # Environment variables (HuggingFace API token)
├── .gitignore
├── app.py                # Main Streamlit application
├── README.md
└── requirements.txt
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv myvenv

# Windows
myvenv\Scripts\activate

# macOS / Linux
source myvenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
```

> Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How It Works

1. **Paste** the job description into the text area.
2. **Upload** one or more PDF resumes.
3. **Set** a minimum match score threshold using the slider.
4. Click **Check Match** — the app encodes the job description and each resume into vector embeddings using `sentence-transformers/all-mpnet-base-v2`, then computes cosine similarity between them.
5. Switch to the **View Results** tab to see candidates ranked by match score, along with a bar chart comparison.

---

## Requirements

```
streamlit
python-dotenv
sentence-transformers
PyPDF2
pandas
altair
streamlit-option-menu
```

---

## Notes

- The embedding model (`all-mpnet-base-v2`) is loaded from HuggingFace and runs locally — no API calls are made for inference.
- The `.env` file is used to store your HuggingFace API token. Make sure it is listed in `.gitignore` and never committed to version control.
- Results are stored in Streamlit session state, so they persist when switching between tabs within the same session.
- Only resumes that meet or exceed the minimum score threshold are displayed in results.

