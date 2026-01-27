from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import streamlit as st
import altair as alt
from streamlit_option_menu import option_menu
import pandas as pd

# Load environment variables
load_dotenv()

# Load pretrained model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join(page.extract_text() for page in reader.pages)

# Calculate similarity
def calculate_similarity(job_desc, resume_text):
    job_emb = model.encode(job_desc)
    resume_emb = model.encode(resume_text)
    return util.cos_sim(job_emb, resume_emb)[0][0].item() * 100

def main():
    st.set_page_config(page_title="TalentAlign AI", layout="wide")

    # Horizontal Navigation Bar
    selected = option_menu(
        menu_title=None, 
        options=["Upload & Configure", "View Results"], 
        icons=["cloud-upload", "bar-chart-line"], 
        orientation="horizontal",
    )

    # Persistent state to store scores between tab switches
    if 'scores' not in st.session_state:
        st.session_state.scores = []

    # --- TAB 1: UPLOAD & CONFIGURE ---
    if selected == "Upload & Configure":
        st.title("TalentAlign AI")
        st.caption("Automatic resume ranking using AI")
        
        col1, col2 = st.columns(2)
        with col1:
            job_desc = st.text_area("Job Description", height=200)
            min_score = st.slider("Minimum Match Score (%)", 0, 100, 50)
        
        with col2:
            resumes = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)
            check = st.button("Check Match", use_container_width=True, type="primary")
        st.info("""
        **How it Works:**
        1. Provide the Job Description and upload PDF resumes.
        2. Our AI compares the semantic meaning of the text using vector embeddings.
        3. A similarity score is generated; only those above your 'Minimum Score' will be shown.
        """)

        if check:
            if not job_desc or not resumes:
                st.warning("Please upload resumes and enter job description.")
                return

            results = []
            for file in resumes:
                text = extract_text_from_pdf(file)
                score = round(calculate_similarity(job_desc, text), 2)
                if score >= min_score:
                    results.append((file.name, score))
            
            # Save to session state and notify user
            st.session_state.scores = sorted(results, key=lambda x: x[1], reverse=True)
            if st.session_state.scores:
                st.success(f"Successfully analyzed {len(st.session_state.scores)} resumes. Switch to 'View Results' tab.")
            else:
                st.warning("No resumes matched the selected score.")

    # --- TAB 2: VIEW RESULTS ---
    if selected == "View Results":
        if not st.session_state.scores:
            st.info("No data available. Please run the analysis in the 'Upload' tab first.")
            return

        st.subheader("Ranked Results")
        
        res_col, chart_col = st.columns([1, 1])

        with res_col:
            for i, (name, score) in enumerate(st.session_state.scores, 1):
                st.write(f"**Rank {i}**")
                st.write(f"Resume: {name}")
                st.write(f"Match Score: {score}%")
                st.divider()

        with chart_col:
            st.subheader("Match Score Visualization")
            df = pd.DataFrame(st.session_state.scores, columns=["Resume", "Match Score (%)"])
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Resume", axis=alt.Axis(labelAngle=0)),
                y="Match Score (%)"
            )
            st.altair_chart(chart, use_container_width=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)  # Creates distance from content
    st.divider()  # Adds a clean horizontal line
    st.markdown(
        "<p style='text-align: center; color: gray;'>© 2026 TalentAlign Resume Matcher System | All Rights Reserved</p>", 
        unsafe_allow_html=True
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()
