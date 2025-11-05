# job_matcher_app_v1.py
import re
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io
import base64
import sqlite3
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx
import plotly.express as px  # used for chart visualization
from pathlib import Path
import tempfile
import os

# -------------------------
# Streamlit configuration
# -------------------------
st.set_page_config(page_title="AI Job Matcher", layout="wide")

# --- Disclaimer Anchor Link ---
st.markdown("<h1 id='disclaimer'></h1>", unsafe_allow_html=True)

with st.expander("📌 Disclaimer – Please Read Before Uploading Your CV"):
    st.markdown("""
    ### Purpose of the Platform
    This AI-Powered Job Matching Platform assists job seekers in finding relevant job opportunities aligned with their skills and experience.

    ### Important Notice
    - ⚠️ Uploading your CV does **not guarantee** matches.
    - 🧠 Works best with text-based CVs (not scanned PDFs).
    - 🔍 Use as a complement to other job search methods.

    ### Data Privacy & Consent
    - 📁 CVs are processed **temporarily** and not stored.
    - 🔐 No personal data is retained without consent.
    - ✅ By uploading your CV, you consent to automated processing for job matching.

    ### Contact
    Email: **support@example.com**

    ---
    *Disclaimer updated: June 2025*
    """)

# -------------------------
# Structured Job Extraction
# -------------------------
def extract_structured_job(text: str) -> dict:
    cleaned_text = re.sub(r'\s+', ' ', (text or "")).strip()
    job_title = re.search(r"(Job\s*Title|Position)\s*[:\-–]?\s*(.*?)(?=\s[A-Z]{2,}|Duty\s*Station|Location|WHO\sWE\sARE|Responsibilities|Summary\s*of\s*the\s*role|Purpose|Requirements|Qualifications|Education)", cleaned_text, re.IGNORECASE)
    location = re.search(r"(Location|Duty\s*Station)\s*[:\-–]?\s*(.*?)(?=\s[A-Z]{2,}|Supervisor|About\sUs|Responsibilities|Summary\s*of\s*the\s*role|Purpose|Requirements|Qualifications|Education)", cleaned_text, re.IGNORECASE)
    organization = re.search(r"(?:WHO\sWE\sARE|About\sUs[:\-–]?)\s*(.*?)(?=\sResponsibilities|Purpose|The\sRole|Requirements|Qualifications|Education)", cleaned_text, re.IGNORECASE)
    responsibilities = re.search(r"(Responsibilities|Summary\s*of\s*the\s*role|Purpose)\s*[:\-–]?\s*(.*?)(?=\sRequirements|Qualifications|Education|How\s*to\s*Apply|Submission\s*Guidelines|Deadline)", cleaned_text, re.IGNORECASE)
    requirements = re.search(r"(Requirements|Qualifications|Education)\s*[:\-–]?\s*(.*?)(?=\sHow\s*to\s*Apply|Submission\s*Guidelines|Deadline|Disclaimer|$)", cleaned_text, re.IGNORECASE)
    application = re.search(r"(How\s*to\s*Apply|Submission\s*Guidelines|Deadline)\s*[:\-–]?\s*(.*?)(?=Disclaimer|$)", cleaned_text, re.IGNORECASE)
    email = re.search(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", cleaned_text)

    return {
        "Job Title": job_title.group(2).strip() if job_title and job_title.group(2) else "",
        "Location": location.group(2).strip() if location and location.group(2) else "",
        "Organization": organization.group(1).strip() if organization and organization.group(1) else "",
        "Responsibilities": responsibilities.group(2).strip() if responsibilities and responsibilities.group(2) else "",
        "Requirements": requirements.group(2).strip() if requirements and requirements.group(2) else "",
        "Application": application.group(2).strip() if application and application.group(2) else "",
        "Contact Email": email.group(0) if email else ""
    }

# -------------------------
# Database Path & Admin List
# Robust resolution (works locally & on Streamlit Cloud)
# -------------------------
REPO_DIR = Path(__file__).resolve().parent
INTENDED_DATA_DIR = REPO_DIR / "data"


def resolve_db_path():
    try:
        INTENDED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        candidate = INTENDED_DATA_DIR / "jobmatcher.db"
        # small write test
        test_file = INTENDED_DATA_DIR / ".write_test"
        with open(test_file, "w") as f:
            f.write("ok")
        try:
            test_file.unlink()
        except Exception:
            pass
        return candidate
    except Exception as e:
        tmp = Path(tempfile.gettempdir()) / "jobmatcher_fallback.db"
        st.warning(f"Couldn't use repo data/ folder ({e}). Falling back to temp DB at {tmp}. This DB is ephemeral on Streamlit Cloud.")
        return tmp

DB_PATH = resolve_db_path()
ADMIN_EMAILS = ["admin@matcher.com", "ayuathm@gmail.com"]

# Show debug info (helpful on Streamlit Cloud)
#st.write("DEBUG: resolved DB_PATH:", str(DB_PATH))
#st.write("DEBUG: repo dir:", str(REPO_DIR))
#st.write("DEBUG: cwd:", os.getcwd())

# -------------------------
# Initialize Streamlit Session State
# -------------------------
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False
if "bookmarked_jobs" not in st.session_state:
    st.session_state.bookmarked_jobs = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# -------------------------
# Database Helpers
# -------------------------
def get_conn():
    # tolerant connection for multi-threaded Streamlit usage
    return sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)

def init_db():
    try:
        with get_conn() as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                timestamp TEXT
            );
            ''')
            conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                login_time TEXT,
                is_admin INTEGER
            );
            ''')
            conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                user_email TEXT,
                feedback TEXT,
                comment TEXT,
                timestamp TEXT
            );
            ''')
    except Exception as e:
        st.error(f"init_db failed: {e}")
        st.write("DB_PATH (init):", DB_PATH)

def add_timestamp_column_if_missing():
    try:
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(jobs)")
            columns = [col[1] for col in cursor.fetchall()]
            if "timestamp" not in columns:
                cursor.execute("ALTER TABLE jobs ADD COLUMN timestamp TEXT")
                conn.commit()
    except Exception as e:
        st.warning(f"Could not ensure timestamp column: {e}")

def insert_single_job(text):
    timestamp = datetime.datetime.now().isoformat()
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO jobs (text, timestamp) VALUES (?, ?)", (text, timestamp))
    except Exception as e:
        st.error(f"Failed to insert job: {e}")
        st.write("DB_PATH (insert):", DB_PATH)

def load_jobs():
    try:
        with get_conn() as conn:
            df = pd.read_sql_query("SELECT * FROM jobs", conn)
        return df
    except Exception as e:
        st.warning(f"Could not load jobs from DB: {e}")
        st.write("DB_PATH (load):", DB_PATH)
        return pd.DataFrame(columns=["id", "text", "timestamp"])

def log_user_login(email, is_admin):
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO users (email, login_time, is_admin) VALUES (?, ?, ?)",
                         (email, datetime.datetime.now().isoformat(), int(is_admin)))
    except Exception as e:
        st.warning(f"Failed to log user login: {e}")

def store_feedback(job_id, email, feedback, comment):
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO feedback (job_id, user_email, feedback, comment, timestamp) VALUES (?, ?, ?, ?, ?)",
                         (job_id, email, feedback, comment, datetime.datetime.now().isoformat()))
    except Exception as e:
        st.warning(f"Failed to store feedback: {e}")

# initialize DB at startup
init_db()

# -------------------------
# File Text Extraction (safer)
# -------------------------
def extract_text_from_pdf(file):
    try:
        # PdfReader can accept file-like objects
        reader = PdfReader(file)
        pages_text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages_text.append(t)
        return "\n".join(pages_text)
    except Exception as e:
        st.warning(f"Failed to extract PDF text: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        # docx.Document can accept a file-like object
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        st.warning(f"Failed to extract DOCX text: {e}")
        return ""

# -------------------------
# App Layout
# -------------------------
st.title("🤖 AI-Powered Job Matching Platform")

# --- Sidebar Login ---
with st.sidebar:
    st.markdown("🔐 **User Login**")
    email = st.text_input("Enter your email")
    if st.button("Login"):
        st.session_state.user_email = email
        st.session_state.is_admin = email in ADMIN_EMAILS
        log_user_login(email, st.session_state.is_admin)
        st.success(f"Logged in as {email}")

# --- Tabs (Admin & User Views) ---
is_admin = st.session_state.get("is_admin", False)
if is_admin:
    tab1, tab2 = st.tabs(["🛠 Admin Panel", "🤖 Job Matching"])
else:
    tab2, = st.tabs(["🤖 Job Matching"])

# --- Admin Panel ---
if is_admin:
    with tab1:
        st.header("Upload Job Descriptions (CSV, PDF, DOCX)")
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "pdf", "docx"])
        if uploaded_file:
            # ensure DB exists and schema updated
            init_db()
            add_timestamp_column_if_missing()

            texts = []
            name = uploaded_file.name.lower()
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

            if name.endswith(".csv"):
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    if "description" in df_upload.columns:
                        texts = df_upload["description"].dropna().astype(str).tolist()
                    elif "text" in df_upload.columns:
                        texts = df_upload["text"].dropna().astype(str).tolist()
                    else:
                        st.warning("CSV file should contain a 'description' or 'text' column.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
            elif name.endswith(".pdf"):
                text = extract_text_from_pdf(uploaded_file)
                if text.strip():
                    texts = [text]
            elif name.endswith(".docx"):
                text = extract_text_from_docx(uploaded_file)
                if text.strip():
                    texts = [text]

            if texts:
                count = 0
                for text in texts:
                    insert_single_job(text)
                    count += 1
                st.success(f"✅ Uploaded and saved {count} job(s) to the database.")
            else:
                st.warning("Could not extract any text from the uploaded file.")

# --- Job Matching Panel ---
with tab2:
    st.header("📎 Upload your CV")
    uploaded_cv = st.file_uploader("Upload your CV (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    user_cv = ""
    if uploaded_cv:
        try:
            uploaded_cv.seek(0)
        except Exception:
            pass

        if uploaded_cv.name.lower().endswith(".pdf"):
            user_cv = extract_text_from_pdf(uploaded_cv)
        elif uploaded_cv.name.lower().endswith(".docx"):
            user_cv = extract_text_from_docx(uploaded_cv)
        elif uploaded_cv.name.lower().endswith(".txt"):
            try:
                user_cv = uploaded_cv.read().decode("utf-8")
            except Exception:
                uploaded_cv.seek(0)
                user_cv = uploaded_cv.getvalue().decode("utf-8", errors="ignore")

    st.sidebar.header("👤 Your Profile")
    user_edu = st.sidebar.selectbox("🎓 Education Level", ["", "High School", "Diploma", "Bachelor's", "Master's", "PhD"])
    user_exp = st.sidebar.slider("💼 Years of Experience", 0, 30, 1)
    user_skills = st.sidebar.multiselect("🛠️ Key Skills", ["Monitoring", "Finance", "Teaching", "Engineering", "Procurement", "Coordination", "Reporting", "Python", "Data Analysis"])

    if user_cv:
        df = load_jobs()
        if df.empty:
            st.warning("❗ No job postings found in the database. Please upload from Admin Panel first.")
        else:
            # prepare TF-IDF and compute similarity
            tfidf = TfidfVectorizer(stop_words="english")
            docs = df["text"].fillna("").astype(str).tolist() + [user_cv]
            try:
                tfidf_matrix = tfidf.fit_transform(docs)
                cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
                df = df.copy()
                df["Match Score"] = cosine_sim
                top_jobs = df.sort_values(by="Match Score", ascending=False).head(5)

                st.subheader("📊 Top 5 Job Matches")
                top_jobs["Short Title"] = top_jobs["text"].apply(lambda x: extract_structured_job(x).get("Job Title", "")[:30] or (str(x)[:30] if pd.notna(x) else ""))

                fig = px.bar(
                    top_jobs,
                    x="Short Title",
                    y="Match Score",
                    title="Top 5 Job Matches",
                    labels={"Short Title": "Job Title"},
                    text="Match Score"
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(yaxis_range=[0, 1], xaxis_tickangle=-30, height=400)
                st.plotly_chart(fig, use_container_width=True)

                for _, row in top_jobs.iterrows():
                    st.markdown(f"### Job ID {row.get('id', '')}")
                    st.write(f"**Match Score:** {round(float(row['Match Score']), 2)}")
                    structured = extract_structured_job(row["text"])
                    with st.expander("📄 View Job Details"):
                        st.write(f"**Job Title:** {structured['Job Title']}")
                        st.write(f"**Location:** {structured['Location']}")
                        st.write(f"**Organization:** {structured['Organization']}")
                        st.markdown("**Responsibilities:**")
                        st.write(structured['Responsibilities'])
                        st.markdown("**Requirements:**")
                        st.write(structured['Requirements'])
                        st.markdown("**Application Instructions:**")
                        st.write(structured['Application'])
                        st.write(f"📧 **Contact Email:** {structured['Contact Email']}")

                        if st.button(f"⭐ Save Job {row['id']}", key=f"save_{row['id']}"):
                            st.session_state.bookmarked_jobs.append(row.to_dict())
                            st.success(f"Job {row['id']} saved to bookmarks!")

                        feedback_key = f"feedback_radio_{row['id']}"
                        feedback = st.radio("Was this job useful?", ["", "👍 Yes", "👎 No"], key=feedback_key)
                        comment_key = f"comment_text_{row['id']}"
                        comment = st.text_area("Optional comment", key=comment_key)

                        if feedback in ["👍 Yes", "👎 No"]:
                            # store feedback only once per session per job
                            session_flag = f"submitted_feedback_{row['id']}"
                            if not st.session_state.get(session_flag, False):
                                store_feedback(row["id"], st.session_state.user_email, feedback, comment)
                                st.session_state[session_flag] = True
                                st.success("Feedback submitted!")

                if not top_jobs.empty:
                    csv_data = top_jobs.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download CSV of Top Jobs",
                        data=csv_data,
                        file_name="top_jobs.csv",
                        mime="text/csv",
                        key="download_top_jobs_csv"
                    )

                if st.session_state.bookmarked_jobs:
                    st.subheader("⭐ Bookmarked Jobs")
                    bookmark_df = pd.DataFrame(st.session_state.bookmarked_jobs)
                    st.dataframe(bookmark_df)

            except Exception as e:
                st.error(f"Failed to compute or display matches: {e}")
    else:
        st.info("Upload your CV (PDF, DOCX, or TXT) to find matches.")

# End of file
