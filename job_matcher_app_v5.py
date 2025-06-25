import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from db import init_db, load_jobs, insert_jobs_from_df, insert_single_job

# --- Utility Functions ---
def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def match_jobs(cv_text, job_texts, top_n=5):
    documents = [cv_text] + job_texts
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=10, stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = np.argsort(cosine_sim)[::-1][:top_n]
    return top_indices, cosine_sim[top_indices]

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Streamlit UI ---
st.set_page_config(page_title="AI Job Matcher", layout="wide")
st.title("📌 AI Job Matching Platform")

# Initialize the database
init_db()

menu = st.sidebar.radio("Navigation", ["Job Matching", "Admin Panel"])

if menu == "Job Matching":
    st.subheader("🔍 Match Jobs Based on Your CV")
    uploaded_file = st.file_uploader("Upload your CV (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            cv_text = extract_text_from_pdf(uploaded_file)
        else:
            cv_text = uploaded_file.read().decode("utf-8")

        st.info("CV uploaded successfully. Generating job matches...")
        jobs_df = load_jobs()
        job_texts = jobs_df["text"].tolist()

        indices, scores = match_jobs(cv_text, job_texts)
        matched_jobs = jobs_df.iloc[indices].copy()
        matched_jobs["Match Score"] = scores

        st.success(f"Top {len(matched_jobs)} job matches:")
        st.dataframe(matched_jobs[["text", "Match Score"]].reset_index(drop=True))

        st.download_button(
            label="📥 Download Matched Jobs",
            data=convert_df_to_csv(matched_jobs[["text", "Match Score"]]),
            file_name="matched_jobs.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("✅ Recommended Job")
        best_match = matched_jobs.iloc[0]
        st.markdown(f"**Recommended Job (Highest Match Score):**\n\n{best_match['text']}")
    else:
        st.warning("Please upload your CV to begin.")

elif menu == "Admin Panel":
    st.subheader("🔐 Admin Upload Panel")
    admin_pass = st.text_input("Enter admin password", type="password")
    if admin_pass == "admin123":
        uploaded_csv = st.file_uploader("Upload new job CSV", type="csv")
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            if "text" in df.columns:
                df["text"] = df["text"].astype(str)
                insert_jobs_from_df(df)
                st.success("Jobs uploaded to database successfully.")
                st.cache_data.clear()
        st.markdown("---")
        st.subheader("🧹 Manual Text Preprocessing")
        raw_text = st.text_area("Paste job advert text here")
        if st.button("Save Job Post") and raw_text.strip():
            insert_single_job(raw_text.strip())
            st.success("Job advert added to database.")
    else:
        st.warning("Admin access required to view this panel.")