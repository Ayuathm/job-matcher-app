
import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import glob
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Utility Functions ---
@st.cache_data
def load_job_data():
    all_files = glob.glob("data/*.csv")
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 50]
    df = df[~df["text"].str.contains("lorem ipsum|dummy text|n/a", case=False)]
    df = df.drop_duplicates(subset="text")
    return df.reset_index(drop=True)

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

def save_uploaded_csv(uploaded_file):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = f"data/job_data_{timestamp}.csv"
    df = pd.read_csv(uploaded_file)
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 50]
    df = df[~df['text'].str.contains("lorem ipsum|dummy text|n/a", case=False)]
    df = df.drop_duplicates(subset="text")
    df.to_csv(save_path, index=False)
    return save_path

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Streamlit UI ---
st.set_page_config(page_title="AI Job Matcher", layout="wide")
st.title("📌 AI Job Matching Platform")

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
        jobs_df = load_job_data()
        job_texts = jobs_df["text"].tolist()

        indices, scores = match_jobs(cv_text, job_texts)
        matched_jobs = jobs_df.iloc[indices].copy()
        matched_jobs["Match Score"] = scores


        # --- Job Category Filtering ---
        if "category" in matched_jobs.columns:
            st.subheader("🔍 Filter by Job Category")
            unique_categories = matched_jobs["category"].dropna().unique()
            selected_categories = st.multiselect("Select categories to filter", sorted(unique_categories))
            if selected_categories:
                matched_jobs = matched_jobs[matched_jobs["category"].isin(selected_categories)]
                st.info(f"{len(matched_jobs)} jobs found in selected categories.")
        else:
            st.info("No job category data available to filter.")

        # --- Job Sector Visualization ---
        if "sector" in matched_jobs.columns:
            st.subheader("📊 Job Sector Distribution")
            sector_counts = matched_jobs["sector"].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=140)
            ax2.axis("equal")
            st.pyplot(fig2)
        else:
            st.info("No job sector data available for visualization.")


        st.success(f"Top {len(matched_jobs)} job matches:")
        st.dataframe(matched_jobs[["text", "Match Score"]].reset_index(drop=True))

        
        # --- Interactivity Enhancements ---
        threshold = st.slider("🔢 Set minimum match score", 0.0, 1.0, 0.5, step=0.05)
        filtered_jobs = matched_jobs[matched_jobs["Match Score"] >= threshold]
        st.write(f"Showing {len(filtered_jobs)} jobs with score ≥ {threshold}")

        # Expandable job descriptions
        st.subheader("📄 Top Job Matches (Expandable View)")
        for idx, row in filtered_jobs.iterrows():
            with st.expander(f"Match #{idx+1} (Score: {row['Match Score']:.2f})"):
                st.markdown(row["text"])

        # Highlight top matching terms
        from collections import Counter
        def extract_keywords(cv_text, job_text):
            cv_words = set(cv_text.lower().split())
            job_words = job_text.lower().split()
            matched = [w for w in job_words if w in cv_words]
            top_terms = Counter(matched).most_common(5)
            return ", ".join([term for term, _ in top_terms])

        filtered_jobs["Top Matched Terms"] = filtered_jobs["text"].apply(lambda x: extract_keywords(cv_text, x))
        st.subheader("📌 Match Details with Top Terms")
        st.dataframe(filtered_jobs[["text", "Match Score", "Top Matched Terms"]].reset_index(drop=True))

        # Visualization of match scores
        import matplotlib.pyplot as plt
        # Extract email from CV
        import re
        email_match = re.search(r"[\w\.-]+@[\w\.-]+", cv_text)
        st.markdown(f"📧 Detected Email: `{email_match.group(0)}`" if email_match else "📧 No email detected")


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
            save_path = save_uploaded_csv(uploaded_csv)
            st.success(f"File uploaded and saved as {os.path.basename(save_path)}")
            st.cache_data.clear()
        st.markdown("---")
        st.subheader("🧹 Manual Text Preprocessing")
        raw_text = st.text_area("Paste job advert text here")
        if st.button("Clean and Save This Job Post") and raw_text.strip():
            df = pd.DataFrame({"text": [raw_text.strip()]})
            df = df[df["text"].str.len() > 50]
            df = df[~df["text"].str.contains("lorem ipsum|dummy text|n/a", case=False)]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            df.to_csv(f"data/manual_job_post_{timestamp}.csv", index=False)
            st.success("Job advert added to database.")
    else:
        st.warning("Admin access required to view this panel.")