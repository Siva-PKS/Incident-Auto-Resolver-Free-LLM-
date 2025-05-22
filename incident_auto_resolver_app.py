# Fix for: "RuntimeError: no running event loop" on Python 3.10+
import asyncio
import sys

if sys.platform.startswith('linux') and sys.version_info >= (3, 10):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# SMTP CONFIG
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "spkincident@gmail.com"
SMTP_PASSWORD = "jaao zsnq peke klgo"

# Load tickets
@st.cache_data(show_spinner=False)
def load_closed_tickets():
    df = pd.read_csv('tickets_closed.csv').dropna()
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_open_tickets():
    df = pd.read_csv('tickets_open.csv').dropna()
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    return df

closed_df = load_closed_tickets()
open_df = load_open_tickets()

# Load model + embeddings
@st.cache_resource(show_spinner=False)
def load_model_and_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = df.copy()
    df['embedding'] = df['description'].apply(lambda x: model.encode(x).tolist())
    embeddings = np.vstack(df['embedding'].to_list()).astype('float32')
    return model, embeddings, df

model, embeddings, closed_df = load_model_and_embeddings(closed_df)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_similar(description, k=3):
    query_emb = model.encode(description).astype('float32')
    sims = np.array([cosine_similarity(query_emb, emb) for emb in embeddings])
    indices = sims.argsort()[-k:][::-1]
    return closed_df.iloc[indices]

# Email sender
def send_email(subject, body, to_email):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception:
        return False

# LLM pipeline
@st.cache_resource(show_spinner=False)
def load_llm_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm_pipeline = load_llm_pipeline()

def generate_llm_response(description, retrieved_df):
    # Deduplicate based on description + resolution
    unique_cases = retrieved_df.drop_duplicates(subset=['description', 'resolution'])

    # Create formatted context from unique retrieved tickets
    context = "\n\n".join([
        f"Ticket ID: {row.ticket_id}; Summary: {row.summary}; Description: {row.description}; Resolution: {row.resolution}"
        for _, row in unique_cases.iterrows()
    ])

    # Prompt for LLM
    llm_prompt = (
        f"User Issue:\n{description}\n\n"
        f"Previous Ticket Context:\n{context}\n\n"
        f"Suggest a resolution:"
    )

    # Generate the response
    output = llm_pipeline(llm_prompt, max_new_tokens=200)
    generated_text = output[0]['generated_text'].strip()

    # Optional: insert newlines after sentence endings for better readability
    formatted_response = generated_text.replace('. ', '.\n')

    # Markdown-formatted prompt for display
    formatted_prompt = (
        f"### 🧾 User Issue\n"
        f"{description}\n\n"
        f"### 📂 Previous Ticket Context\n"
        f"{context}\n\n"
        f"### 💡 Suggested Resolution"
    )

    return formatted_prompt, formatted_response


# Streamlit UI
st.title("🛠️ Incident Auto-Resolver (RAG + LLM + Email)")

desc_input = st.text_area("📝 Enter new incident description:")
user_email = st.text_input("📧 Customer Email")

if st.button("Resolve Ticket"):
    if not desc_input or not user_email:
        st.warning("Please enter both the incident description and email.")
    else:
        retrieved = retrieve_similar(desc_input, k=3)
        best_match = retrieved.iloc[0]
        assigned_group = best_match['assignedgroup']
        matched_description = best_match['description']

        # Check in open and closed tickets for the group and status conditions
        open_match = open_df[
            (open_df['description'].str.lower() == matched_description.lower()) &
            (open_df['assignedgroup'].str.lower() == assigned_group.lower()) &
            (open_df['status'].str.lower() == 'inprogress')
        ]

        closed_match = closed_df[
            (closed_df['description'].str.lower() == matched_description.lower()) &
            (closed_df['assignedgroup'].str.lower() == assigned_group.lower()) &
            (closed_df['status'].str.lower() == 'closed')
        ]

        if not open_match.empty and not closed_match.empty:
            st.success("✅ Similar issue with matching group and status found.")
            st.subheader("💡 Resolution")
            resolution = closed_match.iloc[0]['resolution']
            st.write(resolution)

            email_to = open_match.iloc[0].get('email', user_email)
            email_sent = send_email(
                subject=f"Issue Resolved: {matched_description}",
                body=f"Hello,\n\nHere is the resolution for your reported issue:\n\n{resolution}\n\nRegards,\nSupport Team",
                to_email=email_to
            )
            if email_sent:
                st.info(f"📩 Resolution email sent to {email_to}")
            else:
                st.error("❌ Failed to send resolution email.")
        else:
            st.warning("⚠️ No exact condition match found. Generating resolution via LLM...")
            st.subheader("📜 Similar Past Tickets")
            for i, row in retrieved.iterrows():
                st.markdown(f"""
                **Ticket ID:** {row.ticket_id}  
                **Summary:** {row.summary}  
                **Assigned Group:** {row.assignedgroup}  
                **Status:** {row.status}  
                **Date:** {row.date}  
                **Description:**  
                {row.description}  
                **Resolution:**  
                _{row.resolution}_  
                ---
                """)

            suggested_resolution = generate_llm_response(desc_input, retrieved)
            st.subheader("🤖 Suggested Resolution")
            st.write(suggested_resolution)

            st.session_state['suggestion'] = suggested_resolution

# Manual email sending
if 'suggestion' in st.session_state:
    manual_email = st.text_input("📨 Enter email to send suggested resolution:", key="manual_email")

    if st.button("✉️ Send Suggested Resolution Email"):
        if not manual_email.strip():
            st.warning("Please enter an email address.")
        else:
            email_sent = send_email(
                subject="Suggested Resolution to Your Reported Issue",
                body=f"Hello,\n\nBased on your issue:\n\"{desc_input}\"\n\nHere is a suggested resolution:\n\n{st.session_state['suggestion']}\n\nRegards,\nSupport Team",
                to_email=manual_email.strip()
            )
            if email_sent:
                st.success(f"✅ Email sent to `{manual_email}`")
                st.code(f"Subject: Suggested Resolution\nTo: {manual_email}\n\n{st.session_state['suggestion']}", language='text')
            else:
                st.error("❌ Failed to send email.")
