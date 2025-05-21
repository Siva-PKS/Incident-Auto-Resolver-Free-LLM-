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
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------------
# 🔐 SMTP CONFIGURATION
# ---------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "spkincident@gmail.com"
SMTP_PASSWORD = "jaao zsnq peke klgo"

# ---------------------
# 🙵 Logging configuration
# ---------------------
logging.basicConfig(filename='email_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------
# 🗕 Load tickets
# ---------------------
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

# ---------------------
# 🔍 Load model & embeddings
# ---------------------
@st.cache_resource(show_spinner=False)
def load_model_and_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = df.copy()
    df['embedding'] = df['description'].apply(lambda x: model.encode(x).tolist())
    embeddings = np.vstack(df['embedding'].to_list()).astype('float32')
    return model, embeddings, df

model, embeddings, closed_df = load_model_and_embeddings(closed_df)

# ---------------------
# 🔍 Similarity Search
# ---------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_similar(description, k=3):
    query_emb = model.encode(description).astype('float32')
    sims = np.array([cosine_similarity(query_emb, emb) for emb in embeddings])
    indices = sims.argsort()[-k:][::-1]
    return closed_df.iloc[indices]

def find_exact_match(description):
    desc_lower = description.lower()
    match_rows = closed_df[closed_df['description'].str.lower() == desc_lower]
    return match_rows.iloc[0] if not match_rows.empty else None

def check_open_tickets_for_auto_email(description, assigned_group):
    desc_lower = description.lower()
    assigned_group_lower = assigned_group.lower()
    filtered = open_df[
        (open_df['description'].str.lower() == desc_lower) &
        (open_df['assignedgroup'].str.lower() == assigned_group_lower) &
        (open_df['status'].str.lower() == 'closed')
    ]
    if not filtered.empty:
        return filtered.iloc[0]
    return None

# ---------------------
# 📧 Email sender
# ---------------------
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
        logging.info(f"Email successfully sent to {to_email} | Subject: {subject}")
        return True
    except Exception as e:
        logging.error(f"Failed to send email to {to_email} | Subject: {subject} | Error: {e}")
        return False

# ---------------------
# 🧠 Local LLM + RAG resolution using Hugging Face model
# ---------------------
@st.cache_resource(show_spinner=False)
def load_llm_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm_pipeline = load_llm_pipeline()

def generate_llm_response(description, retrieved_df):
    context = "\n\n".join([
        f"Ticket ID: {row.ticket_id}\nSummary: {row.summary}\nDescription: {row.description}\nResolution: {row.resolution}"
        for _, row in retrieved_df.iterrows()
    ])

    prompt = f"""User Issue:
{description}

Previous Ticket Context:
{context}

Suggest a resolution:"""

    output = llm_pipeline(prompt, max_new_tokens=200)
    return output[0]['generated_text']

# ---------------------
# 🌐 Streamlit UI
# ---------------------
st.title("🎻 Incident Auto-Resolver (RAG + Local LLM + Auto Email)")

desc_input = st.text_area("📝 Enter new incident description:")
user_email = st.text_input("📧 Customer Email")

if st.button("Resolve Ticket"):
    if not desc_input or not user_email:
        st.warning("Please fill in the incident description and email.")
    else:
        match = find_exact_match(desc_input)
        if match is not None:
            st.success("✅ Exact match found!")
            st.write("**Resolution:**", match['resolution'])

            auto_email_ticket = check_open_tickets_for_auto_email(match['description'], match['assignedgroup'])
            if auto_email_ticket is not None:
                auto_email = auto_email_ticket.get('email', None)
                if auto_email:
                    email_sent = send_email(
                        subject=f"Issue Resolved: {match['description']}",
                        body=f"Hello,\n\nHere is the resolution for your reported issue:\n\n{match['resolution']}\n\nRegards,\nSupport Team",
                        to_email=auto_email
                    )
                    if email_sent:
                        st.info(f"📩 Resolution email sent automatically to {auto_email}")
                else:
                    st.warning("No email found in matching open ticket for auto sending.")
            else:
                email_sent = send_email(
                    subject=f"Issue Resolved: {match['description']}",
                    body=f"Hello,\n\nHere is the resolution for your reported issue:\n\n{match['resolution']}\n\nRegards,\nSupport Team",
                    to_email=user_email
                )
                if email_sent:
                    st.info("📩 Resolution email sent to your provided email.")
        else:
            st.warning("No exact match. Retrieving similar tickets and generating resolution...")
            retrieved = retrieve_similar(desc_input)
            st.subheader("📜 Similar Past Tickets")
            st.dataframe(retrieved[['ticket_id', 'summary', 'description', 'resolution', 'assignedgroup', 'status', 'date']])

            suggestion = generate_llm_response(desc_input, retrieved)
            st.subheader("🤔 Suggested Resolution")
            st.write(suggestion)

           manual_email = st.text_input("Enter email to send suggested resolution:", key="manual_email")

if st.button("✉️ Send Suggested Resolution Email"):
    manual_email = st.session_state.get("manual_email", "").strip()
    if not manual_email:
        st.warning("Please enter an email address to send the suggested resolution.")
    else:
        email_sent = send_email(
            subject="Suggested Resolution to Your Reported Issue",
            body=f"Hello,\n\nBased on your issue:\n\"{desc_input}\"\n\nHere is a suggested resolution:\n\n{suggestion}\n\nRegards,\nSupport Team",
            to_email=manual_email
        )
        st.write(f"send_email returned: {email_sent}")  # Debug output

        if email_sent:
            st.success(f"📤 Suggested resolution emailed to {manual_email}.")
            st.markdown("✅ Email dispatch logged. You can check `email_log.txt` for record.")
            st.code(f"Subject: Suggested Resolution\nTo: {manual_email}\n\n{suggestion}", language='text')
        else:
            st.error("❌ Failed to send the email. Please check the address or try again later.")

with st.expander("📄 View Email Log"):
    try:
        with open("email_log.txt", "r") as f:
            log_content = f.read()
        st.text_area("Email Log", log_content, height=200)
    except FileNotFoundError:
        st.info("No email log found yet.")
