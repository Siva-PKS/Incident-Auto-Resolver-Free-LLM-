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
SMTP_PASSWORD = st.secrets["email_password"]

# ---------------------
# 🔵 Logging configuration (commented out as per request)
# ---------------------
# logging.basicConfig(filename='email_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------
# 🕕 Load tickets
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
    similar_tickets = closed_df.iloc[indices].copy()
    
    # Assign similarity scores first
    similar_tickets['similarity'] = sims[indices]
    
    # Then round the similarity column
    similar_tickets['similarity'] = similar_tickets['similarity'].round(3)
    
    return similar_tickets


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
        # logging.info(f"Email successfully sent to {to_email} | Subject: {subject}")
        return True
    except Exception as e:
        # logging.error(f"Failed to send email to {to_email} | Subject: {subject} | Error: {e}")
        return False

# ---------------------
# 🧬 Local LLM + RAG resolution using Hugging Face model
# ---------------------
@st.cache_resource(show_spinner=False)
def load_llm_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm_pipeline = load_llm_pipeline()

def generate_llm_response(description, retrieved_df):
    if retrieved_df.empty:
        context = "No relevant past tickets found."
        ticket_info = None
    else:
        # Extract info from the top (or only) ticket
        row = retrieved_df.iloc[0]
        ticket_info = {
            "ticket_id": row.ticket_id,
            "description": row.description,
            "resolution": row.resolution
        }
        
        # Prepare context text for LLM prompt
        context = (
            f"Ticket ID: {ticket_info['ticket_id']}; "
            f"Description: {ticket_info['description']}; "
            f"Resolution: {ticket_info['resolution']}"
        )

    llm_prompt = (
        f"User Issue:\n{description}\n\n"
        f"Previous Ticket Context:\n{context}\n\n"
        f"Suggest a resolution:"
    )

    output = llm_pipeline(llm_prompt, max_new_tokens=200)
    generated_text = output[0]['generated_text'].strip()
    formatted_response = generated_text.replace('. ', '.\n')

    return ticket_info, formatted_response



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
            retrieved = retrieve_similar(desc_input, k=3)
            st.subheader("📜 Similar Past Tickets")
            st.dataframe(retrieved[['ticket_id', 'summary', 'description', 'resolution', 'similarity', 'assignedgroup', 'status', 'date']])
            
            if not retrieved.empty:
                top_ticket = retrieved.iloc[[0]]
                ticket_info, suggestion = generate_llm_response(desc_input, top_ticket)
            
                st.subheader("🤔 Suggested Resolution")
            
                if ticket_info:
                    st.markdown(f"**Ticket ID:** {ticket_info['ticket_id']}")
                    st.markdown(f"**Description:** {ticket_info['description']}")
                    st.markdown(f"**Resolution:** {ticket_info['resolution']}")           
                    st.session_state['suggestion'] = suggestion
            else:
                st.warning("No similar tickets found.")


# --- Manual email sending of suggested resolution ---
if 'suggestion' in st.session_state:
    manual_email = st.text_input("Enter email to send suggested resolution:", key="manual_email")

    if st.button("✉️ Send Suggested Resolution Email"):
        manual_email = st.session_state.get("manual_email", "").strip()
        if not manual_email:
            st.warning("Please enter an email address to send the suggested resolution.")
        else:
            email_sent = send_email(
                subject="Suggested Resolution to Your Reported Issue" {ticket_info['ticket_id']},
                body=f"Hello,\n\nBased on your issue:\n\"{desc_input}\"\n\nHere is a suggested resolution:\n\n{st.session_state['suggestion']}\n\nRegards,\nSupport Team",
                to_email=manual_email
            )
            if email_sent:
                st.success(f"📤 Suggested resolution emailed to `{manual_email}`.")
                st.code(f"Subject: Suggested Resolution\nTo: {manual_email}\n\n{st.session_state['suggestion']}", language='text')
                manual_email  = ""
            else:
                st.error("❌ Failed to send the email. Please check the address or try again later.")
