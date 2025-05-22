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
# 🕒 Load tickets
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
    matched = open_df[
        (open_df['description'].str.lower() == desc_lower) &
        (open_df['assignedgroup'].str.lower() == assigned_group_lower) &
        (open_df['status'].str.lower() == 'inprogress')
    ]
    if not matched.empty:
        return matched.iloc[0]
    # Try fuzzy match with LLM + RAG fallback
    retrieved = retrieve_similar(description, k=1)
    if not retrieved.empty:
        matched_group = retrieved.iloc[0]['assignedgroup']
        group_match = open_df[
            (open_df['assignedgroup'].str.lower() == matched_group.lower()) &
            (open_df['status'].str.lower() == 'inprogress')
        ]
        if not group_match.empty:
            return group_match.iloc[0]
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
        return True
    except Exception as e:
        return False

# ---------------------
# 🩠 Local LLM + RAG resolution using Hugging Face model
# ---------------------
@st.cache_resource(show_spinner=False)
def load_llm_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base")

llm_pipeline = load_llm_pipeline()

def generate_llm_response(description, retrieved_df, assigned_group=None):
    if assigned_group:
        retrieved_df = retrieved_df[
            (retrieved_df['assignedgroup'].str.lower() == assigned_group.lower()) &
            (retrieved_df['status'].str.lower() == 'closed')
        ]

    if retrieved_df.empty:
        return ("### ℹ️ No relevant previous tickets found.", "Unable to find similar closed tickets for this assigned group.")

    top_k = retrieved_df.head(1)

    context = "\n\n".join([
        f"Summary: {row.summary}\nDescription: {row.description}\nResolution: {row.resolution}"
        for _, row in top_k.iterrows()
    ])

    llm_prompt = (
        f"User Issue:\n{description}\n\n"
        f"Based on the following similar past ticket(s):\n{context}\n\n"
        f"Suggest a concise resolution using the provided 'Resolution' field only."
    )

    output = llm_pipeline(llm_prompt, max_new_tokens=100)
    generated_text = output[0]['generated_text'].strip()
    final_response = generated_text.replace('. ', '.\n')

    tickets_used = ", ".join([f"{row.ticket_id}" for _, row in top_k.iterrows()])

    formatted_prompt = f"{final_response}\n\nUsed Similar Ticket(s): {tickets_used}"

    return formatted_prompt, final_response


# ---------------------
# 🌐 Streamlit UI
# ---------------------
st.title("🎻 Incident Auto-Resolver (RAG + Local LLM + Auto Email)")

desc_input = st.text_area("📝 Enter new incident description:")

is_support_member = st.checkbox("👨‍💻 I am an IT support team member")

auto_email_placeholder = ""
if is_support_member:
    matched_ticket = open_df[open_df['description'].str.lower() == desc_input.lower()]
    if not matched_ticket.empty:
        auto_email_placeholder = matched_ticket.iloc[0].get('email', '')
        st.text_input("📧 Customer Email (auto-filled)", value=auto_email_placeholder, disabled=True, key="user_email")
    else:
        st.warning("No open ticket found with matching description to auto-fill email.")
        st.text_input("📧 Customer Email", key="user_email")
else:
    st.text_input("📧 Customer Email", key="user_email")

user_email = st.session_state.get("user_email", "").strip()

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

            formatted_prompt, suggestion = generate_llm_response(desc_input, retrieved)
            st.subheader("🤔 Suggested Resolution")
            st.write("**Resolution:**", suggestion)

            st.session_state['suggestion'] = suggestion

# --- Manual email sending of suggested resolution ---
if 'suggestion' in st.session_state:
    manual_email = st.text_input("Enter email to send suggested resolution:", key="manual_email")

    if st.button("✉️ Send Suggested Resolution Email"):
        manual_email = st.session_state.get("manual_email", "").strip()
        if not manual_email:
            st.warning("Please enter an email address to send the suggested resolution.")
        else:
            email_sent = send_email(
                subject="Suggested Resolution to Your Reported Issue",
                body=f"Hello,\n\nBased on your issue:\n\"{desc_input}\"\n\nHere is a suggested resolution:\n\n{st.session_state['suggestion']}\n\nRegards,\nSupport Team",
                to_email=manual_email
            )
            if email_sent:
                st.success(f"📤 Suggested resolution emailed to `{manual_email}`.")
                st.code(f"Subject: Suggested Resolution\nTo: {manual_email}\n\n{st.session_state['suggestion']}", language='text')
            else:
                st.error("❌ Failed to send the email. Please check the address or try again later.")

# ---------------------
# 🔄 Auto-Resolve All Open Tickets using LLM + RAG
# ---------------------
st.header("🧠 Auto-Resolve Open Tickets (RAG + LLM)")

if st.button("Run Auto-Resolution for All Open Tickets"):
    results = []
    inprogress_tickets = open_df[open_df['status'].str.lower() == 'inprogress']
    for _, row in inprogress_tickets.iterrows():
        description = row['description']
        assigned_group = row['assignedgroup']
        email = row['email']
        ticket_id = row['ticket_id']

        match = find_exact_match(description)
        if match is not None:
            resolution = match['resolution']
            source = "Exact Match"
        else:
            retrieved = retrieve_similar(description)
            _, resolution = generate_llm_response(description, retrieved, assigned_group)
            source = "LLM + RAG"

        results.append({
            "ticket_id": ticket_id,
            "description": description,
            "assigned_group": assigned_group,
            "email": email,
            "suggested_resolution": resolution,
            "source": source
        })

    st.subheader("📋 Auto-Resolved Ticket Suggestions")
    st.dataframe(pd.DataFrame(results))
