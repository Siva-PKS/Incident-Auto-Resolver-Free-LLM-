import streamlit as st
import pandas as pd
import numpy as np
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import openai

# --- Load tickets CSV ---
@st.cache_data(show_spinner=False)
def load_tickets():
    df = pd.read_csv('tickets.csv').dropna()
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    return df

df = load_tickets()

# --- SentenceTransformer embeddings ---
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def load_model_and_embeddings(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = df.copy()
    df['embedding'] = df['description'].apply(lambda x: model.encode(x).tolist())
    embeddings = np.vstack(df['embedding'].to_list()).astype('float32')
    return model, embeddings, df

model, embeddings, df = load_model_and_embeddings(df)

# --- Cosine similarity ---
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_similar(description, k=3):
    query_emb = model.encode(description).astype('float32')
    sims = np.array([cosine_similarity(query_emb, emb) for emb in embeddings])
    indices = sims.argsort()[-k:][::-1]
    return df.iloc[indices]

# --- Exact match lookup ---
def find_exact_match(description):
    # Simple exact match on description (case-insensitive)
    desc_lower = description.lower()
    match_rows = df[df['description'].str.lower() == desc_lower]
    if not match_rows.empty:
        return match_rows.iloc[0]
    return None

# --- Send email function ---
def send_email(subject, body, to_email, smtp_server, smtp_port, smtp_user, smtp_password):
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# --- LLM generation with OpenAI GPT ---
def generate_llm_response_openai(description, retrieved_df, openai_api_key):
    openai.api_key = openai_api_key

    context = '\n\n'.join([
        f"Ticket ID: {row.ticket_id}\nSummary: {row.summary}\nDescription: {row.description}\nResolution: {row.resolution}\nAssigned Group: {row.assignedgroup}\nStatus: {row.status}\nDate: {row.date}"
        for _, row in retrieved_df.iterrows()
    ])

    prompt = f"""You are a helpful IT support assistant.

The user reported the following issue:

\"\"\"{description}\"\"\"

Here are similar past tickets and their resolutions:

{context}

Based on this, provide a concise and helpful resolution for the user's issue."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful support assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return "Failed to generate a resolution."

# --- Streamlit UI ---

st.title("🎫 Incident Auto-Resolver with RAG + OpenAI")

desc_input = st.text_area("Enter new incident description:")
user_email = st.text_input("User Email")

st.markdown("### SMTP Email Settings (for auto email sending)")
smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535, value=587)
smtp_user = st.text_input("SMTP Username (email address)")
smtp_password = st.text_input("SMTP Password", type="password")

openai_api_key = st.text_input("OpenAI API Key", type="password")

if st.button("Resolve Ticket"):
    if not desc_input or not user_email or not openai_api_key or not smtp_user or not smtp_password:
        st.warning("Please fill in all fields: description, user email, OpenAI API key, SMTP username, and SMTP password.")
    else:
        # 1. Check for exact match
        match = find_exact_match(desc_input)
        if match is not None:
            st.success("✅ Exact match found!")
            st.write("**Resolution:**", match['resolution'])

            # Send email with exact resolution
            email_sent = send_email(
                subject=f"Issue Resolved: {match['description']}",
                body=f"Here is the resolution:\n\n{match['resolution']}",
                to_email=user_email,
                smtp_server=smtp_server,
                smtp_port=smtp_port,
                smtp_user=smtp_user,
                smtp_password=smtp_password,
            )
            if email_sent:
                st.info("📧 Auto-reply email sent to your email address.")
        else:
            st.warning("No exact match found. Retrieving similar tickets and generating suggestion via LLM...")
            retrieved = retrieve_similar(desc_input)
            st.subheader("🧾 Similar Past Tickets")
            st.write(retrieved[['ticket_id', 'summary', 'description', 'resolution', 'assignedgroup', 'status', 'date']])

            suggestion = generate_llm_response_openai(desc_input, retrieved, openai_api_key)
            st.subheader("🤖 Suggested Resolution")
            st.write(suggestion)

            if st.button("Send Suggested Reply via Email"):
                email_sent = send_email(
                    subject="Suggested Resolution to Your Reported Issue",
                    body=f"Issue: {desc_input}\n\nSuggested Resolution:\n{suggestion}",
                    to_email=user_email,
                    smtp_server=smtp_server,
                    smtp_port=smtp_port,
                    smtp_user=smtp_user,
                    smtp_password=smtp_password,
                )
                if email_sent:
                    st.success("📤 Suggested resolution emailed to you.")

# Optionally, add download button for the generated resolution if desired
