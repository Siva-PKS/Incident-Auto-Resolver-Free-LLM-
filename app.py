import streamlit as st
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import openai
from sentence_transformers import SentenceTransformer

# ---------------------
# 🔐 SMTP CONFIGURATION
# ---------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "xxxxxxx@gmail.com"         # 👈 Replace with your email
SMTP_PASSWORD = "jaao zsnq peke klgo"   # 👈 Use Gmail App Password

# ---------------------
# 📥 Load tickets
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
# 🔍 Load model & embeddings on closed tickets only
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
# 🔍 Similarity & exact match on closed tickets
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

# ---------------------
# 🔍 Check open tickets for same description, assignedgroup & closed status
# ---------------------
def check_open_tickets_for_auto_email(description, assigned_group):
    desc_lower = description.lower()
    assigned_group_lower = assigned_group.lower()

    filtered = open_df[
        (open_df['description'].str.lower() == desc_lower) &
        (open_df['assignedgroup'].str.lower() == assigned_group_lower) &
        (open_df['status'].str.lower() == 'closed')
    ]
    if not filtered.empty:
        return filtered.iloc[0]  # return first match
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
        st.error(f"Failed to send email: {e}")
        return False

# ---------------------
# 🤖 GPT-3.5 Turbo + RAG resolution
# ---------------------
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
            model="gpt-3.5-turbo",
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

# ---------------------
# 🌐 Streamlit UI
# ---------------------
st.title("🎫 Incident Auto-Resolver (RAG + GPT-3.5 + Auto Email)")

desc_input = st.text_area("📝 Enter new incident description:")
user_email = st.text_input("📧 Customer Email")
openai_api_key = st.text_input("🔑 OpenAI API Key", type="password")

if st.button("Resolve Ticket"):
    if not desc_input or not user_email or not openai_api_key:
        st.warning("Please fill in the incident description, email, and API key.")
    else:
        # 1. Check exact match in closed tickets
        match = find_exact_match(desc_input)
        if match is not None:
            st.success("✅ Exact match found!")
            st.write("**Resolution:**", match['resolution'])

            # Check open tickets for auto email (matching assigned group & closed status)
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
                # No matching open ticket for auto email; send to user email manually
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
            st.subheader("🧾 Similar Past Tickets")
            st.dataframe(retrieved[['ticket_id', 'summary', 'description', 'resolution', 'assignedgroup', 'status', 'date']])

            suggestion = generate_llm_response_openai(desc_input, retrieved, openai_api_key)
            st.subheader("🤖 Suggested Resolution")
            st.write(suggestion)

            # Show input box for manual email send
            manual_email = st.text_input("Enter email to send suggested resolution:")
            if st.button("✉️ Send Suggested Resolution Email"):
                if not manual_email:
                    st.warning("Please enter an email address to send the suggested resolution.")
                else:
                    email_sent = send_email(
                        subject="Suggested Resolution to Your Reported Issue",
                        body=f"Hello,\n\nBased on your issue:\n\"{desc_input}\"\n\nHere is a suggested resolution:\n\n{suggestion}\n\nRegards,\nSupport Team",
                        to_email=manual_email
                    )
                    if email_sent:
                        st.success("📤 Suggested resolution emailed.")
