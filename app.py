# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import datetime
import requests

# Title
st.title("🎫 Incident Auto-Resolver (Free LLM Edition)")

# Load ticket data
df = pd.read_csv('tickets.csv')
df = df.dropna()
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# Embedding model & Nearest Neighbors index
@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    temp_df = df.copy()
    temp_df['embedding'] = temp_df['description'].apply(lambda x: model.encode(x).tolist())
    embeddings = np.vstack(temp_df['embedding'].to_list()).astype('float32')
    nn = NearestNeighbors(n_neighbors=3, metric='cosine')
    nn.fit(embeddings)
    return model, nn, embeddings, temp_df

model, nn_model, embeddings, df = load_model_and_embeddings()

# Retrieve similar tickets
def retrieve_similar(description, k=3):
    query_embedding = model.encode([description]).astype('float32')
    distances, indices = nn_model.kneighbors(query_embedding, n_neighbors=k)
    return df.iloc[indices[0]]

# Free LLM via Hugging Face
def generate_llm_response(description, retrieved_df, hf_token):
    context = '\n\n'.join([
        f"Ticket ID: {row.ticket_id}\nSummary: {row.summary}\nDescription: {row.description}\nResolution: {row.resolution}\nAssigned Group: {row.assignedgroup}\nStatus: {row.status}\nDate: {row.date}"
        for _, row in retrieved_df.iterrows()
    ])
    
    prompt = f"""The user reported the following issue:

\"\"\"{description}\"\"\"

Here are similar past tickets and their resolutions:

{context}

Based on this, provide a concise and helpful resolution:"""

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            return result.get("error", "Failed to generate a resolution.")
    except Exception as e:
        st.error(f"Hugging Face API error: {e}")
        return "Failed to generate a resolution."

# UI Inputs
hf_token = st.text_input("Enter Hugging Face API Token", type="password")
user_input = st.text_area("📝 Enter a new incident description:")
submit = st.button("🔍 Suggest Resolution")

if submit and user_input and hf_token:
    retrieved = retrieve_similar(user_input)
    st.subheader("🧾 Similar Past Tickets")
    st.write(retrieved[['ticket_id','summary', 'description', 'resolution', 'assignedgroup', 'status','date']])

    resolution = generate_llm_response(user_input, retrieved, hf_token)
    st.subheader("🤖 Suggested Resolution")
    st.write(resolution)

    # Option to download resolution
    history = f"Timestamp: {datetime.datetime.now()}\nUser Input: {user_input}\n\nSuggested Resolution:\n{resolution}\n\n"
    history_bytes = history.encode('utf-8')
    st.download_button("⬇️ Download Resolution", data=history_bytes, file_name="resolution.txt", mime="text/plain")

elif submit:
    st.warning("Please enter both a ticket description and your Hugging Face API token.")
