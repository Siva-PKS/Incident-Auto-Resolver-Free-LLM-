# 🎻 Incident Auto-Resolver (RAG + Local LLM + Auto Email)

This project is a **Streamlit-based web app** that automates incident resolution by combining **Retrieval-Augmented Generation (RAG)**, **local LLMs**, and **email automation**. It helps support teams resolve incident tickets quickly by retrieving similar past tickets and suggesting responses—automatically sending the resolution via email when possible.

---

## 🚀 Features

- 🔍 **Similarity Search**: Finds similar tickets using sentence embeddings (MiniLM).
- 🧠 **Local LLM Inference**: Uses `google/flan-t5-base` to generate resolution suggestions.
- 📧 **Auto Email**: Sends email with suggested resolution to user.
- 🧾 **Exact Match Lookup**: Detects fully matched tickets and directly shows resolution.
- 💾 **Streamlit Caching**: Speeds up repeated model/data loading.

---

## 📁 File Structure
📦incident-auto-resolver
┣ 📄 app.py # Main Streamlit app with logic
┣ 📄 tickets_closed.csv # Historical tickets with resolution
┣ 📄 tickets_open.csv # Currently open tickets
┗ 📄 README.md


---

## 🛠️ Requirements

Install required Python packages:
pip install -r requirements.txt
Typical dependencies include:
streamlit
pandas
numpy
sentence-transformers
transformers
smtplib / email (built-in)
scikit-learn

---

## 🔐 Setup Secrets

Add your email password to Streamlit secrets:
# .streamlit/secrets.toml
email_password = "your_app_specific_password"

## 🧪 Running the App

streamlit run app.py

## ✅ Example Workflow

Enter a new incident description.
System checks for an exact match in historical tickets.
If not found, retrieves top 3 similar tickets and feeds them to LLM.
LLM suggests a resolution based on retrieved context.
You can email the resolution to the user directly.

## 📬 SMTP Configuration

Server: smtp.gmail.com
Port: 587
Add SMTP_USER and SMTP_PASSWORD via Streamlit secrets.

## 📄 License

MIT License © 2025 Siva PK



