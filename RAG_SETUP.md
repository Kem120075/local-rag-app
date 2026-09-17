# 🚀 Local Multi-PDF RAG System Setup & Quickstart

This guide provides step-by-step instructions to get your local Retrieval-Augmented Generation (RAG) system up and running using **Streamlit**, **LangChain**, and **Ollama**.

---

## 📋 Prerequisites

Before running the app, make sure you have the following installed on your machine:

1. **Python 3.10+**
2. **Ollama** installed and running in the background.

---

## 🛠️ Setup Instructions

### 0. (First Time Only) Fix PowerShell Execution Policy

By default, Windows blocks running scripts in PowerShell, which prevents venv activation. If you see an `UnauthorizedAccess` / `running scripts is disabled on this system` error, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Type `Y` and press Enter when prompted. This only needs to be done once per machine.

---

### 1. Activate the Virtual Environment

Open PowerShell or your VS Code terminal in the project directory and activate your virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

> **Note:** You should see `(venv)` at the beginning of your terminal prompt.

---

### 2. Download Required Local Models

Ensure you have pulled the required LLM and embedding models via Ollama:

```powershell
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

---

### 3. Install Python Dependencies

If you haven't already, install the required packages into your active virtual environment:

```powershell
pip install streamlit langchain-community langchain-text-splitters langchain-chroma langchain-ollama pypdf
```

---

## ▶️ Running the Application

Launch the Streamlit web interface with the following command:

```powershell
streamlit run main.py
```

---

## 💡 How to Use the App

1. **Access Web UI:** Your browser will automatically open to `http://localhost:8501`.
2. **Upload PDFs:** Use the **Document Setup** sidebar on the left to drag and drop one or multiple PDF files.
3. **Wait for Indexing:** Wait until you see the green success message (`X PDF file(s) indexed!`).
4. **Chat & Interrogate:** Enter questions in the chat box at the bottom of the screen.
5. **Check Sources:** Expand the **📌 Source Citations** dropdown under any response to inspect the exact file names and page numbers used to answer your question.
