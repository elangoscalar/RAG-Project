# RAG PDF Assistant

A chatbot that lets you upload a PDF and ask questions about it.
Built with Streamlit, Ollama, and ChromaDB. Runs fully offline on your local machine.

---

## What it does

- Upload any PDF from the browser
- Ask questions about the PDF in plain English
- Get answers with the exact page number it came from
- Remembers the last 6 messages in the conversation

---

## Files

- app.py — the main UI (run this file)
- config.py — all settings like model name and chunk size
- resources.py — loads the AI models
- pdf_processor.py — reads the PDF and stores it in the database
- rag_pipeline.py — finds relevant parts and generates the answer

---

## How to run

1. Clone the repo
git clone https://github.com/elangoscalar/My_RAG_Assignment.git
cd My_RAG_Assignment

2. Install packages
pip install -r requirements.txt

3. Install Ollama from ollama.com

4. Download the AI model
ollama pull llama3.2:3b

5. Run the app
streamlit run app.py

6. Open browser and go to
http://localhost:8501

---

## Tools used

- Streamlit — builds the web interface
- Ollama — runs the AI model locally
- ChromaDB — stores and searches the PDF content
- HuggingFace — converts text into vectors
- PyMuPDF — reads PDF files

---

## Note

This project was originally designed to use vLLM, but vLLM requires an NVIDIA GPU
which is not available on Mac. Ollama was used instead as it works on Mac and
provides the same functionality.

---

Author: Elan
GitHub: https://github.com/elangoscalar
