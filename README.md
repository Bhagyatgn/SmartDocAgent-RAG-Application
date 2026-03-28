# SmartDocAgent-RAG-Application

A PDF-based RAG chatbot built with LangChain, FAISS, HuggingFace embeddings, and Groq LLM.

## Features

- Upload and process PDF documents
- Retrieve relevant context with vector search
- Ask questions in natural language
- Structured answers with citations
- Streamlit UI and notebook workflows

## Project Structure

- `rag_streamlit.py` - Streamlit web app
- `simple_rag_core.py` - Core RAG logic and agent implementation
- `simple_rag_core_split.ipynb` - Clean modular notebook version
- `simple_RAG (testing).ipynb` - Earlier testing notebook version
- `uploaded_pdfs/` - PDF inputs
- `.env` - Local environment variables (not committed)

## Prerequisites

- Python 3.10+
- VS Code (recommended)
- Internet connection (for LLM/embedding downloads)

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install streamlit python-dotenv langchain langchain-community langchain-text-splitters langchain-groq langchain-huggingface sentence-transformers faiss-cpu pypdf
```

## Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_new_groq_api_key_here
```

## Run in Localhost (Streamlit UI)

```bash
.\.venv\Scripts\activate
python -m streamlit run rag_streamlit.py --server.address localhost --server.port 8501
```

- `http://localhost:8501`

If port `8501` is already in use, run on `8502`:

```bash
python -m streamlit run rag_streamlit.py --server.address localhost --server.port 8502
```

Then open:

- `http://localhost:8502`

Note: run plain file names in terminal (for example `rag_streamlit.py`), not markdown links.

## Run in Notebook (VS Code)

1. Open `simple_rag_core_split.ipynb` or `simple_RAG (testing).ipynb`
2. Select kernel: `.venv` interpreter
3. Restart kernel
4. Run all cells in order

## Recommended Flow

1. Load PDFs from `uploaded_pdfs/`
2. Build retrievers/vectorstore
3. Initialize `SmartDocAgent`
4. Ask questions and review responses

## Troubleshooting

- **Missing GROQ key**: Ensure `.env` contains `GROQ_API_KEY`
- **Kernel NameError**: Restart kernel and run all cells from top
- **Push blocked by secrets**: Remove secrets from files/history and rotate keys

## Security Notes

- Never hardcode API keys in notebooks or `.py` files
- Keep `.env` local only
- Rotate key immediately if exposed
