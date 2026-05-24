# Unstructured-io-Document-Processing-Pipeline

This project demonstrates an end-to-end document processing pipeline for PDF ingestion, semantic extraction, chunking, embedding generation, and vector search.

## What this project does

The notebook in this repository walks through the following workflow:

1. Extract semantic elements from a PDF using `unstructured`
2. Normalize the output into a consistent schema
3. Chunk the document using `chunk_by_title()`
4. Generate embeddings with `sentence-transformers`
5. Store the chunks and embeddings in ChromaDB
6. Run a semantic query against the stored vectors

## Project structure

- `README.md` — project overview and setup instructions
- `main.py` — Python entry point for the pipeline
- `unstructred-io-document-preprocessing.ipynb` — notebook version of the workflow
- `Training_Data/` — sample PDF files used for processing
- `chroma_db/` — persistent ChromaDB storage directory

## Prerequisites

Before you begin, make sure the following are installed:

- [Python 3.13](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv) — Python package and project manager
- Tesseract (optional but recommended for OCR-heavy PDFs)

> The current project configuration declares Python 3.13 in `pyproject.toml`.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies and environments.

### 1. Install uv

```bash
pip install uv
```

### 2. Install the required Python version

```bash
uv python install 3.13
```

### 3. Initialize the Project

- Use in existing project

    ```bash
    uv init
    ```

- Initialize your new project scaffold (this creates the `Unstructured-io-Document-Processing-Pipeline` directory):

    ```bash
    uv init Unstructured-io-Document-Processing-Pipeline
    cd Unstructured-io-Document-Processing-Pipeline
    ```

### 4. Create and Activate the Virtual Environment

Create an isolated virtual environment specifically for Python 3.12:

```bash
uv venv --python 3.13
```

**Activate the environment:**

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 5. Manage Dependencies

**Adding a new package:**
To install a new package and automatically add it to your `pyproject.toml` file, run:

```bash
uv add <package_name>
```

**Syncing existing packages:**
If you have just cloned the project or manually added dependencies to the `pyproject.toml` file, sync your virtual environment to install everything at once:

```bash
uv sync
```

### 5. Optional: verify Tesseract on macOS

If you are using OCR on macOS, ensure Homebrew's Tesseract binary is available:

```bash
which tesseract
```

## Running the notebook

Open `unstructred-io-document-preprocessing.ipynb` in Jupyter or VS Code and run the cells in order.

The notebook expects the sample PDF to be available in `Training_Data/` and stores the vector database in `./chroma_db`.

## Current dependencies

The project uses the following packages:

- `unstructured[pdf]`
- `sentence-transformers`
- `chromadb`
- `pytesseract`
- `tesseract`
- `pandas`

## Pipeline overview

```text
PDF
 ↓
Unstructured IO (extract + preprocess)
 ↓
Normalize schema
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
```

## Notes

- The notebook is the best place to start if you want to understand the full flow.
- `main.py` can be expanded later to turn the notebook logic into a reusable script.
- The current toy example uses a single PDF and writes embeddings to a persistent ChromaDB collection.