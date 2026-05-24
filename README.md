# Unstructured-io-Document-Processing-Pipeline

This project turns a PDF into searchable embeddings and stores the result in a persistent ChromaDB collection.

## What this project does

The repository currently supports two ways to run the pipeline:

1. A notebook workflow in `unstructred-io-document-preprocessing.ipynb`
2. A script-based workflow in `main.py`

Both workflows follow the same steps:

1. Extract semantic elements from the PDF with `unstructured`
2. Normalize the output into a stable schema
3. Chunk the document using `chunk_by_title()`
4. Generate embeddings with `sentence-transformers`
5. Store chunk metadata and vectors in ChromaDB
6. Run semantic search against the stored vectors

## Project structure

- `README.md` — project overview and setup instructions
- `main.py` — script entry point for the pipeline
- `unstructred-io-document-preprocessing.ipynb` — notebook version of the same workflow
- `Training_Data/` — sample PDF input files
- `chroma_db/` — persistent ChromaDB storage directory

## Prerequisites

Before you begin, make sure the following are installed:

- [Python 3.13](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv) — Python package and project manager
- Tesseract (recommended for OCR-heavy PDFs)

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

## Running the project

### Option A — Jupyter notebook

Open `unstructred-io-document-preprocessing.ipynb` in Jupyter or VS Code and run the cells in order.

The notebook expects the source PDF to be available in `Training_Data/` and writes the vector store to `./chroma_db`.

### Option B — Python script

Run the script directly:

```bash
python main.py
```

The current script uses the following defaults:

- Source file: `Training_Data/AML_NOTES_UNIT_1_2_3_4_5_merged.pdf`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- ChromaDB collection: `single_pdf_collection`
- Storage path: `./chroma_db`

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

- The notebook and `main.py` share the same pipeline logic.
- `main.py` now stores richer metadata, including `document_id`, `chunk_id`, `element_type`, `page_number`, `section`, `source_file`, and `created_at`.
- The current example targets a single PDF and persists the embeddings in a local ChromaDB store.