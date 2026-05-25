# Unstructured-io-Document-Processing-Pipeline

This project turns PDFs into searchable embeddings and stores the results in a persistent ChromaDB collection.

## What this project does

The current pipeline is centered on `multi_pdf_parsing_pipeline.py`, which processes all PDFs found under `Training_Data/`.

The workflow is:

1. Extract semantic elements from each PDF with `unstructured`
2. Normalize the output into a stable schema
3. Chunk the document using `chunk_by_title()`
4. Generate embeddings with `sentence-transformers`
5. Store chunk metadata and vectors in ChromaDB
6. Run semantic search against the stored vectors

## Project structure

- `README.md` — project overview and setup instructions
- `multi_pdf_parsing_pipeline.py` — current entry point for the multi-PDF pipeline
- `main.py` — legacy single-PDF workflow
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

The notebook expects PDFs to be available in `Training_Data/` and writes the vector store to `./chroma_db`.

### Option B — Current Python script

Run the current multi-PDF pipeline directly:

```bash
python multi_pdf_parsing_pipeline.py
```

This script:

- discovers every PDF under `Training_Data/`
- processes all discovered PDFs
- stores embeddings in the `multi_pdf_collection` ChromaDB collection
- persists metadata including `document_id`, `chunk_id`, `element_type`, `page_number`, `section`, `source_file`, and `created_at`
- runs a final semantic search query for "What is k means clustering?"

### Option C — Legacy single-PDF script

If you need the older workflow, run:

```bash
python main.py
```

That script uses `Training_Data/AML_NOTES_UNIT_1_2_3_4_5_merged.pdf` as its source file.

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

- `multi_pdf_parsing_pipeline.py` is the active workflow.
- The current pipeline stores richer metadata, including `document_id`, `chunk_id`, `element_type`, `page_number`, `section`, `source_file`, and `created_at`.
- The active ChromaDB collection is `multi_pdf_collection`, and the data is persisted in `./chroma_db`.