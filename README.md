# Unstructured IO — Document Processing Pipeline

A Python pipeline that converts PDF documents into searchable vector embeddings and persists them in a local [ChromaDB](https://www.trychroma.com/) collection. Built with [`unstructured`](https://github.com/Unstructured-IO/unstructured), [`sentence-transformers`](https://www.sbert.net/), and managed with [`uv`](https://github.com/astral-sh/uv).

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Scripts Reference](#scripts-reference)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Notes & Known Limitations](#notes--known-limitations)

---

## What This Project Does

Given one or more PDF files, this project:

1. Extracts typed semantic elements (headings, narrative text, tables, etc.) using `unstructured`'s layout-aware `hi_res` strategy
2. Normalizes extracted elements into a stable, flat dictionary schema
3. Chunks content at heading/section boundaries using `chunk_by_title`
4. Generates dense vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`
5. Stores chunks, embeddings, and rich metadata in a persistent ChromaDB collection
6. Exposes a semantic search interface to query the stored knowledge

---

## Pipeline Overview

```
PDF files (Training_Data/)
        │
        ▼
┌─────────────────────────┐
│  Step 1 — Partitioning  │  unstructured hi_res strategy
│  (per PDF)              │  → Title, NarrativeText, Table, etc.
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 2 — Normalization │  Flatten to stable dict schema
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 3 — Chunking      │  chunk_by_title (max 1200 chars)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 4 — Embeddings    │  all-MiniLM-L6-v2
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 5 — ChromaDB      │  Persistent local vector store
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Step 6 — Semantic      │  Top-5 results with page, section,
│  Search                 │  element type, and content preview
└─────────────────────────┘
```

---

## Project Structure

```
.
├── README.md                                    # This file
├── pyproject.toml                               # Project metadata and dependencies
├── uv.lock                                      # Locked dependency tree
├── LICENSE                                      # License
│
├── multi_pdf_parsing_pipeline.py                # ← Active entry point (sequential multi-PDF)
├── multi_pdf_parallel_processing_pipeline.py    # Parallel multi-PDF variant (ProcessPoolExecutor)
├── main.py                                      # Legacy single-PDF workflow
├── unstructred-io-document-preprocessing.ipynb  # Notebook version of the pipeline
│
├── Training_Data/                               # Drop PDF files here (recursive glob supported)
│
└── chroma_db/                                   # Auto-created persistent ChromaDB storage
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.13+ | Declared in `pyproject.toml` |
| [uv](https://github.com/astral-sh/uv) | latest | Package and environment manager |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | any | Required only for scanned/image-based PDFs |

### Installing Tesseract (optional but recommended)

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Ubuntu / Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download the installer from the [Tesseract releases page](https://github.com/UB-Mannheim/tesseract/wiki).

> The pipeline sets `/opt/homebrew/bin` on `PATH` automatically for macOS. On other platforms, ensure `tesseract` is on your system `PATH`.

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for reproducible environment management.

### 1. Install uv

```bash
pip install uv
```

### 2. Install the required Python version

```bash
uv python install 3.13
```

### 3. Clone and enter the project

```bash
git clone <your-repo-url>
cd unstructured-io-document-processing-pipeline
```

### 4. Create the virtual environment

```bash
uv venv --python 3.13
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 5. Install dependencies

Sync from the lockfile for a fully reproducible install:

```bash
uv sync
```

To add a new package and record it in `pyproject.toml`:

```bash
uv add <package-name>
```

### 6. Add your PDFs

Place PDF files anywhere under `Training_Data/`. Subdirectories are supported — all PDFs are discovered recursively.

```
Training_Data/
├── unit1.pdf
├── unit2.pdf
└── advanced/
    └── unit5.pdf
```

---

## Running the Pipeline

### Option A — Sequential multi-PDF (recommended)

Processes all PDFs under `Training_Data/` one at a time. Loads the embedding model and opens the ChromaDB collection once, reusing them across all files.

```bash
python multi_pdf_parsing_pipeline.py
```

- Collection: `multi_pdf_collection`
- ChromaDB path: `./chroma_db`

### Option B — Parallel multi-PDF

Same as Option A but uses `ProcessPoolExecutor` with up to 4 workers. Each worker initialises its own model and ChromaDB client. Includes a skip check — already-processed PDFs (matched by `source_file` metadata) are not re-embedded.

```bash
python multi_pdf_parallel_processing_pipeline.py
```

> **Note:** The parallel script hardcodes `device="mps"` in the embedding call. Change this to `"cuda"` for NVIDIA GPUs or `"cpu"` if no accelerator is available.

### Option C — Single PDF (legacy)

Processes a single hardcoded file: `Training_Data/AML_NOTES_UNIT_1_2_3_4_5_merged.pdf`.

```bash
python main.py
```

- Collection: `single_pdf_collection`

### Option D — Jupyter Notebook

Open `unstructred-io-document-preprocessing.ipynb` in Jupyter or VS Code and run cells top-to-bottom. Expects PDFs in `Training_Data/` and writes the vector store to `./chroma_db`.

---

## Scripts Reference

### `multi_pdf_parsing_pipeline.py`

| Function | Description |
|---|---|
| `partition_pdf_and_print_elements(source_file)` | Parses a PDF into typed semantic elements using `hi_res` strategy |
| `normalize_elements(elements)` | Flattens elements to a stable dict schema for logging/debugging |
| `chunk_elements(elements)` | Splits elements into chunks anchored at title/heading boundaries |
| `embed_chunks(chunks, model)` | Encodes chunks into dense vectors using a pre-loaded SentenceTransformer |
| `store_chunks(collection, chunks, embeddings, source_file)` | Inserts chunks + metadata into ChromaDB |
| `query_chromadb(collection, model, question)` | Runs a semantic search and prints top-5 results |

### `multi_pdf_parallel_processing_pipeline.py`

Same function set as above, plus:

| Function | Description |
|---|---|
| `process_single_pdf(pdf_path)` | Self-contained worker function — loads model, checks for existing embeddings, and runs the full pipeline for one PDF |

### `main.py`

Single-file variant of the pipeline. Steps 1–6 are the same; the embedding model is loaded inside `generate_embeddings()` and returned to `main()` for reuse in the query step.

---

## Configuration

All configuration constants live at the top of each script:

| Constant | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model identifier |
| `CHROMA_DB_PATH` | `./chroma_db` | Persistent ChromaDB storage directory |
| `COLLECTION_NAME` | varies per script | ChromaDB collection name |
| `SOURCE_FILE_TO_ADD` | *(main.py only)* | Path to the single PDF for the legacy script |

### Chunking parameters (`chunk_by_title`)

| Parameter | Value | Description |
|---|---|---|
| `max_characters` | `1200` | Hard cap on chunk size |
| `new_after_n_chars` | `1000` | Soft target to begin a new chunk |
| `combine_text_under_n_chars` | `200` | Merge tiny fragments to reduce noise |

### Metadata stored per chunk

Each chunk stored in ChromaDB includes:

| Field | Description |
|---|---|
| `document_id` | UUID5 derived from the source file path — stable across re-runs |
| `chunk_id` | Sequential identifier within the document (`chunk_0`, `chunk_1`, …) |
| `element_type` | Unstructured element category (e.g. `Title`, `NarrativeText`, `Table`) |
| `page_number` | Page number in the source PDF (falls back to `0` if unavailable) |
| `section` | Section or title heading for the chunk (empty string if unavailable) |
| `source_file` | Relative path to the source PDF |
| `created_at` | UTC ISO 8601 timestamp of insertion |

---

## Dependencies

Declared in `pyproject.toml` and pinned in `uv.lock`:

| Package | Purpose |
|---|---|
| `unstructured[pdf]` | PDF parsing, element extraction, and chunking |
| `sentence-transformers` | Dense vector embeddings |
| `chromadb` | Local persistent vector store |
| `pytesseract` | Python bindings for Tesseract OCR |
| `tesseract` | OCR backend (system dependency) |
| `pandas` | Tabular data utilities |

---

## Notes & Known Limitations

- **`hi_res` strategy is slow.** It uses a layout detection model (and optionally OCR) per page. For large PDFs, expect significant processing time. Switch to `strategy="fast"` for a speed/accuracy trade-off.
- **Parallel script and `device="mps"`.** The parallel pipeline hardcodes Apple Silicon GPU (`device="mps"`). Update to `"cuda"` or `"cpu"` as needed before running on other hardware.
- **No deduplication in sequential script.** Unlike the parallel variant, `multi_pdf_parsing_pipeline.py` does not check for existing embeddings before inserting. Running it twice on the same PDFs will create duplicate chunks. Add a `collection.get(where={"source_file": pdf_path})` guard if needed.
- **ChromaDB is local only.** The pipeline uses `PersistentClient` which stores data on disk. For multi-machine or production deployments, consider switching to ChromaDB's HTTP client or a managed vector database.
- **`normalize_elements` is diagnostic only.** The normalized output is printed for inspection but not used downstream. It exists to decouple debugging from the main pipeline logic.
- **macOS PATH override.** The line `os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")` is safe to leave on non-macOS systems; it has no effect if the path does not exist.