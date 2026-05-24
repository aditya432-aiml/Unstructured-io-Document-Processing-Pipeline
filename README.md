# Unstructured-io-Document-Processing-Pipeline

## Prerequisites

Before you begin, make sure the following are installed on your system:

- [Python 3.12](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv) — Python package and project manager
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — required to run the local LLM

---

## Project Setup (Using UV)

This project uses [uv](https://github.com/astral-sh/uv), an extremely fast Python package and project manager. Follow the steps below to set up your local development environment.

### 1. Install UV

First, install `uv` globally on your system via pip:

```bash
pip install uv
```

### 2. Install the Required Python Version

Use `uv` to download and install Python 3.12:

```bash
uv python install 3.12
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
uv venv --python 3.12
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

---

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

For your use case (Open Source Unstructured, NOT API), this is stack:

* `unstructured` → extraction + semantic elements
* custom normalization layer → stable schema
* `chunk_by_title()` → intelligent chunking
* `sentence-transformers` → embeddings
* `chromadb` → vector storage