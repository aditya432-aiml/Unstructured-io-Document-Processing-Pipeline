# =============================================================================
# PDF → ChromaDB Semantic Search Pipeline
# =============================================================================
# This script converts a PDF into semantic chunks, embeds them, stores them in
# ChromaDB, and executes a sample semantic search query.
#
# Pipeline:
#   1. Extract semantic elements from a PDF using `unstructured`
#   2. Chunk content by headings and section structure
#   3. Normalize chunks into a consistent metadata schema
#   4. Generate dense embeddings with a sentence-transformer model
#   5. Persist chunks, embeddings, and metadata in ChromaDB
#   6. Query the collection with a natural language prompt
#
# Input  : Training_Data/<your_pdf>.pdf
# Output : ./chroma_db  (persistent vector store)
# =============================================================================

# Optional helper: verify Tesseract is installed before enabling OCR support.
# Uncomment if your PDF contains scanned or image-based pages.
# import subprocess
# result = subprocess.run(["which", "tesseract"], capture_output=True, text=True)
# print(result.stdout)

from unstructured.chunking.title import chunk_by_title
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from datetime import datetime, timezone
import chromadb
import uuid
import os

# Add Homebrew's Tesseract location to PATH on macOS so OCR works if needed.
# This is harmless when OCR is not used.
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_FILE_TO_ADD  = "Training_Data/AML_NOTES_UNIT_1_2_3_4_5_merged.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH       = "./chroma_db"
COLLECTION_NAME      = "single_pdf_pipeline_collection"


# ---------------------------------------------------------------------------
# Step 1 — PDF Partitioning
# ---------------------------------------------------------------------------
def partition_pdf_and_print_elements(source_file):
    """
    Parse the PDF into typed semantic elements (Title, NarrativeText, Table, etc.)
    using unstructured's hi_res strategy for layout-aware extraction.

    Args:
        source_file (str): Path to the PDF file.

    Returns:
        list: Unstructured element objects extracted from the PDF.
    """
    print(f"\n{'='*60}")
    print(f"[Step 1] Partitioning PDF: {source_file}")
    print(f"{'='*60}")

    elements = partition_pdf(
        filename=source_file,
        strategy="hi_res",          # layout-aware parsing with OCR fallback
        languages=["eng"],
        include_metadata=True,
        infer_table_structure=True,
        extract_images_in_pdf=False,
    )

    print(f"[Step 1] ✓ Extracted {len(elements)} elements from PDF\n")

    # Preview the first 50 elements for inspection
    print(f"--- Preview: first 50 elements ---")
    for i, el in enumerate(elements[:50]):
        print(f"  [{i}] Type : {type(el).__name__}")
        print(f"       Text : {el.text[:120]}{'...' if len(el.text) > 120 else ''}")
        print(f"       Meta : {el.metadata}")
        print("  " + "-" * 76)

    return elements


# ---------------------------------------------------------------------------
# Step 2 — Chunking
# ---------------------------------------------------------------------------
def chunk_elements(elements):
    """
    Split elements into coherent chunks anchored at title/heading boundaries.
    Keeps chunks within the embedding model's token budget.

    Chunking parameters:
        max_characters       : hard cap per chunk
        new_after_n_chars    : soft target to start a new chunk
        combine_text_under_n_chars : merge tiny fragments to avoid noise

    Args:
        elements (list): Raw unstructured element objects.

    Returns:
        list: Chunked unstructured element objects.
    """
    print(f"\n{'='*60}")
    print(f"[Step 2] Chunking elements by title/section boundaries")
    print(f"{'='*60}")

    chunks = chunk_by_title(
        elements,
        max_characters=1200,
        new_after_n_chars=1000,
        combine_text_under_n_chars=200,
    )

    print(f"[Step 2] ✓ Created {len(chunks)} chunks\n")

    # Preview the first 3 chunks
    print("--- Preview: first 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  [Chunk {i}] ({len(chunk.text)} chars)")
        print(f"  {chunk.text[:300]}{'...' if len(chunk.text) > 300 else ''}")
        print("  " + "-" * 76)

    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Normalization
# ---------------------------------------------------------------------------
def normalize_elements(elements):
    """
    Flatten heterogeneous unstructured chunks into a uniform dictionary schema.
    This decouples downstream logic from unstructured's internal object types.

    Args:
        elements (list): Chunk objects returned by chunk_elements.

    Returns:
        list[dict]: Normalized document records.
    """
    print(f"\n{'='*60}")
    print(f"[Step 3] Normalizing {len(elements)} elements")
    print(f"{'='*60}")

    normalized_docs = []

    for idx, el in enumerate(elements):
        doc = {
            "id":           f"chunk_{idx}",
            "text":         el.text.strip(),
            "element_type": el.category,
            "page_number":  getattr(el.metadata, "page_number", None) or 0,
            "section":      getattr(el.metadata, "section", None) or
                            getattr(el.metadata, "title",   None) or "",
        }
        normalized_docs.append(doc)

    if normalized_docs:
        print(f"[Step 3] ✓ Normalization complete — sample record (index 0):")
        print(f"  {normalized_docs[0]}\n")
    else:
        print("[Step 3] ⚠ Warning: No elements were normalized. Check the PDF input.\n")

    return normalized_docs


# ---------------------------------------------------------------------------
# Step 4 — Embedding Generation
# ---------------------------------------------------------------------------
def generate_embeddings(normalized_chunks):
    """
    Encode each chunk into a dense vector using a local sentence-transformer model.
    Returns both the embedding array and the loaded model (to avoid reloading later).

    Args:
        normalized_chunks (list): List of normalized chunk dictionaries.

    Returns:
        tuple: (numpy array of embeddings, loaded SentenceTransformer model)
    """
    print(f"\n{'='*60}")
    print(f"[Step 4] Generating embeddings with model: {EMBEDDING_MODEL_NAME}")
    print(f"{'='*60}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chunk_texts = [c["text"] for c in normalized_chunks]  # extract plain text for embedding
    print(f"[Step 4] Encoding {len(chunk_texts)} chunks...")

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=True
    )

    print(f"[Step 4] ✓ Embeddings generated — shape: {embeddings.shape}\n")

    return embeddings, embedding_model


# ---------------------------------------------------------------------------
# Step 5 — ChromaDB Storage
# ---------------------------------------------------------------------------
def initialize_chromadb_and_store_chunks(normalized_chunks, embeddings, source_file):
    """
    Persist chunk embeddings and metadata into a local ChromaDB collection.
    Uses a deterministic document_id (UUID5) derived from the source file path
    so the same file always maps to the same ID across runs.

    Args:
        normalized_chunks (list[dict]): Normalized chunk dictionaries.
        embeddings        (np.ndarray): Embedding vectors aligned with chunks.
        source_file       (str): Path to the original PDF (used for metadata).

    Returns:
        chromadb.Collection: The collection with all chunks inserted.
    """
    print(f"\n{'='*60}")
    print(f"[Step 5] Storing chunks in ChromaDB at: {CHROMA_DB_PATH}")
    print(f"{'='*60}")

    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, source_file))
    created_at  = datetime.now(timezone.utc).isoformat()

    client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"[Step 5] Collection: '{COLLECTION_NAME}' | Document ID: {document_id}")
    print(f"[Step 5] Inserting {len(normalized_chunks)} chunks...")

    for i, chunk in enumerate(normalized_chunks):
        collection.add(
            ids        = [str(uuid.uuid4())],          # unique record id for ChromaDB
            documents  = [chunk["text"]],
            embeddings = [embeddings[i].tolist()],     # serialize vector to list format
            metadatas  = [{
                "document_id":  document_id,
                "chunk_id":     chunk["id"],
                "element_type": chunk["element_type"],
                "page_number":  chunk["page_number"],
                "section":      chunk["section"],
                "source_file":  source_file,
                "created_at":   created_at,
            }]
        )

    print(f"[Step 5] ✓ Successfully inserted {len(normalized_chunks)} chunks into ChromaDB\n")

    return collection


# ---------------------------------------------------------------------------
# Step 6 — Semantic Search
# ---------------------------------------------------------------------------
def query_chromadb(collection, embedding_model, question):
    """
    Embed the user's question and retrieve the top-5 semantically similar chunks
    from ChromaDB. Results include page number, section, and element type for
    easy source tracing.

    Args:
        collection      (chromadb.Collection): Target ChromaDB collection.
        embedding_model (SentenceTransformer): Already-loaded embedding model.
        question        (str): Natural language search query.

    Returns:
        None
    """
    print(f"\n{'='*60}")
    print(f"[Step 6] Running semantic search")
    print(f"{'='*60}")
    print(f"[Step 6] Query: \"{question}\"\n")

    query_embedding = embedding_model.encode([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    print(f"[Step 6] ✓ Top {len(results['documents'][0])} results:\n")
    for rank, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
        print(f"  Result #{rank}")
        print(f"  Page    : {meta['page_number']}")
        print(f"  Section : {meta['section'] or '(none)'}")
        print(f"  Type    : {meta['element_type']}")
        print(f"  Content : {doc[:300]}{'...' if len(doc) > 300 else ''}")
        print("  " + "-" * 76)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  PDF → ChromaDB Semantic Search Pipeline")
    print("=" * 60)
    print(f"  Source file : {SOURCE_FILE_TO_ADD}")
    print(f"  Model       : {EMBEDDING_MODEL_NAME}")
    print(f"  ChromaDB    : {CHROMA_DB_PATH}")
    print("=" * 60)

    # Step 1 — Extract semantic elements from the PDF
    elements = partition_pdf_and_print_elements(SOURCE_FILE_TO_ADD)

    # Step 2 — Group elements into chunks based on titles and section breaks
    chunks = chunk_elements(elements)
    
    # Step 3 — Convert chunk objects into consistent metadata records
    normalized_chunks = normalize_elements(chunks)

    # Step 4 — Generate dense vectors for each normalized chunk
    embeddings, embedding_model = generate_embeddings(normalized_chunks)

    # Step 5 — Save chunks, embeddings, and metadata in ChromaDB
    collection = initialize_chromadb_and_store_chunks(normalized_chunks, embeddings, SOURCE_FILE_TO_ADD)

    # Step 6 — Query the vector store with a sample question
    query_chromadb(collection, embedding_model, question="What is k means clustering?")

    print("\n[Done] Pipeline completed successfully.")


if __name__ == "__main__":
    main()