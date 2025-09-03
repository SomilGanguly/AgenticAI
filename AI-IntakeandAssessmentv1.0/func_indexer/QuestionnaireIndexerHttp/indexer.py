import os
import io
import base64
import logging
import json
from typing import List, Dict

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    CorsOptions,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    # Added for vector search
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField,
)
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# -----------------------------
# Environment configuration
# -----------------------------
SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "")
SEARCH_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY", "")
SEM_CONFIG_NAME = os.environ.get("AZURE_SEARCH_SEMANTIC_CONFIG")

# Storage: prefer connection string, then account URL (optionally with SAS), else MSI
AZ_STORAGE_CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZ_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")

# Azure OpenAI (embeddings) configuration
AOAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AOAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AOAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview")
AOAI_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
# default to 3072 for text-embedding-3-large
AOAI_EMBED_DIM = int(os.getenv("AZURE_OPENAI_EMBED_DIM", "3072"))


# -----------------------------
# Helpers matching existing repo
# -----------------------------

def _blob_client() -> BlobServiceClient:
    if AZ_STORAGE_CONN_STR:
        return BlobServiceClient.from_connection_string(AZ_STORAGE_CONN_STR)
    if AZ_STORAGE_ACCOUNT_URL:
        # If SAS provided inline, no credential needed
        if "?" in AZ_STORAGE_ACCOUNT_URL:
            return BlobServiceClient(account_url=AZ_STORAGE_ACCOUNT_URL)
        return BlobServiceClient(account_url=AZ_STORAGE_ACCOUNT_URL, credential=DefaultAzureCredential())
    raise RuntimeError("Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL")


def _get_aoai_client():
    """Create Azure OpenAI client using OpenAI SDK (AzureOpenAI). Returns None if not configured."""
    if not AOAI_ENDPOINT or not AOAI_API_KEY or not AOAI_EMBED_DEPLOYMENT:
        logging.warning("Azure OpenAI not fully configured; proceeding without embeddings")
        return None
    try:
        from openai import AzureOpenAI  # provided by openai>=1.x
        return AzureOpenAI(api_key=AOAI_API_KEY, api_version=AOAI_API_VERSION, azure_endpoint=AOAI_ENDPOINT)
    except Exception as e:  # pragma: no cover
        logging.exception("Failed to create AzureOpenAI client: %s", e)
        return None


def _embed_texts(texts: List[str]) -> List[List[float] | None]:
    """Return embeddings for texts via AzureOpenAI. If not available, returns [None,...]."""
    client = _get_aoai_client()
    if not client:
        return [None] * len(texts)

    vectors: List[List[float] | None] = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(model=AOAI_EMBED_DEPLOYMENT, input=batch)
            vecs = [d.embedding for d in resp.data]
            for v in vecs:
                if isinstance(v, list) and len(v) != AOAI_EMBED_DIM:
                    logging.warning("Embedding dimension %s != expected %s", len(v), AOAI_EMBED_DIM)
            vectors.extend(vecs)
        except Exception as e:  # pragma: no cover
            logging.exception("Embedding request failed: %s", e)
            vectors.extend([None] * len(batch))
    return vectors


def create_or_update_index():
    if not SEARCH_ENDPOINT or not SEARCH_INDEX:
        raise RuntimeError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX must be set")
    if not SEARCH_KEY:
        logging.warning("AZURE_SEARCH_ADMIN_KEY not set; index creation may fail without permissions")
    idx_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="appId", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="path", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunkId", type=SearchFieldDataType.String, filterable=True),
        # Vector field for semantic/vector search
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=AOAI_EMBED_DIM,
            vector_search_profile_name="v1",
        ),
    ]

    index = SearchIndex(
        name=SEARCH_INDEX,
        fields=fields,
        cors_options=CorsOptions(allowed_origins=["*"], max_age_in_seconds=60),
    )

    # Vector search config (HNSW)
    index.vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="v1", algorithm_configuration_name="hnsw")],
    )

    if SEM_CONFIG_NAME:
        index.semantic_search = SemanticSearch(configurations=[
            SemanticConfiguration(
                name=SEM_CONFIG_NAME,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ])

    try:
        # If exists, delete then create fresh (matching repo behavior)
        idx_client.get_index(SEARCH_INDEX)
        idx_client.delete_index(SEARCH_INDEX)
    except Exception:
        pass

    idx_client.create_index(index)
    logging.info("Index %s created/updated", SEARCH_INDEX)


def _download_text(container: str, blob_name: str) -> str:
    bc = _blob_client().get_blob_client(container=container, blob=blob_name)
    data = bc.download_blob().readall()
    name = blob_name.lower()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".docx"):
        from docx import Document  # lazy import
        f = io.BytesIO(data)
        doc = Document(f)
        return "\n".join([p.text for p in doc.paragraphs])
    if name.endswith(".pdf"):
        from pypdf import PdfReader  # lazy import
        f = io.BytesIO(data)
        reader = PdfReader(f)
        return "\n".join([(page.extract_text() or "") for page in reader.pages])
    if name.endswith(".xlsx"):
        # Flatten all sheets into a tab-delimited text (values_only) for indexing
        from openpyxl import load_workbook  # lazy import
        f = io.BytesIO(data)
        wb = load_workbook(f, read_only=True, data_only=True)
        lines: List[str] = []
        for ws in wb.worksheets:
            lines.append(f"# Sheet: {ws.title}")
            for row in ws.iter_rows(values_only=True):
                values = [(str(c) if c is not None else "") for c in row]
                # Collapse excessive whitespace to keep chunks balanced
                line = " \t ".join(v.strip() for v in values if v is not None)
                if line.strip():
                    lines.append(line)
        return "\n".join(lines)
    # Fallback: treat as text
    return data.decode("utf-8", errors="ignore")


def _chunk(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + size]
        chunks.append(" ".join(chunk_words))
        i += max(1, size - overlap)
    return chunks


def _safe_doc_id(app_id: str, path: str, ci: int) -> str:
    raw = f"{app_id}|{path}|{ci}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


# -----------------------------
# Public functions called by HTTP
# -----------------------------

# Batch controls: Azure Search REST cap is 1000 docs or ~16MB per request.
MAX_DOCS_PER_BATCH = int(os.getenv("SEARCH_MAX_DOCS_PER_BATCH", "500"))
# Keep some headroom below 16MB to account for HTTP and JSON overhead
MAX_BYTES_PER_BATCH = int(os.getenv("SEARCH_MAX_BYTES_PER_BATCH", str(12 * 1024 * 1024)))

def _estimate_doc_bytes(doc: Dict) -> int:
    # Estimate serialized size of a single doc
    # ensure_ascii=False keeps unicode; separators minimize overhead
    s = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    return len(s.encode("utf-8"))

def _flush_batch(search: SearchClient, batch: List[Dict]) -> Dict[str, int]:
    if not batch:
        return {"uploaded": 0, "failed": 0}
    resp = search.upload_documents(documents=batch)
    uploaded = sum(1 for r in resp if r.succeeded)
    failed = len(resp) - uploaded
    return {"uploaded": uploaded, "failed": failed}

def index_blob(app_id: str, container: str, blob_name: str) -> Dict:
    # Ensure index exists (idempotent)
    create_or_update_index()

    search = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_KEY))

    text = _download_text(container, blob_name)
    if not text or not text.strip():
        return {"blobName": blob_name, "chunks": 0, "uploaded": 0, "failed": 0}

    chunks = _chunk(text)
    vectors = _embed_texts(chunks)

    batch: List[Dict] = []
    batch_bytes = 0
    uploaded = 0
    failed = 0

    for ci, ch in enumerate(chunks):
        path = f"{container}/{blob_name}"
        doc = {
            "id": _safe_doc_id(app_id, path, ci),
            "appId": app_id,
            "title": os.path.basename(blob_name),
            "content": ch,
            "source": "blob",
            "path": path,
            "chunkId": f"{ci}",
        }
        # attach embedding if available
        vec = vectors[ci] if vectors and ci < len(vectors) else None
        if isinstance(vec, list):
            doc["contentVector"] = [float(x) for x in vec]
        size = _estimate_doc_bytes(doc)
        if (len(batch) + 1 > MAX_DOCS_PER_BATCH) or (batch_bytes + size > MAX_BYTES_PER_BATCH):
            res = _flush_batch(search, batch)
            uploaded += res["uploaded"]
            failed += res["failed"]
            batch = []
            batch_bytes = 0
        batch.append(doc)
        batch_bytes += size

    if batch:
        res = _flush_batch(search, batch)
        uploaded += res["uploaded"]
        failed += res["failed"]

    return {"blobName": blob_name, "chunks": len(chunks), "uploaded": uploaded, "failed": failed}


def index_container(app_id: str, container: str) -> Dict:
    # Ensure index exists (idempotent)
    create_or_update_index()

    search = SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX, AzureKeyCredential(SEARCH_KEY))
    bs = _blob_client().get_container_client(container)

    total_uploaded = 0
    total_failed = 0
    total_chunks = 0
    total_blobs = 0

    batch: List[Dict] = []
    batch_bytes = 0

    for blob in bs.list_blobs():
        name = blob.name
        try:
            text = _download_text(container, name)
            if not text or not text.strip():
                continue
            chunks = _chunk(text)
            total_chunks += len(chunks)
            total_blobs += 1

            vectors = _embed_texts(chunks)

            for ci, ch in enumerate(chunks):
                path = f"{container}/{name}"
                doc = {
                    "id": _safe_doc_id(app_id, path, ci),
                    "appId": app_id,
                    "title": os.path.basename(name),
                    "content": ch,
                    "source": "blob",
                    "path": path,
                    "chunkId": f"{ci}",
                }
                vec = vectors[ci] if vectors and ci < len(vectors) else None
                if isinstance(vec, list):
                    doc["contentVector"] = [float(x) for x in vec]
                size = _estimate_doc_bytes(doc)
                if (len(batch) + 1 > MAX_DOCS_PER_BATCH) or (batch_bytes + size > MAX_BYTES_PER_BATCH):
                    res = _flush_batch(search, batch)
                    total_uploaded += res["uploaded"]
                    total_failed += res["failed"]
                    batch = []
                    batch_bytes = 0
                batch.append(doc)
                batch_bytes += size
        except Exception as e:  # pragma: no cover
            logging.exception("Failed %s: %s", name, e)
            total_failed += 1

    if batch:
        res = _flush_batch(search, batch)
        total_uploaded += res["uploaded"]
        total_failed += res["failed"]

    return {"blobs": total_blobs, "chunks": total_chunks, "uploaded": total_uploaded, "failed": total_failed}
