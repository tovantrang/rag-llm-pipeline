# RAG-LLM-pipeline — Local RAG over PDF

Designed to explore retrieval robustness on semi-structured administrative PDF documents.

A local Retrieval-Augmented Generation (RAG) application that:
- parses PDF documents (layout-aware + OCR),
- builds a FAISS vector index,
- retrieves relevant chunks using LangChain,
- generates answers with a local LLM (Ollama),
- provides an interactive Streamlit interface.

## How It Works

1. PDF documents are parsed using Marker (layout-aware parsing + OCR when required).
2. Extracted content is converted to Markdown.
3. Markdown is split into layout-aware chunks.
4. Chunks are embedded using a multilingual embedding model.
5. Embeddings are indexed in FAISS for dense vector retrieval.
6. Complex queries are decomposed into multiple sub-queries to reduce noise.
7. Retrieved chunks are re-ranked using a Cross-Encoder model to improve precision.
8. The selected context is passed to a local LLM for answer generation.
9. A Streamlit interface provides interactive querying.

- Embedding model: `paraphrase-multilingual-mpnet-base-v2`
- Cross-Encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM: `mistral`

## Current Limitations

- Pure dense vector retrieval (FAISS) without hybrid sparse search.
- Tables are serialized as Markdown and treated as plain text.
- No structured indexing of key administrative fields (e.g., visa categories, tuition fees).
- Images extracted from PDFs are not embedded or referenced.
- No formal evaluation benchmark was implemented.
- No metadata filtering or hierarchical retrieval in the current version.

## Future Directions

- Hybrid retrieval (BM25 + dense search).
- Structured extraction of tables into a JSON schema.
- Parent-child hierarchical chunking.
- Multimodal extension with image embeddings.
- Evaluation dataset with recall/precision analysis.
- Metadata-aware filtering (document type, topic).
- Numeric consistency checking for administrative values.

## Environment

Python 3.10–3.11 recommended. Run the commands below from the repository root.

### Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

### Embedding device

`config.py` sets `DEVICE = "cuda"` by default. The embedding and reranking models
therefore require a CUDA-capable NVIDIA GPU and a working CUDA-enabled PyTorch
installation. The pinned `requirements.txt` includes CUDA 12 PyTorch packages;
make sure your NVIDIA driver supports them. On a machine without a compatible
GPU/CUDA setup, change `DEVICE` in `config.py` to `"cpu"` before ingesting data
(CPU processing will be slower).

## Ollama Setup

This project uses a local LLM served by **Ollama**.

**Tested on Linux.**\
Ollama is also officially available on Windows and macOS, but these
platforms were not tested.

### Install Ollama

Follow the official installation instructions:\
https://docs.ollama.com/

### Download a model

``` bash
ollama pull mistral
```

This downloads the model locally (no execution yet).

### Run the model (interactive test)

``` bash
ollama run mistral
```

- Starts an interactive CLI session.
- If the model is not downloaded, it will automatically download it first.
- Exit with `Ctrl + D` or `/exit`.

### Local API

Ollama exposes a local API by default at:

`http://127.0.0.1:11434`

The Streamlit app connects to this local API.

## Linux Service Management (optional)

On Linux, Ollama may run as a systemd service.

Check status:

``` bash
sudo systemctl status ollama
```

Stop:

``` bash
sudo systemctl stop ollama
```

Start:

``` bash
sudo systemctl start ollama
```

Restart:

``` bash
sudo systemctl restart ollama
```

## Usage Instructions

Before starting, install the Python dependencies above and make sure the Ollama
server is running with the `mistral` model available (see Ollama Setup). Put the
PDF files to ingest in the `data/` folder; the ingestion script writes processed
documents and the vector store to the configured output folders.

1. **Ingest the PDFs** (run once, and again when the source PDFs change):

```bash
python 1_ingest.py
```

2. **Start the Streamlit app**:

```bash
streamlit run 2_app.py
```
