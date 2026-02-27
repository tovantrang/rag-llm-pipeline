# RAG-LLM-pipeline — Local RAG over PDF

A local Retrieval-Augmented Generation (RAG) application that:
- parses PDF documents (layout-aware + OCR),
- builds a FAISS vector index,
- retrieves relevant chunks using LangChain,
- generates answers with a local LLM (Ollama),
- provides an interactive Streamlit interface.

## Environment

Python 3.10 recommended.


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

-   Starts an interactive CLI session.
-   If the model is not downloaded, it will automatically download it
    first.
-   Exit with `Ctrl + D` or `/exit`.

### Local API

Ollama exposes a local API by default at:

    http://127.0.0.1:11434

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
With the PDF files in a "data" folder:

1. **Ingest data**:
```bash
python 1_ingest.py
```
2. **Run the Streamlit app**:
```bash
streamlit run 2_app.py
```