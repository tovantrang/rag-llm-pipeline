# FastFood-Project

## Setup Instructions
Considering you already have git and conda installed:

Linux system required
1. **Install ollama**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
2. **Install ollama models**:
```bash
ollama pull mistral
```
3. **Clone the repository**:
```bash
git clone https://github.com/tovantrang/FastFood-Project.git
cd FastFood-Project
```
4. **Create and activate a conda environment**:
```bash
conda create -p ./envs python=3.10 -y
conda activate ./envs
```
5. **Install the required packages**:
```bash
pip install pymupdf pdfplumber
pip install "marker-pdf[all]"
pip install pytesseract
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install layoutparser[layoutmodels,tesseract]
pip install sentence-transformers
pip install git+https://github.com/openai/CLIP.git
pip install faiss-cpu
pip install langchain langchain_community huggingface_hub openai
pip install streamlit
```
6. **Test the installation**:
```bash
python -c "import torch; print(f'CUDA dispo: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Aucun\"}')"
```
```bash
ollama run mistral
```

## Usage Instructions
With the pdf in a "data" folder:

1. **Ingest data**:
```bash
python 1_ingest.py
```
2. **Run the Streamlit app**:
```bash
streamlit run 2_app.py
```