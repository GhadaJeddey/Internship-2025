# AxeFinance - Multi-Modal NLP Pipeline

A comprehensive Natural Language Processing pipeline for financial text analysis, featuring Named Entity Recognition (NER), Entity Linking, and Text Summarization with FastAPI-based microservices architecture.

## Features

### 1. Text Summarization
- **Model**: facebook/bart-large-cnn
- **Capabilities**:
  - Progressive summarization for long texts
  - Paragraph-based and token-based chunking
  - Configurable length and quality parameters
  - ROUGE and BERTScore evaluation

### 2. Named Entity Recognition (NER)
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Entities**: PERSON, ORG, LOC, WORK_OF_ART, NATIONALITIES_RELIGIOUS_GROUPS, DATE, TIME, MONEY, PERCENT, LAW
- **Capabilities**: 
  - Text chunking for long documents
  - Post-processing for entity validation
  - Comprehensive evaluation metrics
  - Confusion matrix generation

### 3. Entity Linking
- **Knowledge Base**: Wikidata
- **Model**: sentence-transformers/all-mpnet-base-v2
- **Features**:
  - Multi-strategy candidate search
  - Semantic similarity scoring
  - String matching algorithms
  - Context compatibility assessment
  - Comprehensive evaluation framework

### 4. Relation Extraction (RE)
- **Model**: Custom or transformer-based RE model
- **Capabilities**:
  - Extracts semantic relationships between recognized entities
  - Outputs head, tail, and relation type for each detected relationship
  - Integrates with Neo4j for live graph updates

### 5. Neo4j Graph Integration
- **Component**: `Neo4jGraphManager` (see `neo4j_graph.py`)
- **Features**:
  - Live update of a Neo4j graph database with extracted entities and relations
  - Fetch, add, and search nodes and relationships
  - Visualize graph structure in the Streamlit dashboard
  - Credentials and connection managed via the dashboard sidebar

## Architecture

```
AxeFinance/
├── app.py                       # Streamlit dashboard orchestrator
├── NER-task/
    ├── NER.py                  # Core NER logic and model
    ├── NerAPI.py               # FastAPI server exposing NER endpoints
    ├── NerAPI.ipynb            # For google colab 
    ├── evaluation_dataset.json # Dataset for evaluation
    ├── evaluation_results.json # Evaluation metrics/results
    ├── predictions_results.json# Model predictions on evaluation data
    ├── requirements.txt        # Python dependencies
    ├── .env                    # Environment variables (e.g., API keys)
    ├── ReadME.md               # Documentation 
├── EntityLinking-task/
    ├── EntityLinking.py         # Core logic for entity linking and Wikidata API interaction
    ├── LinkerAPI.py             # FastAPI server exposing entity linking endpoints
    ├── requirements.txt         # Python dependencies
    ├── entity_linking_results.json  # Evaluation results and metrics
    ├── ReadME.md                # Documentation
├── Summarization-task/
    ├── Summarizer.py             # Summarization implementation
    ├── SummarizerAPI.py          # Summarization FastAPI service
    ├── LLM-Summary.py            # LLM that generated reference summaries for the evaluation articles
    ├── articles.md               # Evaluation articles
    ├── summarization_results.json # Summarization results
    ├── summarization_result6s.json # evaluation results 
    ├── requirements.txt 
    └── ReadME.md
├── RelationExtraction-task/
    ├── RE.py                  # Core RE logic and model
    ├── ReAPI.py               # FastAPI server exposing RE endpoints
    ├── ReAPI.ipynb            # For Google Colab
    ├── evaluation_dataset.json # Dataset for evaluation
    ├── evaluation_results.json # Evaluation metrics/results
    ├── predictions_results.json# Model predictions on evaluation data
    ├── requirements.txt        # Python dependencies
    ├── .env                    # Environment variables (e.g., API keys)
    ├── README.md               # Documentation 
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 20GB+ storage for models

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/GhadaJeddey/AxeFinance.git
cd AxeFinance
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Start NER Service
```bash
cd NER-task
python NerAPI.py
# Service available at http://localhost:8001
```

### 2. Start Entity Linking Service
```bash
cd EntityLinking-task
python LinkerAPI.py
# Service available at http://localhost:8002
```

### 3. Start Summarization Service
```bash
cd Summarization-task
python SummarizerAPI.py
# Service available at http://localhost:8000
```

### 4. Start Relation Extraction Service
```bash
cd RelationExtraction-task
python ReAPI.py
# Service available at http://localhost:8003
```

### 5. Run the Streamlit Dashboard
```bash
streamlit run app.py
# Dashboard available at http://localhost:8501
```
### 6. Neo4j Graph Integration
The dashboard allows you to connect to a Neo4j Aura instance. Enter your credentials in the sidebar, and the graph will be updated live with extracted entities and relations after each relation extraction job.


## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Wikidata](https://www.wikidata.org/)
- [BART](https://arxiv.org/abs/1910.13461)
- [Qwen](https://github.com/QwenLM/Qwen)

---

**AxeFinance** - Empowering financial text analysis with state-of-the-art NLP technologies.
