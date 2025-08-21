# Text Summarization API

A high-performance FastAPI-based text summarization service using Facebook's BART-large-CNN model with intelligent text chunking and progressive summarization capabilities.

## 🚀 Features

- **High-quality summarization** using Facebook's BART-large-CNN model
- **Intelligent text chunking** for long documents (paragraph-based and token-based)
- **Progressive summarization** for documents exceeding token limits
- **Configurable parameters** (length, beam search, length penalty)
- **RESTful API** with comprehensive OpenAPI documentation
- **Evaluation framework** with ROUGE and BERTScore metrics
- **Health monitoring** and model information endpoints

## 🏗️ Architecture

```text
Summarization/
    ├── Summarizer.py             # Summarization implementation
    ├── SummarizerAPI.py          # Summarization FastAPI service
    ├── LLM-Summary.py            # LLM that generated reference summaries for the evaluation articles
    ├── articles.md               # Evaluation articles
    ├── summarization_results.json # Summarization results
    ├── summarization_result6s.json # evaluation results 
    ├── requirements.txt 
    └── ReadME.md
```
### Core Components

1. **Summarizer Class** (`Summarizer.py`)
   - BART model integration
   - Text chunking algorithms
   - Progressive summarization logic
   - Evaluation framework

2. **FastAPI Service** (`SummarizerAPI.py`)
   - RESTful endpoints
   - Request/response validation
   - Error handling and logging
   - Health checks

3. **Evaluation Dataset** (`articles.md`)
   - 20+ curated articles with reference summaries
   - Performance benchmarking data

## 📊 Performance Metrics

Based on evaluation with 20 articles:

- **Average ROUGE-1 F1**: 0.416
- **Average ROUGE-2 F1**: 0.154  
- **Average ROUGE-L F1**: 0.270
- **Average BERTScore F1**: 0.888


## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- 10GB+ storage for models
- CUDA-capable GPU (optional, for better performance)

### Setup

1. **Clone and navigate to the directory:**
```bash
git clone https://github.com/GhadaJeddey/AxeFinance.git
cd AxeFinance/Summarization-task
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

## 🚀 Quick Start

### 1. Start the API Server
```bash
python SummarizerAPI.py
```
The API will be available at `http://localhost:8000`

### 2. API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation

### 3. Basic Usage

**Health Check:**
```bash
curl http://localhost:8000/Summarizer/health
```

**Model Information:**
```bash
curl http://localhost:8000/Summarizer/info
```

**Summarize Text:**
```bash
curl -X POST "http://localhost:8000/Summarizer/summarize" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your article text here...",
     }'
```

## 📡 API Reference

### Endpoints

#### 1. `/Summarizer/summarize` (POST)
Main summarization endpoint with configurable parameters.

**Request Body:**
```json
{
  "text": "string (required, min 50 chars)",
}
```

**Response:**
```json
{
  "summary": "Generated summary text..."
}
```

#### 2. `/Summarizer/health` (GET)
Health check endpoint returning service status.

**Response:**
```json
{
  "status": "healthy",
  "model": "facebook/bart-large-cnn"
}
```

#### 3. `/Summarizer/info` (GET)
Detailed model information and configuration.

**Response:**
```json
{
  "model_name": "facebook/bart-large-cnn",
  "model_type": "BART",
  "task": "text-summarization",
  "max_input_length": 1024,
  "max_output_length": 500,
  "min_output_length": 100,
  "num_beams": 2,
  "length_penalty": 2.0,
  "device": "cuda:0"
}
```

## 🧠 Technical Details

### Text Processing Logic

1. **Short Text (≤1024 tokens)**: Direct summarization
2. **Long Text (>1024 tokens)**: Intelligent chunking
   - **Primary**: Paragraph-based chunking (semantic boundaries)
   - **Fallback**: Token-based chunking with overlap (400 tokens, 100 overlap)
3. **Progressive Summarization**: Previous summary + current chunk → new summary


## 🧪 Evaluation

### Run Evaluation
```python
from Summarizer import Summarizer

# Initialize summarizer
summarizer = Summarizer()

# Evaluate on all articles
results = summarizer.evaluate_on_all_articles(
    md_path="articles.md",
    output_path="summarization_results.json"
)

# Print results
print(f"Average ROUGE-1 F1: {results[0]:.4f}")
print(f"Average ROUGE-2 F1: {results[1]:.4f}")
print(f"Average ROUGE-L F1: {results[2]:.4f}")
print(f"Average BERTScore F1: {results[3]:.4f}")
```

### Custom Evaluation
```python
# Evaluate single summary
rouge_score, bert_score = summarizer.evaluation(
    reference_summary="Reference text...",
    summary="Generated summary..."
)
```

## 🙏 Acknowledgments

- [Facebook BART](https://arxiv.org/abs/1910.13461) - Base summarization model
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Model implementation
- [LangChain](https://github.com/hwchase17/langchain) - Text processing utilities
- [FastAPI](https://github.com/tiangolo/fastapi) - Web framework


---

**Summarization Task** - Part of the AxeFinance NLP Pipeline
