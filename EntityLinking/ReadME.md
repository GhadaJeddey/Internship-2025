# Entity Linking API

A high-performance FastAPI service for linking entity mentions to Wikidata knowledge base using semantic similarity and multi-stage ranking algorithms.

## Project Structure

```
EntityLinking/
├── EntityLinking.py         # Core logic for entity linking and Wikidata API interaction
├── LinkerAPI.py             # FastAPI server exposing entity linking endpoints
├── requirements.txt         # Python dependencies
├── entity_linking_results.json  # Evaluation results and metrics
├── ReadME.md                # Documentation

```

##  Performance Metrics

| Model | Accuracy | Top-3 Accuracy | 
|-------|-----------|--------|
| all-mpnet-base-v2 | 0.7297 | 0.9189 |


## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- Internet connection for Wikidata API access
- CUDA-capable GPU (optional, for better performance)

### Setup

1. **Clone and navigate to the directory:**
```bash
git clone https://github.com/GhadaJeddey/AxeFinance.git
cd AxeFinance/EntityLinking-task
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

##  Quick Start

### 1. Start the API Server
```bash
python LinkerAPI.py
```
The API will be available at `http://localhost:8001`

### 2. API Documentation
Visit `http://localhost:8001/docs` for interactive API documentation

### 3. Basic Usage

**Health Check:**
```bash
curl http://localhost:8001/Linkerhealth
```

**Model Information:**
```bash
curl http://localhost:8001/EntityLinking/ModelInfo
```

**Link Entity:**
```bash
curl -X POST "http://localhost:8001/EntityLinking/extract" \
     -H "Content-Type: application/json" \
     -d '{
       "mention": "Apple",
       "context": "Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California."
     }'
```
## 📡 API Reference

### Endpoints

#### 1. `/EntityLinker/extract` (POST)
Main entity linking endpoint for linking mentions to Wikidata entities.

**Request Body:**
```json
{
  "mention": "string (required, entity mention to link)",
  "context": "string (required, surrounding text context)",
  "top_k": 1  // Optional, number of top candidates to return (default: 1, max: 10)
}
```

**Response:**
```json
{
  "mention": "Apple",
  "context": "Apple Inc. is a technology company founded by Steve Jobs in Cupertino, California.",
  "best_entity": {
    "qid": "Q312",
    "label": "Apple Inc.",
    "description": "American multinational technology company"
  },
  "timestamp": "2025-08-21T10:30:45.123456",
  "processing_time_seconds": 0.245
}
```

#### 2. `/EntityLinker/health` (GET)
Health check endpoint returning service status and model information.


#### 3. `/EntityLinker/info` (GET)
Detailed model information and configuration.


#### 4. `/` (GET)
Root endpoint with API metadata and available endpoints.


## Technical Details

### Entity Linking Features

The entity linking process follows a multi-stage approach:

1. **Candidate Generation**
   - Wikidata API search using mention text
   - Fuzzy string matching for variations
   - Context keyword expansion

2. **Multi-stage Scoring**
   - **Semantic Similarity**: Sentence-transformer embeddings
   - **String Similarity**: Levenshtein distance and fuzzy matching
   - **Context Compatibility**: Context window analysis

3. **Final Ranking**
   - Weighted combination of all similarity scores
   - Confidence threshold filtering
   - Top candidate selection


## References

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Wikidata API Documentation](https://www.wikidata.org/w/api.php)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Entity Linking Survey Paper](https://arxiv.org/abs/1707.02956)

- [Hugging Face](https://huggingface.co/) - Pre-trained models and transformers library

**Entity Linking Task** - Part of the AxeFinance NLP Pipeline