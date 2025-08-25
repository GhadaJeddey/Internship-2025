# AxeFinance - Multi-Modal NLP Pipeline

A comprehensive Natural Language Processing pipeline for financial text analysis, featuring Named Entity Recognition (NER), Entity Linking, and Text Summarization with FastAPI-based microservices architecture.

## 🚀 Features

### 1. Named Entity Recognition (NER)
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Entities**: PERSON, ORG, LOC, WORK_OF_ART, NATIONALITIES_RELIGIOUS_GROUPS, DATE, TIME, MONEY, PERCENT, LAW
- **Capabilities**: 
  - Text chunking for long documents
  - Post-processing for entity validation
  - Comprehensive evaluation metrics
  - Confusion matrix generation

### 2. Entity Linking
- **Knowledge Base**: Wikidata
- **Model**: sentence-transformers/all-mpnet-base-v2
- **Features**:
  - Multi-strategy candidate search
  - Semantic similarity scoring
  - String matching algorithms
  - Context compatibility assessment
  - Comprehensive evaluation framework

### 3. Text Summarization
- **Model**: facebook/bart-large-cnn
- **Capabilities**:
  - Progressive summarization for long texts
  - Paragraph-based and token-based chunking
  - Configurable length and quality parameters
  - ROUGE and BERTScore evaluation

## 🏗️ Architecture

```
AxeFinance/
├── NER-task/
│   ├── NER.py                    # NER implementation
│   ├── NerAPI.py                 # NER FastAPI service
│   ├── evaluation_dataset.json  # NER evaluation data
│   └── evaluation_results.json  # NER results
├── EntityLinking-task/
│   ├── EntityLinking.py          # Entity linking implementation
│   ├── LinkerAPI.py              # Entity linking FastAPI service
│   ├── evaluation.json           # Entity linking evaluation data
│   └── entity_linking_results.json # Entity linking results
├── Summarization-task/
│   ├── Summarizer.py             # Summarization implementation
│   ├── SummarizerAPI.py          # Summarization FastAPI service
│   ├── articles.md               # Evaluation articles
│   └── summarization_results.json # Summarization results
└── RelationExtraction-task/
    ├── RE.py                     # Relation extraction
    ├── neo4j.py                  # Neo4j integration
    └── Neo4jGraph.py             # Graph database operations
```

## 📊 Performance Metrics

### NER Performance
- **Global Metrics**: Precision: 0.7473, Recall: 0.7313, F1: 0.7171
- **Best Performing Entities**: 
  - PERSON: F1 = 0.8475
  - PERCENT: F1 = 0.9565
  - MONEY: F1 = 0.8929

### Entity Linking Performance
- **Accuracy**: 54.3% (19/35 valid samples)
- **Average Confidence**: 0.642
- **Model**: sentence-transformers/all-mpnet-base-v2

### Summarization Performance
- **Average ROUGE-1 F1**: 0.416
- **Average ROUGE-2 F1**: 0.154
- **Average ROUGE-L F1**: 0.270
- **Average BERTScore F1**: 0.888

## 🛠️ Installation

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

## 🚀 Quick Start

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

## 📡 API Usage

### NER API

**Extract entities:**
```bash
curl -X POST "http://localhost:8001/NER/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
  }'
```

**Response:**
```json
{
  "entities": {
    "ORG": ["Apple Inc."],
    "PERSON": ["Steve Jobs"],
    "LOC": ["Cupertino", "California"],
    "DATE": ["1976"]
  },
  "processing_time": 2.34
}
```

### Entity Linking API

**Link entities:**
```bash
curl -X POST "http://localhost:8002/EntityLinking/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "mention": "Apple",
    "context": "Apple Inc. is a technology company based in Cupertino, California."
  }'
```

**Response:**
```json
{
  "mention": "Apple",
  "context": "Apple Inc. is a technology company...",
  "best_entity": {
    "qid": "Q312",
    "label": "Apple Inc.",
    "description": "American multinational technology company",
    "score": 0.95,
    "semantic_score": 0.92,
    "string_score": 0.98,
    "context_score": 0.85
  }
}
```

### Summarization API

**Summarize text:**
```bash
curl -X POST "http://localhost:8000/Summarizer/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long article text here...",
    "max_length": 400,
    "min_length": 80
  }'
```

**Response:**
```json
{
  "summary": "Generated summary text..."
}
```

## 🧪 Evaluation

### Run NER Evaluation
```python
from NER import NERExtractor

extractor = NERExtractor()
results = extractor.evaluation("evaluation_dataset.json")
```

### Run Entity Linking Evaluation
```python
from EntityLinking import EntityLinker

linker = EntityLinker()
results = linker.evaluation(data, output_file="results.json")
```

### Run Summarization Evaluation
```python
from Summarizer import Summarizer

summarizer = Summarizer()
results = summarizer.evaluate_on_all_articles()
```

## 🔧 Configuration

### Model Configuration
- **NER Model**: Change in `NERExtractor.__init__()`
- **Entity Linking Model**: Modify `EntityLinker.__init__()`
- **Summarization Model**: Update `Summarizer.__init__()`

### API Configuration
- **Ports**: Modify in respective API files
- **CORS**: Configure in FastAPI apps
- **Rate Limiting**: Add middleware as needed

## 📈 Monitoring

### Health Checks
```bash
# NER Service
curl http://localhost:8001/NER/health

# Entity Linking Service  
curl http://localhost:8002/Linkerhealth

# Summarization Service
curl http://localhost:8000/Summarizer/health
```

### Model Information
```bash
# Get model details
curl http://localhost:8001/NER/ModelInfo
curl http://localhost:8002/EntityLinking/ModelInfo
curl http://localhost:8000/Summarizer/info
```

## 🐳 Docker Deployment

### Build Images
```bash
# Build all services
docker-compose build

# Or build individually
docker build -t axefinance-ner ./NER-task
docker build -t axefinance-linker ./EntityLinking-task
docker build -t axefinance-summarizer ./Summarization-task
```

### Run Services
```bash
# Start all services
docker-compose up

# Or run individually
docker run -p 8001:8001 axefinance-ner
docker run -p 8002:8002 axefinance-linker
docker run -p 8000:8000 axefinance-summarizer
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch sizes
   - Use CPU-only models
   - Increase GPU memory

2. **Model Download Failures**
   - Check internet connection
   - Verify Hugging Face access
   - Clear cache and retry

3. **API Connection Issues**
   - Check port availability
   - Verify firewall settings
   - Ensure services are running

### Performance Optimization

1. **GPU Acceleration**
   - Install CUDA toolkit
   - Use GPU-optimized models
   - Configure device mapping

2. **Memory Management**
   - Monitor RAM usage
   - Use model quantization
   - Implement batch processing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/GhadaJeddey/AxeFinance/issues)
- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Email**: [contact@axefinance.com](mailto:contact@axefinance.com)

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FastAPI](https://github.com/tiangolo/fastapi)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [Wikidata](https://www.wikidata.org/)
- [BART](https://arxiv.org/abs/1910.13461)
- [Qwen](https://github.com/QwenLM/Qwen)

---

**AxeFinance** - Empowering financial text analysis with state-of-the-art NLP technologies.
