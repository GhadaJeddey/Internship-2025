# Named Entity Recognition (NER) - AxeFinance

This module provides a high-performance Named Entity Recognition (NER) API using state-of-the-art transformer models (e.g., Qwen2.5-7B-Instruct). It is designed for robust extraction of entities such as PERSON and ORG from text, and is easily deployable as a FastAPI service.

---
##  Project Structure

```
NER/
├── NER.py                  # Core NER logic and model
├── NerAPI.py               # FastAPI server exposing NER endpoints
├── NerAPI.ipynb            # For google colab 
├── evaluation_dataset.json # Dataset for evaluation
├── evaluation_results.json # Evaluation metrics/results
├── predictions_results.json# Model predictions on evaluation data
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (e.g., API keys)
├── ReadME.md               # Documentation (this file)
```

## Quick Start

### 1. **Run the FastAPI Server**
```sh
uvicorn NerAPI:app --reload
```
- The API will be available at `http://127.0.0.1:8000`

### 2. **Expose via ngrok (optional)**
- Set your NGROK_AUTH_TOKEN in a `.env` file or environment variable.
- The API will be accessible via a public ngrok URL.

### 3. **API Endpoints**

- `POST /NER/extract`  
  Extract named entities from input text.
  ```json
  {
    "text": "Apple Inc. is located in Cupertino. Tim Cook is the CEO."
  }
  ```
  **Response:**
  ```json
  {
    "entities": {
      "PERSON": ["Tim Cook"],
      "ORG": ["Apple Inc."]
    },
    "processing_time": 0.42
  }
  ```

- `GET /NER/health`  
  Health check endpoint.

- `GET /NER/info`  
  Model and service information.

---


##  Performance

| Metric         | Value   |
|----------------|---------|
| Precision      | 0.8618  |
| Recall         | 0.8339  |
| F1-score       | 0.8372  |
| Evaluation Set | 257     |


---

## 📝 Requirements

See [`requirements.txt`](./requirements.txt) for all dependencies.

---
