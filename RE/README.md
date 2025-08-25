# Relation Extraction (RE) - AxeFinance

This module provides a high-performance Relation Extraction (RE) API using state-of-the-art transformer models (e.g., Qwen2.5-7B-Instruct). It is designed for robust extraction of relationships between entities in text, and is easily deployable as a FastAPI service.

---

## Project Structure

```
RE/
├── RE.py                  # Core RE logic and model
├── ReAPI.py               # FastAPI server exposing RE endpoints
├── ReAPI.ipynb            # For Google Colab
├── evaluation_dataset.json # Dataset for evaluation
├── evaluation_results.json # Evaluation metrics/results
├── predictions_results.json# Model predictions on evaluation data
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (e.g., API keys)
├── README.md               # Documentation (this file)
```

## Quick Start

### 1. **Run the FastAPI Server**
```sh
uvicorn ReAPI:app --reload
```
- The API will be available at `http://127.0.0.1:8000`

### 2. **Expose via ngrok (optional)**
- Set your NGROK_AUTH_TOKEN in a `.env` file or environment variable.
- The API will be accessible via a public ngrok URL.

### 3. **API Endpoints**

- `POST /RE/extract`  
  Extract relations between entities from input text and provided entities.
  ```json
  {
    "text": "Tim Cook is the CEO of Apple Inc.",
    "entities": {
      "PERSON": ["Tim Cook"],
      "ORG": ["Apple Inc."]
    }
  }
  ```
  **Response:**
  ```json
  {
    "relations": [
      {
        "subject": "Tim Cook",
        "relation": "CEO_of",
        "object": "Apple Inc."
      }
    ]
  }
  ```

- `GET /RE/health`  
  Health check endpoint.

- `GET /RE/info`  
  Model and service information.

---

## Performance

| Metric         | Value   |
|----------------|---------|
| Precision      | 0.929   |
| Recall         | 0.722   |
| F1-score       | 0.813   |

*Note: Replace with your actual evaluation results if available.*

---

## Requirements

See [`requirements.txt`](./requirements.txt) for all dependencies.