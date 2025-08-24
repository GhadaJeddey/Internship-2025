# Relation Extraction (RE) - AxeFinance

This module provides a high-performance Relation Extraction (RE) API using the Qwen2.5-7B-Instruct transformer model. It is designed to extract structured relationships between entities from text, and is deployable as a FastAPI service with ngrok support for public endpoints.

---

## 🚀 Features
- **Transformer-based RE**: Uses Qwen2.5-7B-Instruct for accurate relation extraction.
- **FastAPI Service**: REST API for easy integration.
- **ngrok Integration**: Expose your local API securely for testing or demo purposes.
- **Customizable Schema**: Easily extend supported relation and entity types.
- **Robust Prompting**: Adapts instructions based on text length and entity context.
- **Evaluation Tools**: Includes scripts for batch processing and pretty-printing results.

---

## 📦 Installation

1. **Clone the repository** and navigate to the RE folder:
    ```sh
    git clone <your-repo-url>
    cd AxeFinance/RE
    ```

2. **Create a virtual environment** (recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

---

## ⚡ Usage

### 1. **Run the FastAPI Server**
```sh
uvicorn ReAPI:app --reload --port 8003
```
- The API will be available at `http://127.0.0.1:8003`

### 2. **Expose via ngrok (optional)**
- Set your NGROK_AUTH_TOKEN in a `.env` file or environment variable.
- The API will be accessible via a public ngrok URL.

### 3. **API Endpoints**

- `POST /RE/extract`  
  Extract relations from input text and NER entities.
  ```json
  {
    "text": "Satya Nadella is the CEO of Microsoft Corporation. He was born in Hyderabad and graduated from the University of Wisconsin. Microsoft was founded by Bill Gates.",
    "entities": {
      "PERSON": ["Satya Nadella", "Bill Gates"],
      "ORG": ["Microsoft Corporation", "Microsoft", "University of Wisconsin"],
      "LOC": ["Hyderabad"]
    }
  }
  ```
  **Response:**
  ```json
  {
    "relations": [
      {"subject": "Satya Nadella", "relation": "per:works", "object": "Microsoft Corporation"},
      {"subject": "Satya Nadella", "relation": "per:born_in", "object": "Hyderabad"},
      {"subject": "Satya Nadella", "relation": "per:education", "object": "University of Wisconsin"},
      {"subject": "Microsoft", "relation": "org:founded_by", "object": "Bill Gates"}
    ],
    "processing_time": 1.23
  }
  ```

- `GET /RE/health`  
  Health check endpoint.

- `GET /RE/info`  
  Model and schema information.

---

## 🧪 Evaluation & Batch Processing
- Use the provided `run_pipeline()` in `RE.py` for batch extraction from uploaded files (Colab or local).
- Results are saved as JSON and can be pretty-printed using the `display_relations` method.

---

## 🛠️ Configuration
- **Model selection**: Change the model name in `RE.py` if needed.
- **Relation/entity types**: Extend the `initialize_schema` method in `RE.py`.
- **ngrok**: Set your token in `.env` for public API exposure.

---

## 📝 Requirements
See [`requirements.txt`](./requirements.txt) for all dependencies.

---

## 🤝 Contributing
Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

---

## 📄 License
MIT License (or your project’s license)

---

## 📬 Contact
For questions or support, contact the AxeFinance NLP team.
