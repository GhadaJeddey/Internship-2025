# Named Entity Recognition (NER) - AxeFinance

This module provides a high-performance Named Entity Recognition (NER) API using state-of-the-art transformer models (e.g., Qwen2.5-7B-Instruct). It is designed for robust extraction of entities such as PERSON and ORG from text, and is easily deployable as a FastAPI service.

---

## 🚀 Features

- **Transformer-based NER**: Utilizes large language models for accurate entity extraction.
- **FastAPI Service**: Easy-to-use REST API for integration with other systems.
- **CORS Enabled**: Ready for web and cross-origin requests.
- **ngrok Integration**: Expose your local API securely for testing or demo purposes.
- **Customizable**: Easily switch models or extend entity types.
- **Evaluation Tools**: Includes scripts for evaluating and visualizing NER performance.

---

## 📦 Installation

1. **Clone the repository** and navigate to the NER folder:
    ```sh
    git clone <your-repo-url>
    cd AxeFinance/NER
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

## 🧪 Evaluation

- Use the provided scripts to evaluate model predictions and print results in tabular format.
- Example:  
  ```sh
  python pretty.py
  ```

---

## 🛠️ Configuration

- **Model selection**: Change the `model_name` in `NerAPI.py` or `NER.py`.
- **Entity types**: Extend the `NEREntities` class in `NER.py` to add more entity categories.

---

## 📝 Requirements

See [`requirements.txt`](./requirements.txt) for all dependencies.

---

## 🤝 Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

