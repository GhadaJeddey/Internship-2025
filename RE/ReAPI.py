from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import time
import datetime
import nest_asyncio
import threading
from pyngrok import ngrok
import uvicorn
import os
from dotenv import load_dotenv
from RE import RelationExtractor
import json

load_dotenv(".env")
auth_token = os.environ.get("NGROK_AUTH_TOKEN")

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

extractor = RelationExtractor()

app = FastAPI(
    title="Relation Extraction API",
    description="A high-performance RE service using Qwen2.5-7B-Instruct",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RERequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze for relations")
    entities: Dict[str, List[str]] = Field(..., description="NER entities for the text")

class REResponse(BaseModel):
    relations: List[Dict[str, Any]]
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Relation Extraction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/RE/health"
    }

@app.get("/RE/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if extractor is not None else "unhealthy",
        model_loaded=extractor is not None,
        timestamp=datetime.datetime.now().isoformat()
    )

@app.post("/RE/extract", response_model=REResponse)
async def extract_relations(request: RERequest):
    if extractor is None:
        raise HTTPException(
            status_code=503,
            detail="RE model is not loaded. Please try again later."
        )
    try:
        start_time = time.time()
        result = extractor.extract_relations(request.text, request.entities)
        processing_time = time.time() - start_time
        return REResponse(
            relations=result["relations"],
            processing_time=round(processing_time, 3),
        )
    except Exception as e:
        logger.error(f"Error processing RE request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text: {str(e)}"
        )

@app.get("/RE/info")
async def get_re_model_info():
    if extractor is None:
        raise HTTPException(
            status_code=503,
            detail="RE model is not loaded"
        )
    return {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "supported_relations": extractor.all_relations,
        "supported_entity_types": list(extractor.supported_entity_types)
    }

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")

def start_server_with_ngrok():
    ngrok.set_auth_token(auth_token)
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(3)
    public_url = ngrok.connect(8003)
    print("FastAPI Server Started!")
    print(f"Public URL: {public_url}")
    print(f"API Documentation: {public_url}/docs")
    print(f"Health Check: {public_url}/RE/health")
    print(f"Extract Relations: {public_url}/RE/extract")
    return public_url

if __name__ == "__main__":
    colab = True  # Change to False if no need for a public API (not using colab)
    if colab:
        public_url = start_server_with_ngrok()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped")
            ngrok.disconnect(public_url)
    else:
        run_server()
