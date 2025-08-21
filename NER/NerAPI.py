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
from NER import NERExtractor , NEREntities 
load_dotenv(".env")

auth_token = os.environ.get("NGROK_AUTH_TOKEN")

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ner_extractor = NERExtractor()

app = FastAPI(
    title="Named Entity Recognition API",
    description="A high-performance NER service using Qwen2.5-7B-Instruct",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze for named entities")

class NERResponse(BaseModel):
    entities: Dict[str, List[str]]
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Named Entity Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/NERhealth"
    }

@app.get("/NER/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if ner_extractor is not None else "unhealthy",
        model_loaded=ner_extractor is not None,
        timestamp=datetime.datetime.now().isoformat()
    )

@app.post("/NER/extract", response_model=NERResponse)
async def extract_entities(request: NERRequest):
    """Extract named entities from text"""
    if ner_extractor is None:
        raise HTTPException(
            status_code=503,
            detail="NER model is not loaded. Please try again later."
        )

    try:
        start_time = time.time()
        entities_raw = ner_extractor.ner_predict(request.text)

        if isinstance(entities_raw, str):
            entities = json.loads(entities_raw)
        else:
            entities = entities_raw
        processing_time = time.time() - start_time

        return NERResponse(
            entities=entities,
            processing_time=round(processing_time, 3),
        )

    except Exception as e:
        logger.error(f"Error processing NER request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text: {str(e)}"
        )

@app.get("/NER/info")
async def get_ner_model_info():
    """Get information about the loaded model"""
    if ner_extractor is None:
        raise HTTPException(
            status_code=503,
            detail="NER model is not loaded"
        )

    return {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "entity_types": list(NEREntities.__fields__.keys())
    }

def run_server():
    """Run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")

def start_server_with_ngrok():
    """Start server with ngrok tunnel for Colab"""
    ngrok.set_auth_token(auth_token)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    time.sleep(3)


    public_url = ngrok.connect(8002)

    print("FastAPI Server Started!")
    print(f"Public URL: {public_url}")
    print(f"API Documentation: {public_url}/docs")
    print(f"Health Check: {public_url}/NER/health")
    print(f"Extract Entities: {public_url}/NER/extract")

    return public_url

if __name__ == "__main__":
    
    colab = True  # Change to false if no need for a public api (not using colab)
    if colab :
        public_url = start_server_with_ngrok()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nServer stopped")
            ngrok.disconnect(public_url)
    
    else : 
        run_server()