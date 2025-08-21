from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from Summarizer import Summarizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Summarization API",
    description="API for text summarization using BART model",
    version="1.0.0"
)

summarizer = Summarizer() 
logger.info("Model loaded successfully!")

class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Text to be summarized", min_length=50)
    max_length: Optional[int] = Field(500, description="Maximum length of the summary", ge=50, le=1000)
    min_length: Optional[int] = Field(100, description="Minimum length of the summary", ge=20, le=500)
    num_beams: Optional[int] = Field(2, description="Number of beams for beam search", ge=1, le=10)
    length_penalty: Optional[float] = Field(2.0, description="Length penalty for the summary", ge=0.1, le=5.0)

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/Summarizer/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    - **text**: The input text to be summarized (minimum 50 characters)
    - **max_length**: Maximum length of the generated summary (50-1000)
    - **min_length**: Minimum length of the generated summary (20-500)
    - **num_beams**: Number of beams for beam search (1-10)
    - **length_penalty**: Length penalty for controlling summary length (0.1-5.0)
    
    """
    try:
        logger.info(f"Received summarization request for text of length: {len(request.text)}")
        
        if request.min_length >= request.max_length:
            raise HTTPException(
                status_code=400, 
                detail="min_length must be less than max_length"
            )
        
        summary = summarizer.summarize_text(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty
        )
        
       
        logger.info(f"Summary generated successfully.")
        
        return SummarizeResponse(
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.get("/Summarizer/health")
async def summarizer_health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "facebook/bart-large-cnn"}


@app.get("/Summarizer/info")
async def get_summarizer_model_info():
    """Get information about the loaded model"""
    return {
        "model_name": str(summarizer.model_name),
        "model_type": "BART",
        "task": "text-summarization",
        "max_input_length": 1024,
        "max_output_length": 500,
        "min_output_length": 100,
        "num_beams": 2,
        "length_penalty": 2.0,
        "device": str(summarizer.device)
    }

@app.get('/')
async def root():
    return {
            "message": "Summarization API",
            "version": "1.0.0",
            "port" : "8000",
            "endpoints": {
                "health": "/Summarizer/health",
                "model_info": "/Summarizer/info",
                "summarize": "/Summarizer/summarize"
            },
            "documentation": "/docs" 
        }


if __name__ == "__main__":
    uvicorn.run(
        "SummarizerAPI:app", 
        host="0.0.0.0",
        port=8000,
        reload=True
    )