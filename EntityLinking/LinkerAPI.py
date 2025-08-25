from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
from EntityLinking import EntityLinker
import uvicorn ,sys , json , os

linker = None
model_loaded_at = None

def initialize_model():
    """Initialize the EntityLinker model"""
    global linker, model_loaded_at
    try:
        if linker is None:
            print("Loading EntityLinker model...")
            linker = EntityLinker("sentence-transformers/all-mpnet-base-v2")
            model_loaded_at = datetime.now().isoformat()
            print("EntityLinker model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading EntityLinker model: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    print("Starting up Entity Linking API...")
    initialize_model()
    yield
    print("Shutting down Entity Linking API...")

app = FastAPI(
    title="Entity Linking API",
    description="API for linking entities to Wikidata using semantic similarity",
    version="1.0.0",
    lifespan=lifespan
)

class EntityLinkingRequest(BaseModel):
    mention: str = Field(..., description="The entity mention to link")
    context: str = Field(..., description="The context surrounding the mention")
    top_k: Optional[int] = Field(default=1, description="Number of top candidates to return", ge=1, le=10)

class EntityResult(BaseModel):
    qid: str = Field(..., description="Wikidata QID")
    label: str = Field(..., description="Entity label")
    description: Optional[str] = Field(None, description="Entity description")

class EntityLinkingResponse(BaseModel):
    mention: str = Field(..., description="The original mention")
    context: str = Field(..., description="The original context")
    best_entity: Optional[EntityResult] = Field(None, description="The best linked entity")
    timestamp: str = Field(..., description="Processing timestamp")
    processing_time_seconds: float = Field(..., description="Time taken to process the request")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., description="Name of the sentence transformer model")
    status: str = Field(..., description="Model status")
    loaded_at: str = Field(..., description="Timestamp when model was loaded")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    timestamp: str = Field(..., description="Current timestamp")
    model : str = Field(..., description="Model name")
    version: str = Field(..., description="API version")

@app.get("/EntityLinker/health", response_model=HealthResponse)
async def linker_health():
    """
    Health check endpoint for the Entity Linking API
    """
    return HealthResponse(
        status="healthy" if linker is not None else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model = "sentence-transformers/all-mpnet-base-v2",
        version="1.0.0"
    )

@app.get("/EntityLinker/info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded model
    """
    if linker is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name=linker.model_name,
        status="loaded",
        loaded_at=model_loaded_at or "unknown"
    )


@app.post("/EntityLinker/extract", response_model=EntityLinkingResponse)
async def extract_entity(request: EntityLinkingRequest):
    """
    Extract and link entities from the given mention and context
    Returns the best entity result (top_results[0])
    """
    start_time = datetime.now()

    if linker is None:
        raise HTTPException(status_code=503, detail="Entity Linking model not loaded")

    try:
        if not request.mention.strip():
            raise HTTPException(status_code=400, detail="Mention cannot be empty")
        if not request.context.strip():
            raise HTTPException(status_code=400, detail="Context cannot be empty")

        print(f"Processing mention: '{request.mention}' with context length: {len(request.context)}")

        candidates = linker.candidate_search(request.mention, request.context)
        print(f"Found {len(candidates)} candidates for '{request.mention}'")

        if not candidates:
            print(f"No candidates found for mention: '{request.mention}'")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            return EntityLinkingResponse(
                mention=request.mention,
                context=request.context,
                best_entity=None,
                timestamp=end_time.isoformat(),
                processing_time_seconds=processing_time
            )

        top_results, best_result = linker.link_entities(
            candidates,
            request.mention,
            request.context,
            top_k=request.top_k
        )
        
        print(f"Link entities returned {len(top_results)} results")
        print(f"Best result: {best_result}")

        # Fix: If best_result is None but top_results is not empty, use top_results[0]
        if not best_result and top_results:
            best_result = top_results[0]
            print(f"Using first result as best: {best_result}")

        best_entity = None
        if best_result:
            best_entity = EntityResult(
                qid=best_result["qid"],
                label=best_result["label"],
                description=best_result.get("description", "")
            )
            print(f"Created best_entity: {best_entity}")
        else:
            print("No best_result found - returning null")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return EntityLinkingResponse(
            mention=request.mention,
            context=request.context,
            best_entity=best_entity,
            timestamp=end_time.isoformat(),
            processing_time_seconds=processing_time
        )

    except Exception as e:
        print(f"Error during entity linking: {e}")
        raise HTTPException(status_code=500, detail=f"Entity linking failed: {str(e)}")


@app.get('/')
async def root():
    return {
            
            "message": "Entity Linking API",
            "version": "1.0.0",
            "port" : "8001",
            "endpoints": {
                "health": "/EntityLinker/health",
                "model_info": "/EntityLinker/info",
                "extract": "/EntityLinker/extract"
            },
            "documentation": "/docs"
            
        }

    
if __name__ == "__main__":
    print("Starting Entity Linking API...")
    uvicorn.run(
        "LinkerAPI:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
