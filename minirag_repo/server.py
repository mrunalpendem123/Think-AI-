
import os
import shutil
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import MiniRAG from the local repository
from minirag.minirag import MiniRAG, QueryParam

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
# We assume Ollama is running locally for the embeddings and generation
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = "llama3" # Default, will be updated by request

# --- Adapter Functions for MiniRAG ---

async def ollama_embedding_func(texts: list[str]) -> np.ndarray:
    """
    Computes embeddings using the local Ollama instance.
    MiniRAG expects a numpy array of shape (n_texts, embedding_dim).
    """
    embeddings = []
    for text in texts:
        try:
            # We use 'nomic-embed-text' or similar if available, or fall back to llama3
            # It is recommended to pull a dedicated embedding model: `ollama pull nomic-embed-text`
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
                timeout=5 
            )
            if response.status_code != 200:
                # Fallback to the main model
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": OLLAMA_MODEL, "prompt": text},
                    timeout=5
                )
            
            response.raise_for_status()
            data = response.json()
            if "embedding" in data:
                embeddings.append(data["embedding"])
            else:
                # Failure case: returning zero vector (not ideal but prevents crash)
                embeddings.append([0.0] * 768) 
        except Exception as e:
            print(f"Embedding failed: {e}")
            embeddings.append([0.0] * 768)

    return np.array(embeddings)

async def ollama_chat_func(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    """
    Generates text using the local Ollama instance.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                # Pass only basic types in options to avoid serialization errors
                "options": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        print(f"Generation failed: {e}")
        return "Error generating response from local LLM."


# --- API Models ---

class SearchResultItem(BaseModel):
    title: str
    content: str
    url: str

class RagQuery(BaseModel):
    query: str
    results: List[SearchResultItem]
    model: Optional[str] = "llama3"
    mode: Optional[str] = "mini"
    only_need_context: Optional[bool] = False

class OllamaEmbedding:
    def __init__(self, func):
        self.func = func
        # Nomic-embed-text is 768, Llama3 is 4096. 
        # We can dynamically check or default to 768 (safest for RAG models).
        self.embedding_dim = 768 
    
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return await self.func(texts)

# --- Endpoints ---

@app.post("/api/rag")
async def process_rag(request: RagQuery):
    global OLLAMA_MODEL
    OLLAMA_MODEL = request.model if request.model else "llama3"

    workspace_dir = "./temp_rag_workspace"
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    try:
        embedding_wrapper = OllamaEmbedding(ollama_embedding_func)
        
        rag = MiniRAG(
            working_dir=workspace_dir,
            llm_model_func=ollama_chat_func,
            embedding_func=embedding_wrapper,
            chunk_token_size=500,
            chunk_overlap_token_size=50,
        )

        documents = [
            f"Source: {r.url}\nTitle: {r.title}\nContent: {r.content}" 
            for r in request.results
        ]
        
        if documents:
            await rag.ainsert(documents)

        # Pass only_need_context to MiniRAG
        response = await rag.aquery(
            request.query, 
            QueryParam(
                mode=request.mode, 
                only_need_context=request.only_need_context
            )
        )
        
        if request.only_need_context:
            return {"context": response}
        
        return {"answer": response}

    except Exception as e:
        print(f"RAG Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
