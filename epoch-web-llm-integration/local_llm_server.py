
import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent OpenMP deadlocks on Mac
torch.set_num_threads(4)

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold model/tokenizer
model = None
tokenizer = None
MODEL_ID = "google/translategemma-4b-it"

def load_model():
    global model, tokenizer
    try:
        logger.info(f"Loading model: {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        # Force CPU usage to avoid MPS/Meta device conflicts
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=None, torch_dtype=torch.float32, trust_remote_code=True)
        model.to("cpu") 
        logger.info(f"Model {MODEL_ID} loaded successfully on CPU!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        pass

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global model, tokenizer
    
    # 1. Immediate Health Check (No Model Required)
    # This allows the UI to see the server is "up" even if model is loading
    try:
         body = await request.json()
         if body.get("check_health"):
             if not model or not tokenizer:
                 return {"status": "loading"}
             return {"status": "ok", "model": MODEL_ID}
    except Exception: 
         pass # Proceed to normal flow if not a health check or body parsing fails

    # 2. Model Loaded Check
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # body is already parsed above if successful, but request.json() caches it
        # so calling it again is fine
        body = await request.json()

        messages = body.get("messages", [])
        if not messages:
            # Handle empty messages gracefully (ping or warmup)
            messages = [{"role": "user", "content": "Hello"}]
            
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 800)
        
        logger.info(f"Received request: Stream={stream}, MaxTokens={max_tokens}")

        # Apply chat template
        try:
             logger.info(f"Applying chat template to {len(messages)} messages...")
             inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
             ).to(model.device)
             logger.info("Chat template applied successfully.")
        except Exception as e:
             logger.error(f"Error applying chat template: {e}")
             logger.error(f"Messages content: {json.dumps(messages)}")
             raise e

        # Basic Generation
        # Note: True streaming support with transformers is a bit more involved (using TextIteratorStreamer)
        # For verifying "try now", we will just do non-streaming first or simulate simple streaming
        
        from transformers import TextStreamer, TextIteratorStreamer
        from threading import Thread

        if stream:
            logger.info("Starting Streaming Generation...")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                inputs, 
                streamer=streamer, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # Generator for streaming response
            async def generate():
                for token in streamer:
                    # Format as OAI stream
                    chunk = {
                        "id": "chatcmpl-123",
                        "object": "chat.completion.chunk",
                        "created": 1234567890,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Final done
                yield "data: [DONE]\n\n"

            from fastapi.responses import StreamingResponse
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            logger.info("Starting Sync Generation...")
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            logger.info("Generation Complete.")
            response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            return {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }]
            }

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=11434)
