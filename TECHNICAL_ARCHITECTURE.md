# Technical Architecture: The Engine Behind Think AI

## Overview
Think AI is not just a wrapper around an API; it is a fully-integrated local retrieval-augmented generation (RAG) system running entirely within the browser. 

We have re-engineered the search stack to remove the dependency on cloud-based inference. By combining **WebGPU** hardware acceleration with **MiniRAG** (a novel lightweight retrieval framework), we enable Small Language Models (SLMs) to perform at a level previously reserved for massive server-side models.

---

## 1. The Compute Engine: WebGPU & WebLLM
Traditional AI relies on CUDA cores running in data centers. Think AI utilizes **WebGPU**, a modern web standard that allows the browser to access the device's native GPU resources (Metal on Mac, Vulkan/DX12 on Windows) for general-purpose parallel computing.

We build upon the **WebLLM** runtime, which maps Large Language Model operations (matrix multiplications, attention mechanisms) directly to GPU shaders. This allows us to load optimized model weights (quantized to 4-bit) into VRAM and execute inference with near-native performance.

**Key capabilities:**
- **Zero-Latency Inference:** No network round-trips for token generation.
- **Privacy compliance:** Input text never leaves the local memory buffer.
- **Hardware Agnostic:** Runs on Apple Silicon, NVIDIA, AMD, and Intel GPUs.

---

## 2. The Retrieval System: MiniRAG
Running an LLM locally is only half the battle. To be useful, it needs knowledge.
Standard RAG (Retrieval-Augmented Generation) requires massive vector databases and heavy embedding models that are too slow for a browser.

We use **MiniRAG**, a framework designed specifically for on-device scenarios. It replaces brute-force vector search with a smarter, graph-based approach.

### How MiniRAG Works
Instead of just storing text chunks, MiniRAG builds a **Heterogeneous Graph Index**:
1.  **Graph Construction:** It identifies "Named Entities" within text and links them to their source chunks.
2.  **Topology-Enhanced Retrieval:** When you search, it doesn't just look for keyword matches. It traverses the graph structure to find related concepts that are topologically connected, even if they don't share exact keywords.
3.  **Efficiency:** It achieves state-of-the-art performance with **75% less storage** and significantly lower compute requirements than traditional GraphRAG or VectorRAG systems.

This allows a Small Language Model (SLM) to "hop" through information logically, simulating the reasoning capabilities of a much larger model.

---

## 3. The Brain: Small Language Models (SLMs)
We do not use GPT-4. We use highly optimized **Small Language Models** that are fine-tuned for instruction following and reasoning.

**Primary Models:**
- **Qwen-2.5-3B:** The current state-of-the-art for <4GB VRAM devices. It offers reasoning capabilities that rival GPT-3.5 while fitting comfortably in browser memory.
- **Phi-3.5-mini:** A Microsoft-developed model optimized for reasoning-heavy tasks with a minimal footprint.
- **Llama-3-8B (Quantized):** For users with higher-end hardware (16GB+ RAM), offering near-GPT-4 logic.

By offloading the "knowledge" part to MiniRAG and the "reasoning" part to the SLM, we create a system that is greater than the sum of its parts. The SLM doesn't need to memorize the internet; it just needs to know how to read the targeted graph data provided by MiniRAG.

---

## Summary of Data Flow
1.  **User Query:** "How does the Photosystem II complex work?"
2.  **MiniRAG Retrieval:** The system queries the local graph index, traversing nodes from "Photosystem II" to "Electron Transport" to "ATP Synthase".
3.  **Context Construction:** Relevant text chunks are retrieved and assembled into a structured context window.
4.  **Inference:** The context + query are sent to the WebLLM engine running on the local GPU.
5.  **Generation:** The SLM generates a coherent, cited answer in real-time.

This architecture proves that intelligence is not about size; it is about efficiency.
