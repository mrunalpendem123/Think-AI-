"use client";

import { CreateMLCEngine, InitProgressCallback, MLCEngine } from "@mlc-ai/web-llm";
import { useEffect, useRef, useState } from "react";

// Helper to mimic WebLLM's streaming response from our API
async function* apiStreamGenerator(response: Response) {
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    if (!reader) return;

    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep the last incomplete line

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed === "data: [DONE]") continue;
            if (trimmed.startsWith("data: ")) {
                try {
                    const json = JSON.parse(trimmed.slice(6));
                    if (json.choices && json.choices[0]?.delta) {
                        yield {
                            choices: [{
                                delta: {
                                    content: json.choices[0].delta.content
                                }
                            }]
                        };
                    }
                } catch (e) {
                   // ignore parse errors for partial chunks
                }
            }
        }
    }
}

  export interface ModelOption {
    id: string;
    name: string;
    provider: string;
    size: string;
    isLocal?: boolean;
    modelRecord?: any;
  }
  
  export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: "DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC",
    name: "DeepSeek R1 Distill Qwen 7B",
    provider: "DeepSeek",
    size: "7B",
  },
  {
    id: "DeepSeek-R1-Distill-Llama-8B-q4f32_1-MLC",
    name: "DeepSeek R1 Distill Llama 8B",
    provider: "DeepSeek",
    size: "8B",
  },
  {
    id: "LFM2.5-VL-1.6B-Instruct",
    name: "LiquidAI LFM2.5 VL (via Ollama)",
    provider: "Liquid AI",
    size: "1.6B",
    isLocal: true, // Needs Ollama
  },
  {
    id: "Qwen2.5-3B-Instruct-q4f32_1-MLC",
    name: "Qwen 2.5 3B (Browser)",
    provider: "Alibaba",
    size: "3B",
  },
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 3B (Browser)",
    provider: "Meta",
    size: "3B",
  },
  {
    id: "gemma-2-2b-it-q4f32_1-MLC",
    name: "Gemma 2 2B (Browser)",
    provider: "Google",
    size: "2B",
  },
  {
    id: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    name: "Phi 3.5 Mini (Browser)",
    provider: "Microsoft",
    size: "3.8B",
  },
  {
      id: "LFM2-VL-3B-MLC",
      name: "Liquid LFM2-VL 3B (Conversion Failed - Unsupported Arch)",
      provider: "Liquid AI",
      size: "3B",
      modelRecord: {
          "model_id": "LFM2-VL-3B-MLC",
          "model_lib_url": "", 
          "vram_required_MB": 3000,
          "low_resource_required": true,
          "model_url": "http://localhost:8080/LFM2-VL-3B-MLC/" 
      }
  }
];

export function useWebLLM() {
  const [engine, setEngine] = useState<MLCEngine | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState("");
  const [currentModelId, setCurrentModelId] = useState<string>("Qwen2.5-3B-Instruct-q4f32_1-MLC"); // Default to Qwen
  
  const isInitializingRef = useRef(false);

  const loadModel = async (modelId: string) => {
    if (isInitializingRef.current) return;
    
    // Check if it's a local model (Ollama)
    const availableModel = AVAILABLE_MODELS.find(m => m.id === modelId);
    const isLocal = availableModel?.isLocal;

    // Check if it's a Custom WebLLM model (defined with modelRecord)
    const customRecord = availableModel?.modelRecord;

    isInitializingRef.current = true;
    setIsLoading(true);
    setCurrentModelId(modelId);
    setProgress(isLocal ? "Connecting to Local API..." : "Initializing WebLLM...");

    try {
      if (engine) {
          await engine.unload();
          setEngine(null);
      }

      if (isLocal) {
          await new Promise(r => setTimeout(r, 500)); 
          console.log(`Switched to Local API mode for: ${modelId}`);
          setProgress("");
      } else {
          // WebLLM Loading
          const initProgressCallback: InitProgressCallback = (report) => {
            setProgress(report.text);
          };
          
          const engineConfig: any = { initProgressCallback };
          
          // Inject custom model config if present
          if (customRecord) {
              engineConfig.appConfig = {
                  model_list: [customRecord]
              };
          }

          const eng = await CreateMLCEngine(modelId, engineConfig);
          console.log(`WebLLM Engine loaded: ${modelId}`);
          setEngine(eng);
      }
    } catch (error) {
      console.error("Failed to load model", error);
      setProgress(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsLoading(false);
      isInitializingRef.current = false;
    }
  };

  // Unified Chat Completion Interface
  const chat = {
    completions: {
       create: async (params: any) => {
           // 1. WebLLM Mode
           if (engine) {
               return engine.chat.completions.create(params);
           }
           
           // 2. Local API Mode (Fallback if no engine but we have a Local Model selected)
           const modelConfig = AVAILABLE_MODELS.find(m => m.id === currentModelId);
           if (modelConfig?.isLocal) {
                // Call our proxy
                const response = await fetch("/api/local-llm", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        model: currentModelId, // Pass the ID, user must have matching model in Ollama or we map it
                        messages: params.messages,
                        temperature: params.temperature
                    })
                });
                
                if (!response.ok) throw new Error("Local API Request Failed");
                
                // Return an async iterable that matches WebLLM's expectation
                if (params.stream) {
                    return apiStreamGenerator(response);
                }
                // Handle non-stream if needed (though UI uses stream)
                const json = await response.json();
                return json; 
           }

           throw new Error("No Engine Loaded and not in Local Mode");
       }
    }
  };

  useEffect(() => {
    // Load default if needed
    if (!engine && !isInitializingRef.current && !AVAILABLE_MODELS.find(m => m.id === currentModelId)?.isLocal) {
        loadModel("Qwen2.5-3B-Instruct-q4f32_1-MLC");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { 
    engine: { chat }, // Mock the engine interface for the UI
    isLoading, 
    progress, 
    loadModel, 
    currentModelId,
    availableModels: AVAILABLE_MODELS 
  };
}
