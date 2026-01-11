import { CreateMLCEngine, InitProgressCallback, MLCEngine } from "@mlc-ai/web-llm";
import { create } from 'zustand';

export const AVAILABLE_MODELS = [
  {
    id: "Qwen2.5-3B-Instruct-q4f32_1-MLC",
    name: "Qwen 2.5 3B (Browser)",
    provider: "Alibaba",
    size: "3B",
    isLocal: false,
  },
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    name: "Llama 3.2 3B (Browser)",
    provider: "Meta",
    size: "3B",
    isLocal: false,
  },
  {
    id: "gemma-2-2b-it-q4f32_1-MLC",
    name: "Gemma 2 2B (Browser)",
    provider: "Google",
    size: "2B",
    isLocal: false,
  },
  {
    id: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    name: "Phi 3.5 Mini (Browser)",
    provider: "Microsoft",
    size: "3.8B",
    isLocal: false,
  },
];

interface WebLLMState {
  engine: MLCEngine | null;
  currentModelId: string;
  isLoading: boolean;
  progress: string;
  
  loadModel: (modelId: string) => Promise<void>;
}

export const useWebLLMStore = create<WebLLMState>((set, get) => ({
  engine: null,
  currentModelId: "", 
  isLoading: false,
  progress: "",

  loadModel: async (modelId: string) => {
      const { engine, currentModelId, isLoading } = get();
      if (isLoading || (engine && currentModelId === modelId)) return;

      const modelConfig = AVAILABLE_MODELS.find(m => m.id === modelId);
      // @ts-ignore
      const isLocal = modelConfig?.isLocal;

      set({ isLoading: true, currentModelId: modelId, progress: isLocal ? "Connecting to Local API..." : "Initializing WebLLM..." });
      
      try {
          if (engine) {
              await engine.unload();
              set({ engine: null });
          }

          if (isLocal) {
              set({ progress: "" });
          } else {
              const initProgressCallback: InitProgressCallback = (report) => {
                  set({ progress: report.text });
              };
              
              const timeoutPromise = new Promise((_, reject) => 
                  setTimeout(() => reject(new Error("Model initialization timed out (check internet connection)")), 60000)
              );
              
              const engPromise = CreateMLCEngine(modelId, { 
                  initProgressCallback,
                  logLevel: "INFO" 
              });

              // Race between creation and timeout
              const eng = await Promise.race([engPromise, timeoutPromise]) as MLCEngine;
              set({ engine: eng });
          }
      } catch (err: any) {
          console.error("Failed to load model:", err);
          set({ progress: `Error: ${err.message}` });
          set({ engine: null }); // Ensure engine is null on error
      } finally {
          set({ isLoading: false });
      }
  }
}));
