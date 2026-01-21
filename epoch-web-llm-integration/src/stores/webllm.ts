import { create } from "zustand";
import { CreateMLCEngine, InitProgressCallback, MLCEngine } from "@mlc-ai/web-llm";

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
        id: "Qwen2.5-3B-Instruct-q4f32_1-MLC",
        name: "Qwen 2.5 3B",
        provider: "Alibaba",
        size: "3B",
        isLocal: false
    },
    {
        id: "Llama-3.2-3B-Instruct-q4f32_1-MLC",
        name: "Llama 3.2 3B",
        provider: "Meta",
        size: "3B",
        isLocal: false
    },
    {
        id: "gemma-2-2b-it-q4f32_1-MLC",
        name: "Gemma 2 2B",
        provider: "Google",
        size: "2B",
        isLocal: false
    },
    {
        id: "Phi-3.5-mini-instruct-q4f16_1-MLC",
        name: "Phi 3.5 Mini",
        provider: "Microsoft",
        size: "3.8B",
        isLocal: false
    }
];

interface WebLLMState {
    engine: MLCEngine | null;
    isLoading: boolean;
    progress: string;
    currentModelId: string;

    // Actions
    loadModel: (modelId: string) => Promise<void>;
    resetEngine: () => Promise<void>;
    setEngine: (engine: MLCEngine | null) => void;
}

// Helper for Local API Streaming
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
        buffer = lines.pop() || "";

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
                    // ignore
                }
            }
        }
    }
}

export const useWebLLMStore = create<WebLLMState>((set, get) => ({
    engine: null,
    isLoading: false,
    progress: "",
    currentModelId: "Qwen2.5-3B-Instruct-q4f32_1-MLC",

    setEngine: (engine) => set({ engine }),

    loadModel: async (modelId: string) => {
        const { engine } = get();
        // Check if already loaded? 
        // Logic from use-web-llm.ts:

        const availableModel = AVAILABLE_MODELS.find(m => m.id === modelId);
        const isLocal = availableModel?.isLocal;
        const customRecord = availableModel?.modelRecord;

        set({ isLoading: true, progress: isLocal ? "Connecting to Local API..." : "Initializing WebLLM...", currentModelId: modelId });

        try {
            if (engine) {
                await engine.unload();
                set({ engine: null });
            }

            if (isLocal) {
                await new Promise(r => setTimeout(r, 500));
                console.log(`Switched to Local API mode for: ${modelId}`);
                set({ progress: "" });

                // For local, we mock the engine
                const mockEngine = {
                    chat: {
                        completions: {
                            create: async (params: any) => {
                                const response = await fetch("http://127.0.0.1:11434/v1/chat/completions", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({
                                        model: modelId,
                                        messages: params.messages,
                                        temperature: params.temperature,
                                        stream: params.stream
                                    })
                                });

                                if (!response.ok) throw new Error("Local API Request Failed");

                                if (params.stream) {
                                    return apiStreamGenerator(response);
                                }
                                return await response.json();
                            }
                        }
                    },
                    unload: async () => { }
                } as unknown as MLCEngine;

                set({ engine: mockEngine });

            } else {
                const initProgressCallback: InitProgressCallback = (report) => {
                    set({ progress: report.text });
                };

                const engineConfig: any = { initProgressCallback };
                if (customRecord) {
                    engineConfig.appConfig = {
                        model_list: [customRecord],
                        use_indexed_db_cache: true
                    };
                }

                console.log("Calling CreateMLCEngine with:", { modelId, engineConfig });
                const eng = await CreateMLCEngine(modelId, engineConfig);
                console.log(`WebLLM Engine loaded: ${modelId}`);
                set({ engine: eng });
            }

        } catch (error) {
            console.error("Failed to load model", error);
            set({ progress: `Error: ${error instanceof Error ? error.message : String(error)}` });
        } finally {
            set({ isLoading: false });
        }
    },

    resetEngine: async () => {
        const { engine } = get();
        if (engine) {
            await engine.unload();
        }
        set({ engine: null, currentModelId: "", progress: "" });
    }
}));
