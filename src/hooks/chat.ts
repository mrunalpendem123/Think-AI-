import { searchWeb } from "@/lib/scraper";
import { useChatStore, useConfigStore } from "@/stores";
import { AVAILABLE_MODELS, useWebLLMStore } from "@/stores/webllm";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import {
    AgentSearchStep,
    ChatMessage,
    ChatRequest,
    ChatResponseEvent,
    ErrorStream,
    MessageRole,
    SearchResult,
    SearchResultStream,
    StreamEndStream,
    StreamEvent,
    TextChunkStream
} from "../../generated";

export const useChat = () => {
  const { addMessage, messages, threadId, setThreadId } = useChatStore();
  const { model, proMode } = useConfigStore();
  
  // Initialize WebLLM Store
  const { engine, loadModel, currentModelId } = useWebLLMStore();

  const [streamingMessage, setStreamingMessage] = useState<ChatMessage | null>(null);
  const [isStreamingProSearch, setIsStreamingProSearch] = useState(false);
  const [isStreamingMessage, setIsStreamingMessage] = useState(false);

  // Stub for agent steps to keep types happy
  let steps_details: AgentSearchStep[] = [];

  const handleEvent = (eventItem: ChatResponseEvent, state: ChatMessage) => {
    switch (eventItem.event) {
      case StreamEvent.BEGIN_STREAM:
        setIsStreamingMessage(true);
        setStreamingMessage({
          ...state,
          role: MessageRole.ASSISTANT,
          content: "",
          related_queries: [],
          sources: [],
          images: [],
        });
        break;
      case StreamEvent.SEARCH_RESULTS:
        const data = eventItem.data as SearchResultStream;
        state.sources = data.results ?? [];
        state.images = data.images ?? [];
        break;
      case StreamEvent.TEXT_CHUNK:
        state.content += (eventItem.data as TextChunkStream).text;
        break;
      case StreamEvent.STREAM_END:
        const endData = eventItem.data as StreamEndStream;
        addMessage({ ...state });
        setStreamingMessage(null);
        setIsStreamingMessage(false);
        setIsStreamingProSearch(false);
        if (endData.thread_id) {
           setThreadId(endData.thread_id);
        }
        break;
      case StreamEvent.ERROR:
         const errorData = eventItem.data as ErrorStream;
         console.error("Stream Error:", errorData.detail);
         setStreamingMessage(null);
         setIsStreamingMessage(false);
         break;
    }
    
    // Update UI state
    setStreamingMessage({
      role: MessageRole.ASSISTANT,
      content: state.content,
      related_queries: state.related_queries,
      sources: state.sources,
      images: state.images,
      agent_response: null,
    });
  };

  const { mutateAsync: chat } = useMutation<void, Error, ChatRequest>({
    retry: false,
    mutationFn: async (request) => {
      if (!engine) {
          console.warn("Engine not ready, attempting to auto-load default...");
          const { loadModel, currentModelId } = useWebLLMStore.getState();
          const targetModel = currentModelId || AVAILABLE_MODELS[0].id;
          
          handleEvent({
              event: StreamEvent.TEXT_CHUNK,
              data: { text: "Initializing AI Model (" + targetModel + ")... Please wait.\n\n" }
          }, {
            role: MessageRole.ASSISTANT,
            content: "",
            sources: [],
            related_queries: [],
            images: [],
            agent_response: null,
          });

          try {
              await loadModel(targetModel);
              // Wait a bit for state to settle
              await new Promise(r => setTimeout(r, 100));
          } catch (e) {
              console.error("Auto-load failed:", e);
              throw new Error("Failed to initialize AI model.");
          }
      }
      
      // Refresh engine reference
      const { engine: readyEngine } = useWebLLMStore.getState();

      const state: ChatMessage = {
        role: MessageRole.ASSISTANT,
        content: "",
        sources: [],
        related_queries: [],
        images: [],
        agent_response: null,
      };
      
      addMessage({ role: MessageRole.USER, content: request.query });
      
      // 1. Simulate BEGIN
      handleEvent({ event: StreamEvent.BEGIN_STREAM, data: {} }, state);

      try {
          // 2. Perform Search
          const searchData = await searchWeb(request.query);
          const searchResults: SearchResult[] = searchData.results.map((r: any, i: number) => ({
              id: i,
              title: r.title,
              url: r.url,
              content: r.content,
              icon: "", 
              metadata: ""
          }));
          const images = (searchData.images || []).map((img: any) => typeof img === 'string' ? img : img.url);

          // Emit Search Results
          handleEvent({
              event: StreamEvent.SEARCH_RESULTS,
              data: { results: searchResults, images: images }
          }, state);

          // 3. Construct Prompt
          const context = searchResults.map(r => `Title: ${r.title}\nContent: ${r.content}`).join("\n\n");
          // 4. Optimize Context with MiniRAG (The "In Between" Layer)
          let ragContext = "";
          
          if (searchResults && searchResults.length > 0) {
              handleEvent({
                  event: StreamEvent.TEXT_CHUNK,
                  data: { text: "Analyzing context with MiniRAG..." } 
              }, state);

              try {
                  const controller = new AbortController();
                  const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s max for MiniRAG

                  const ragResponse = await fetch('http://localhost:8000/api/rag', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                          query: request.query,
                          results: searchResults,
                          mode: 'mini',
                          only_need_context: true
                      }),
                      signal: controller.signal
                  });
                  clearTimeout(timeoutId);

                  if (ragResponse.ok) {
                      const ragData = await ragResponse.json();
                      if (ragData.context) {
                          ragContext = ragData.context;
                          console.log("MiniRAG Context Retrieved:", ragContext.substring(0, 100) + "...");
                      }
                  } else {
                      console.error("MiniRAG Service Error:", ragResponse.statusText);
                      ragContext = searchResults.map(r => `Title: ${r.title}\nContent: ${r.content}`).join('\n\n');
                  }
              } catch (e) {
                  console.error("Failed to reach MiniRAG service:", e);
                  ragContext = searchResults.map(r => `Title: ${r.title}\nContent: ${r.content}`).join('\n\n');
              }
              
              // Clear the "Analyzing..." status text before streaming answer
              state.content = ""; 
          }

          // 5. Generate Answer with Browser Model (Streaming)
          let systemPrompt = "";
          let userContent = "";

          if (proMode) {
              const { DEEPRESEARCH_SYS_PROMPT } = await import("@/lib/deep-research-prompt");
              systemPrompt = DEEPRESEARCH_SYS_PROMPT;
              
              const trace = `TRACE:
User Question: ${request.query}
Action: Search Web
Results: ${searchResults.length} items retrieved.
`;
              const toolCalls = JSON.stringify(searchResults.map(r => ({
                  url: r.url,
                  title: r.title,
                  content: r.content.substring(0, 500) // Truncate for prompt limit fitting
              })), null, 2);

              userContent = `
QUESTION: ${request.query}

TRACE:
${trace}

TOOL_CALLS:
${toolCalls}

Please produce the Plan and Report as instructed.
`;
          } else {
              systemPrompt = `You are a helpful AI assistant. Use the following retrieved context to answer the user's question. Focus on the provided context.
If the context contains "Entities", "Relationships", and "Sources", use them to construct a comprehensive answer.

Context:
${ragContext.substring(0, 6000)}

User Query: ${request.query}
`;
              userContent = request.query;
          }

          const messages = [
              { role: "system", content: systemPrompt },
              { role: "user", content: userContent }
          ];

          if (!readyEngine) {
              handleEvent({
                  event: StreamEvent.TEXT_CHUNK,
                  data: { text: "\n\n**Error:** WebLLM Engine failed to initialize." }
              }, state);
              return;
          }

          const completion = await readyEngine.chat.completions.create({
              stream: true,
              messages: messages as any,
              temperature: 0.7,
              max_tokens: 1024
          });

          for await (const chunk of completion) {
              const delta = chunk.choices[0]?.delta?.content || "";
              if (delta) {
                  handleEvent({
                      event: StreamEvent.TEXT_CHUNK,
                      data: { text: delta }
                  }, state);
              }
          }

          // 5. Finish
          handleEvent({
              event: StreamEvent.STREAM_END,
              data: { thread_id: threadId ? Number(threadId) : Date.now() }
          }, state);

      } catch (err) {
          console.error("Chat Error:", err);
          handleEvent({
              event: StreamEvent.ERROR,
              data: { detail: String(err) }
          }, state);
      }
    },
  });

  const handleSend = async (query: string) => {
    await chat({ query, history: [] }); // History not fully implemented
  };

  return {
    handleSend,
    streamingMessage,
    isStreamingMessage,
    isStreamingProSearch,
    // Expose WebLLM controls
    loadModel,
    currentModelId,
    availableModels: AVAILABLE_MODELS,
    isModelLoading: !engine && !currentModelId.includes("Liquid") // Rough check
  };
};
