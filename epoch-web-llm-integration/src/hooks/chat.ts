import { saveThread } from "@/lib/indexed-db";
import { performMiniRAG } from "@/lib/minirag";
import { searchWeb } from "@/lib/scraper";
import { useChatStore, useConfigStore, useStore } from "@/stores";
import { AVAILABLE_MODELS, useWebLLMStore } from "@/stores/webllm";
import { useMutation, useQueryClient } from "@tanstack/react-query";
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
    const queryClient = useQueryClient();
    const { addMessage, messages, threadId, setThreadId } = useChatStore();
    const { model, proMode, offlineMode, webSearch } = useConfigStore();

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

                // Save to IndexedDB
                // We use a timeout to let the store update? Or just read current messages + new one.
                // Better: Read store messages AFTER addMessage.
                setTimeout(async () => {
                    const { messages, threadId } = useStore.getState();
                    const title = messages[0]?.content.slice(0, 50) || "New Chat";
                    const newId = await saveThread({
                        id: threadId ? Number(threadId) : undefined,
                        title,
                        messages,
                    });
                    if (!threadId) {
                        setThreadId(newId);
                    }
                    queryClient.invalidateQueries({ queryKey: ["threads"] });
                }, 100);

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
        networkMode: 'always',
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
            const { messages: history } = useStore.getState();

            const state: ChatMessage = {
                role: MessageRole.ASSISTANT,
                content: "",
                sources: [],
                related_queries: [],
                images: [],
                agent_response: null,
            };

            // Filter out empty or error messages if any
            const validHistory = history.filter((m: ChatMessage) => m.content);

            let searchQueries = [request.query];

            // 0. Contextualize Query (if history exists and query is ambiguous)
            // Heuristic: If query is short (< 4 words) or contains pronouns, we contextulize.
            // Otherwise, we trust the direct query (Perplexity style speed).
            const isAmbiguous = request.query.split(' ').length < 4 ||
                /\b(it|this|that|he|she|they|him|her)\b/i.test(request.query);

            // Perform Contextualization if ONLINE and WEB SEARCH ENABLED
            const shouldSearch = !offlineMode && webSearch;

            if (shouldSearch && validHistory.length > 0 && readyEngine && isAmbiguous) {
                handleEvent({
                    event: StreamEvent.TEXT_CHUNK,
                    data: { text: "Understanding context..." }
                }, { ...state, content: "" }); // Temporary status

                const contextPrompt = `
Given the following conversation history, rephrase the last user query to be a standalone search query that incorporates necessary context.
If the query is already standalone, return it as is. Do not answer the question, just return the search query.

Conversation:
${validHistory.slice(-4).map((m: ChatMessage) => `${m.role.toUpperCase()}: ${m.content}`).join('\n')}
User: ${request.query}

Standalone Search Query:
`;
                try {
                    const contextCompletion = await readyEngine.chat.completions.create({
                        messages: [{ role: "user", content: contextPrompt as string }],
                        temperature: 0.1, // Low temp for precision
                        max_tokens: 64
                    });
                    const rephrased = contextCompletion.choices[0]?.message?.content?.trim();
                    if (rephrased) {
                        searchQueries = [rephrased];
                        console.log("Rephrased Query:", rephrased);
                    }
                } catch (e) {
                    console.warn("Contextualization failed:", e);
                }

                // Clear status
                state.content = "";
                handleEvent({
                    event: StreamEvent.TEXT_CHUNK,
                    data: { text: "" }
                }, state);
            } else if (shouldSearch) {
                console.log("Skipping contextualization for speed (Query deemed standalone or history empty)");
            }

            addMessage({ role: MessageRole.USER, content: request.query });

            // 1. Simulate BEGIN
            handleEvent({ event: StreamEvent.BEGIN_STREAM, data: {} }, state);

            try {
                // 2. Perform Search
                let searchResults: SearchResult[] = [];

                if (shouldSearch) {
                    handleEvent({
                        event: StreamEvent.TEXT_CHUNK,
                        data: { text: "Searching..." }
                    }, { ...state, content: "" });

                    // Use the rephrased query for search
                    const searchData = await searchWeb(searchQueries[0]);
                    searchResults = searchData.results.map((r: any, i: number) => ({
                        id: i,
                        title: r.title,
                        url: r.url,
                        content: r.content,
                        icon: "",
                        metadata: ""
                    }));
                    state.images = (searchData.images || []).map((img: any) => typeof img === 'string' ? img : img.url);

                    // Emit Search Results
                    handleEvent({
                        event: StreamEvent.SEARCH_RESULTS,
                        data: { results: searchResults, images: state.images }
                    }, state);
                }

                // 3. Construct Prompt
                const context = searchResults.map(r => `Title: ${r.title}\nContent: ${r.content}`).join("\n\n");
                // 4. Optimize Context with MiniRAG
                let ragContext = "";

                if (searchResults && searchResults.length > 0) {
                    // Show Analyzing status
                    state.content = "Analyzing context...";
                    handleEvent({
                        event: StreamEvent.TEXT_CHUNK,
                        data: { text: "" } // Dummy update to trigger UI with new content
                    }, state);

                    try {
                        const ragContextResult = performMiniRAG(searchQueries[0], searchResults);
                        if (ragContextResult) {
                            ragContext = ragContextResult;
                            console.log("MiniRAG Context Retrieved (Client-side)");
                        }
                    } catch (e) {
                        console.warn("MiniRAG unavailable (using raw results):", e);
                    }

                    // Clear status text
                    state.content = "";
                    handleEvent({
                        event: StreamEvent.TEXT_CHUNK,
                        data: { text: "" }
                    }, state);

                    if (!ragContext) {
                        ragContext = searchResults.map(r => `Title: ${r.title}\nContent: ${r.content}`).join('\n\n');
                    }
                }

                // 5. Generate Answer with Browser Model (Streaming)
                // Show Thinking status
                state.content = "Thinking...";
                handleEvent({
                    event: StreamEvent.TEXT_CHUNK,
                    data: { text: "" }
                }, state);

                let systemPrompt = "";
                let userContent = "";

                if (!shouldSearch) {
                    state.content = ""; // Clear thinking for offline
                    handleEvent({ event: StreamEvent.TEXT_CHUNK, data: { text: "" } }, state);

                    if (offlineMode) {
                        systemPrompt = "You are a helpful AI assistant running in offline mode. Answer the user's question based on your internal knowledge. Do not try to search the web.";
                    } else {
                        // Web Search Disabled, but Online
                        systemPrompt = "You are a helpful AI assistant. Answer the user's question based on your internal knowledge. Do not try to search the web unless explicitly enabled.";
                    }
                    userContent = request.query;
                } else if (proMode) {
                    const { DEEPRESEARCH_SYS_PROMPT } = await import("@/lib/deep-research-prompt");
                    systemPrompt = DEEPRESEARCH_SYS_PROMPT;

                    const trace = `TRACE:
User Question: ${searchQueries[0]}
Action: Search Web
Results: ${searchResults.length} items retrieved.
`;
                    const toolCalls = JSON.stringify(searchResults.map((r: SearchResult) => ({
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
${ragContext.substring(0, 4000)}

User Query: ${request.query}
`;
                    userContent = request.query;
                }

                // Construct full message history for the final call
                const formattedHistory = validHistory.slice(-6).map((m: ChatMessage) => ({
                    role: m.role.toLowerCase() as "user" | "assistant",
                    content: m.content
                }));

                const messages = [
                    { role: "system", content: systemPrompt },
                    ...formattedHistory,
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

                // Clear Thinking status once generation starts
                if (state.content === "Thinking...") {
                    state.content = "";
                }
                handleEvent({ event: StreamEvent.TEXT_CHUNK, data: { text: "" } }, state);

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
                if (String(err).includes("disposed")) {
                    useWebLLMStore.setState({ engine: null });
                }
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
