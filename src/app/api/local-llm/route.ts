import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const { messages, model, temperature } = await req.json();

    // Default to a common local LLM port (Ollama default)
    // You can make this configurable via env vars or UI
    const OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions";

    const response = await fetch(OLLAMA_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: model || "llama3", // Fallback
        messages,
        stream: true, // We want streaming
        temperature: temperature || 0.7,
      }),
    });

    if (!response.ok) {
        const errorText = await response.text();
        return NextResponse.json({ error: `Ollama Error: ${errorText}` }, { status: response.status });
    }

    // Create a streaming response
    const stream = new ReadableStream({
      async start(controller) {
        if (!response.body) {
            controller.close();
            return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value, { stream: true });
          // Ollama/OpenAI chunks come as "data: {...}\n\n"
          // We can just pass-through raw chunks to the client if the client expects standard SSE
          // checking specific format might be safer but pass-through is fastest
          controller.enqueue(new TextEncoder().encode(chunk));
        }
        controller.close();
      },
    });

    return new NextResponse(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
      },
    });

  } catch (error) {
    console.error("Local LLM Proxy Error:", error);
    return NextResponse.json({ error: "Failed to connect to Local LLM" }, { status: 500 });
  }
}
