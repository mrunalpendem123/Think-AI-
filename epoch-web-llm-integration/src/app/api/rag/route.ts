import { NextResponse } from 'next/server';

interface SearchResult {
    title: string;
    url: string;
    content: string;
}

interface RagRequest {
    query: string;
    results: SearchResult[];
    mode?: string;
    only_need_context?: boolean;
}

// Simple text similarity/scoring function
function calculateRelevance(text: string, queryTerms: string[]): number {
    const textLower = text.toLowerCase();
    let score = 0;
    queryTerms.forEach(term => {
        if (textLower.includes(term)) score += 1;
    });
    return score;
}

export async function POST(req: Request) {
    try {
        const body: RagRequest = await req.json();
        const { query, results } = body;

        if (!results || results.length === 0) {
             return NextResponse.json({ context: "" });
        }

        console.log(`[MiniRAG] Processing ${results.length} results for query: "${query}"`);

        // 1. Basic preprocessing
        const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 3);
        
        // 2. Score and Sort Segments
        // We split content into "chunks" (paragraphs) and rank them.
        const allChunks: { text: string; score: number; source: string }[] = [];

        results.forEach(result => {
            // Split by double newline or simple newline
            const chunks = result.content.split(/\n\n+/);
            chunks.forEach(chunk => {
                const cleanChunk = chunk.trim();
                if (cleanChunk.length > 50) { // Ignore tiny chunks
                    const score = calculateRelevance(cleanChunk, queryTerms);
                    // Boost score if it's from the first chunk (often title/intro)
                    const titleBoost = calculateRelevance(result.title, queryTerms);
                    
                    allChunks.push({
                        text: `Source: ${result.title}\n${cleanChunk}`,
                        score: score + (titleBoost * 0.5),
                        source: result.title
                    });
                }
            });
        });

        // 3. Sort by score descending
        allChunks.sort((a, b) => b.score - a.score);

        // 4. Select top chunks until limit (e.g. 6000 chars)
        let context = "";
        let currentLength = 0;
        const MAX_LENGTH = 8000;

        for (const chunk of allChunks) {
            if (currentLength + chunk.text.length < MAX_LENGTH) {
                context += chunk.text + "\n\n";
                currentLength += chunk.text.length;
            } else {
                break;
            }
        }

        if (context.length === 0) {
            // Fallback if no relevant chunks found: just take raw top results
            context = results.slice(0, 3).map(r => `Source: ${r.title}\n${r.content}`).join("\n\n");
        }

        return NextResponse.json({
            context: context.trim(),
            debug_info: {
                chunks_processed: allChunks.length,
                context_length: context.length
            }
        });

    } catch (error) {
        console.error('[MiniRAG] Error:', error);
        return NextResponse.json({ error: 'Failed to process context' }, { status: 500 });
    }
}
