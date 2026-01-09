// scraper.ts
export interface ScrapeResult {
  title: string;
  content: string;
  url: string;
}

export interface SearchResponse {
    results: {
        title: string;
        url: string;
        content: string;
    }[];
    images?: {
        url: string;
        title: string;
    }[];
}

export async function scrapeUrl(url: string): Promise<ScrapeResult> {
    const response = await fetch('/api/search-proxy', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ url })
    });

    if (!response.ok) {
       const err = await response.json();
       throw new Error(err.error || 'Failed to fetch page via internal proxy');
    }

    const data = await response.json();
    return {
       title: data.title || 'No Title',
       content: data.content || '',
       url: data.url || url
    };
  }


export async function searchWeb(query: string): Promise<SearchResponse> {
    const response = await fetch('/api/search-proxy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode: 'search' })
     });
 
     if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'Search failed via internal proxy');
     }
 
     const data = await response.json();
     // The proxy API returns { results: [...], images: [...] } or checks implementation
     return {
        results: data.results || [],
        images: data.images || []
     };
 }
