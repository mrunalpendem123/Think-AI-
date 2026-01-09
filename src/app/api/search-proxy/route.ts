import { Readability } from '@mozilla/readability';
import * as cheerio from 'cheerio';
import { JSDOM } from 'jsdom';
import type { NextRequest } from 'next/server';
import { NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const { url, query, mode } = await req.json();

    if (mode === 'search') {
      return await handleSearch(query);
    } else {
      return await handleScrape(url);
    }
  } catch (e) {
    console.error('API Error:', e);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

async function handleSearch(query: string) {
  try {
    const apiKey = process.env.PARALLEL_API_KEY || 'sawKl_nOFldN78HAQHFwxixaj90aySp4PTa6trRx';
    
    console.log('[Search API] Querying Parallel.ai for:', query);
    
    const response = await fetch('https://api.parallel.ai/v1beta/search', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'parallel-beta': 'search-extract-2025-10-10'  
      },
      body: JSON.stringify({
        search_queries: [query],
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error('[Search API] Parallel API Error:', response.status, errText);
      throw new Error(`Search provider failed: ${response.status}`);
    }

    const data = await response.json();
    console.log('[Search API] Raw response:', JSON.stringify(data).substring(0, 500));
    
    // Parallel.ai returns: { search_queries: [{ query, results: [...] }] }
    // OR { results: [...] } depending on version
    let results = [];
    
    if (data.search_queries && Array.isArray(data.search_queries) && data.search_queries[0]?.results) {
      results = data.search_queries[0].results;
    } else if (data.results && Array.isArray(data.results)) {
      results = data.results;
    }
    
    if (!results || results.length === 0) {
      console.error('[Search API] No results in response');
      return NextResponse.json({ error: 'No results found' }, { status: 404 });
    }

    console.log(`[Search API] Found ${results.length} results`);

    // Filter out generic landing pages and get first actual article
    const relevantResults = results.filter((r: any) => {
      const url = r.url || '';
      // Skip generic landing pages (exact matches only) and aggregators
      return !url.match(/\/world\/india\/?$/) && 
             !url.match(/\/world\/asia\/india\/?$/) &&
             !url.match(/\/(india|news)\/?$/) &&
             !url.includes('news.google.com');
    });

    // If we have relevant articles, scrape the top 2 in parallel
    // DISABLED FOR SPEED: Deep scraping takes 4-8s. Relying on snippets is near-instant.
    // If we have relevant articles, scrape the top 3 in parallel
    const urlsToScrape = relevantResults.slice(0, 3).map((r: any) => r.url).filter(Boolean);
    const scrapedContentMap: Record<string, string> = {};

    if (urlsToScrape.length > 0) {
      console.log(`[Search API] Scraping articles: ${urlsToScrape.join(', ')}`);
      
      const scrapePromises = urlsToScrape.map(async (url: string) => {
        try {
          // 4s timeout for each scrape
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 4000); 
          
          const res = await handleScrape(url, controller.signal);
          clearTimeout(timeoutId);
          
          if (res.ok) {
            const json = await res.json();
            if (json.content && json.content.length > 200) {
                 scrapedContentMap[url] = json.content;
            }
          }
        } catch (e) {
          console.log(`[Search API] Failed to scrape ${url}:`, e);
        }
      });

      await Promise.all(scrapePromises);
    }

    // Construct structured results for MiniRAG
    const structuredResults = results.slice(0, 6).map((r: any) => {
        let text = scrapedContentMap[r.url];
        if (!text) {
             // Fallback to snippets
             if (r.excerpts && Array.isArray(r.excerpts)) text = r.excerpts.join('\n');
             else if (r.content) text = r.content;
             else if (r.snippet) text = r.snippet;
        }
        return {
            title: r.title,
            url: r.url,
            content: text || ""
        };
    });

    const finalContent = structuredResults.map((r: any) => `Title: ${r.title}\nURL: ${r.url}\nContent: ${r.content.substring(0, 500)}...\n`).join('\n\n');

    return NextResponse.json({
        title: `Search Query: "${query}"`,
        content: finalContent,
        results: structuredResults,
        images: results.flatMap((r: any) => r.thumbnail ? [{ url: r.thumbnail, title: r.title }] : []),
        url: 'https://parallel.ai'
    });

  } catch (e) {
    console.error('[Search API] Error:', e);
    return NextResponse.json({ 
      error: e instanceof Error ? e.message : 'Search failed' 
    }, { status: 500 });
  }
}

async function handleScrape(url: string, signal?: AbortSignal) {
  try {
    const response = await fetch(url, {
       headers: {
         'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)' 
       },
       signal
    });

    if (!response.ok) {
        return NextResponse.json({ error: `Failed to fetch URL: ${response.statusText}` }, { status: response.status });
    }

    const html = await response.text();
    const dom = new JSDOM(html, { url });
    const reader = new Readability(dom.window.document);
    const article = reader.parse();

    if (!article) {
        // Fallback: use Cheerio to get body text
        const $ = cheerio.load(html);
        $('script, style, noscript').remove();
        return NextResponse.json({
            title: $('title').text() || 'No Title',
            content: $('body').text().replace(/\s+/g, ' ').trim().substring(0, 15000),
            url
        });
    }

    return NextResponse.json({
        title: article.title || 'No Title',
        content: (article.textContent || '').replace(/\s+/g, ' ').trim().substring(0, 15000),
        url
    });

  } catch (e) {
     return NextResponse.json({ error: 'Failed to scrape URL' }, { status: 500 });
  }
}
