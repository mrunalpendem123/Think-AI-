import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    const url = request.nextUrl.searchParams.get('url');
    if (!url) return NextResponse.json({ error: 'Missing URL' }, { status: 400 });

    try {
        const response = await fetch(url, {
            headers: {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
        });

        const contentType = response.headers.get('content-type') || '';
        if (!contentType.includes('text/html')) {
            return console.log("Can only browse HTML pages");
        }

        const html = await response.text();

        // Basic URL rewriting to keep browsing within the proxy
        // This is a naive implementation; a real one needs a robust parser
        const baseUrl = new URL(url).origin;
        const proxiedHtml = html
            .replace(/href="\//g, `href="${baseUrl}/`)
            .replace(/src="\//g, `src="${baseUrl}/`)
            // Force links to open in the proxy? 
            // Ideally we intercept clicks in the frontend, so keeping them as absolute URLs is better.
            // But for now, let's just ensure assets load.
            ;

        return new NextResponse(proxiedHtml, {
            headers: {
                'Content-Type': 'text/html',
                'X-Frame-Options': 'SAMEORIGIN' // Allow embedding in our own iframe
            }
        });
    } catch (e: any) {
        return NextResponse.json({ error: e.message }, { status: 500 });
    }
}
