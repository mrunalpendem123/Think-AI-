"use client";

import { useAppModeStore } from "@/stores/app-mode";
import { useEffect } from "react";

export function BrowserView() {
    const { browserUrl, mode } = useAppModeStore();

    useEffect(() => {
        // Expose helper for agent script to set url
        // @ts-ignore
        window._setBrowserUrl = useAppModeStore.getState().setBrowserUrl;
    }, []);

    if (mode !== 'agent' || !browserUrl) return null;

    return (
        <div className="absolute inset-0 z-40 bg-background flex flex-col">
            <div className="h-10 border-b flex items-center px-4 bg-muted/50 text-sm font-mono truncate">
                Browsing: {browserUrl}
            </div>
            {/* 
               We use the proxy route to fetch content. 
               Normally we'd use an iframe pointing to our proxy route 
            */}
            <iframe
                src={`/api/browser?url=${encodeURIComponent(browserUrl)}`}
                className="flex-1 w-full border-none h-full"
                sandbox="allow-same-origin allow-scripts allow-forms"
                id="agent-browser-frame"
            />
        </div>
    );
}
