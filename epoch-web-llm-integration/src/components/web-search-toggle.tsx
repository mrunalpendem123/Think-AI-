"use client";

import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useConfigStore } from "@/stores";
import { Globe, SearchX } from "lucide-react";

export function WebSearchToggle() {
    const { webSearch, toggleWebSearch, offlineMode } = useConfigStore();

    const isDisabled = offlineMode; // If offline, web search implies disabled

    return (
        <div className={`flex items-center space-x-2 border p-1 px-3 rounded-full transition-colors ${isDisabled ? "opacity-50 cursor-not-allowed" : "hover:bg-accent/50"}`}>
            <Switch
                id="web-search-toggle"
                checked={webSearch && !offlineMode}
                onCheckedChange={toggleWebSearch}
                disabled={isDisabled}
                className="scale-75"
            />
            <Label htmlFor="web-search-toggle" className={`flex items-center gap-2 text-xs font-medium ${isDisabled ? "cursor-not-allowed" : "cursor-pointer"}`}>
                {(webSearch && !isDisabled) ? (
                    <>
                        <Globe className="w-3.5 h-3.5 text-blue-500" />
                        <span>Web Search</span>
                    </>
                ) : (
                    <>
                        <SearchX className="w-3.5 h-3.5 text-muted-foreground" />
                        <span className="text-muted-foreground">Local Only</span>
                    </>
                )}
            </Label>
        </div>
    );
}
