"use client";

import { Button } from "@/components/ui/button";
import { useAppModeStore } from "@/stores/app-mode";
import { Bot, MessageSquare } from "lucide-react";

export function AgentModeToggle() {
    const { mode, toggleMode } = useAppModeStore();

    return (
        <Button
            variant="outline"
            size="sm"
            onClick={toggleMode}
            className={`gap-2 ${mode === 'agent' ? 'bg-primary text-primary-foreground hover:bg-primary/90' : ''}`}
        >
            {mode === 'chat' ? (
                <>
                    <MessageSquare className="h-[1.2rem] w-[1.2rem]" />
                    Chat
                </>
            ) : (
                <>
                    <Bot className="h-[1.2rem] w-[1.2rem]" />
                    Agent
                </>
            )}
        </Button>
    );
}
