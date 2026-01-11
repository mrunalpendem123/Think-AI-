"use client";

import { useChatThreads, useDeleteThread } from "@/hooks/threads";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores";
import { ChevronLeft, ChevronRight, MessageSquare, Plus, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { SiGithub } from "react-icons/si";
import { Button } from "./ui/button";

export function Sidebar() {
  const { data: threads, isLoading } = useChatThreads();
  const { mutate: deleteThread } = useDeleteThread();
  const { setThreadId, threadId: currentThreadId, setMessages } = useChatStore();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return <div className="w-[260px] h-full bg-muted/20 border-r hidden md:block" />;

  const handleSelect = (id: number) => {
    setThreadId(id);
    router.push('/');
  };

  const handleNew = () => {
    setThreadId(null);
    setMessages([]);
    router.push('/');
  };

  const handleDelete = (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    deleteThread(id);
    if (currentThreadId === id) {
        handleNew();
    }
  };

  return (
    <div 
        className={cn(
            "h-full flex flex-col bg-muted/10 border-r hidden md:flex shrink-0 transition-all duration-300 relative",
            isCollapsed ? "w-[60px]" : "w-[260px]"
        )}
    >
      <div className={cn("p-2 border-b flex items-center", isCollapsed ? "justify-center flex-col gap-2" : "justify-between")}>
        {isCollapsed ? (
             <Button onClick={handleNew} variant="ghost" size="icon" title="New Chat">
                <Plus className="w-4 h-4" />
             </Button>
        ) : (
            <Button onClick={handleNew} variant="outline" className="flex-1 justify-start gap-2 mr-2">
                <Plus className="w-4 h-4" />
                New Chat
            </Button>
        )}

        <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="h-8 w-8 text-muted-foreground"
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-2 scroller">
        {!isCollapsed && isLoading && <div className="text-center text-sm text-muted-foreground p-4">Loading...</div>}
        
        {!isCollapsed && threads && threads.length === 0 && (
            <div className="text-center text-sm text-muted-foreground p-4">
                No history.
            </div>
        )}

        {threads?.map((thread) => (
          <div
            key={thread.id}
            onClick={() => handleSelect(thread.id)}
            className={cn(
              "flex items-center p-2 rounded-md cursor-pointer hover:bg-accent group transition-colors relative",
              currentThreadId === thread.id ? "bg-accent" : "transparent",
              isCollapsed ? "justify-center" : "justify-between"
            )}
            title={thread.title}
          >
            <div className="flex items-center gap-3 overflow-hidden">
              <MessageSquare className="w-4 h-4 shrink-0 text-muted-foreground" />
              {!isCollapsed && <span className="truncate text-sm">{thread.title || "Untitled Chat"}</span>}
            </div>
            {!isCollapsed && (
                <button
                onClick={(e) => handleDelete(e, thread.id)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-opacity"
                >
                <Trash2 className="w-3.5 h-3.5" />
                </button>
            )}
          </div>
        ))}
      </div>

      <div className={cn("p-2 border-t flex items-center gap-1", isCollapsed ? "justify-center flex-col" : "justify-around")}>
        <a href="https://github.com/mrunalpendem123/Think-AI-" target="_blank" rel="noopener noreferrer">
             <Button variant="ghost" size="icon" title="View Source on GitHub">
                <SiGithub className="w-4 h-4" />
             </Button>
        </a>
      </div>
    </div>
  );
}
