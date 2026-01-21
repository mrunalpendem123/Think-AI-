"use client";

import { useChatThreads, useDeleteThread } from "@/hooks/threads";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores";
import { SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/nextjs";
import { ChevronLeft, ChevronRight, MessageSquare, Plus, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { SiGithub, SiX } from "react-icons/si";
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

      <div className={cn("p-2 border-t flex flex-col items-center gap-2", isCollapsed ? "justify-center" : "justify-between")}>
         <div className="w-full flex justify-center">
            <SignedOut>
                 {isCollapsed ? (
                    <SignInButton mode="modal">
                        <Button variant="ghost" size="icon" title="Sign In">
                            <span className="sr-only">Sign In</span>
                             <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-log-in"><path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" x2="3" y1="12" y2="12"/></svg>
                        </Button>
                    </SignInButton>
                 ) : (
                    <SignInButton mode="modal">
                        <Button variant="outline" className="w-full">Sign In</Button>
                    </SignInButton>
                 )}
            </SignedOut>
            <SignedIn>
                <UserButton afterSignOutUrl="/" />
            </SignedIn>
         </div>
         
          <div className={cn("flex items-center gap-1 w-full", isCollapsed ? "justify-center flex-col gap-2" : "justify-around")}>
            <a href="https://x.com/MaxMill06" target="_blank" rel="noopener noreferrer" className={cn("flex justify-center", isCollapsed ? "w-full" : "")}>
                <Button variant="ghost" size="icon" title="Follow on X">
                    <SiX className="w-4 h-4" />
                </Button>
            </a>
            <a href="https://github.com/mrunalpendem123/Think-AI-" target="_blank" rel="noopener noreferrer" className={cn("flex justify-center", isCollapsed ? "w-full" : "")}>
                <Button variant="ghost" size="icon" title="View Source on GitHub">
                    <SiGithub className="w-4 h-4" />
                </Button>
            </a>
            <a href="https://buymeacoffee.com/mrunalpend7" target="_blank" rel="noopener noreferrer" className={cn("flex justify-center", isCollapsed ? "w-full" : "")}>
                <Button variant="ghost" size="icon" title="Buy me a coffee">
                    <span className="text-base">☕</span>
                </Button>
            </a>
         </div>
      </div>
    </div>
  );
}
