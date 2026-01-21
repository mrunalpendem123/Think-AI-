"use client";

import { useChatStore } from "@/stores";
import { PlusIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { useRouter } from "next/navigation";
import { UserButton } from "@clerk/nextjs";
import { ModeToggle } from "./mode-toggle";
import { OfflineModeToggle } from "./offline-mode-toggle";
import { WebSearchToggle } from "./web-search-toggle";
import { AgentModeToggle } from "./agent-mode-toggle";
import { Button } from "./ui/button";

const NewChatButton = () => {
  return (
    <Button variant="secondary" size="sm" onClick={() => (location.href = "/")}>
      <PlusIcon className="w-4 h-4" />
      <span className="block">&nbsp;&nbsp;New</span>
    </Button>
  );
};

const TextLogo = () => {
  return <div className="text-2xl font-medium">Think AI</div>;
};

export function Navbar() {
  const router = useRouter();
  const { theme } = useTheme();
  const { messages } = useChatStore();

  const onHomePage = messages.length === 0;

  return (
    <header className="w-full flex fixed p-1 z-50 px-2 bg-background/95 justify-between items-center">
      <div className="flex items-center gap-2">

        {onHomePage ? <TextLogo /> : <NewChatButton />}
      </div>
      <div className="flex items-center gap-4">
        <OfflineModeToggle />
        <WebSearchToggle />
        <AgentModeToggle />
        <ModeToggle />
        <UserButton />
      </div>
    </header>
  );
}
