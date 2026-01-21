"use client";

import { useChat } from "@/hooks/chat";
import { useChatStore, useConfigStore } from "@/stores";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { AskInput } from "./ask-input";

import { useChatThread } from "@/hooks/threads";
import { LoaderIcon } from "lucide-react";
import { MessageRole } from "../../generated";
import MessagesList from "./messages-list";
import { StarterQuestionsList } from "./starter-questions";

const useAutoScroll = (ref: React.RefObject<HTMLDivElement>) => {
  const { messages } = useChatStore();

  useEffect(() => {
    if (messages.at(-1)?.role === MessageRole.USER) {
      ref.current?.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
    }
  }, [messages, ref]);
};

const useAutoResizeInput = (
  ref: React.RefObject<HTMLDivElement>,
  setWidth: (width: number) => void,
) => {
  const { messages } = useChatStore();

  useEffect(() => {
    const updatePosition = () => {
      if (ref.current) {
        setWidth(ref.current.scrollWidth);
      }
    };
    updatePosition();
    window.addEventListener("resize", updatePosition);
    return () => {
      window.removeEventListener("resize", updatePosition);
    };
  }, [messages, ref, setWidth]);
};

const useAutoFocus = (ref: React.RefObject<HTMLTextAreaElement>) => {
  useEffect(() => {
    ref.current?.focus();
  }, [ref]);
};

export const ChatPanel = ({ threadId }: { threadId?: number }) => {
  const searchParams = useSearchParams();
  const queryMessage = searchParams.get("q");
  const hasRun = useRef(false);

  const {
    handleSend,
    streamingMessage,
    isStreamingMessage,
    isStreamingProSearch,
  } = useChat();
  const { messages, setMessages, setThreadId } = useChatStore();
  const { data: thread, isLoading, error } = useChatThread(threadId);

  const [width, setWidth] = useState(0);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const messageBottomRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useAutoScroll(messageBottomRef);
  useAutoResizeInput(messagesRef, setWidth);
  useAutoFocus(inputRef);

  useEffect(() => {
    if (queryMessage && !hasRun.current) {
      setThreadId(null);
      hasRun.current = true;
      handleSend(queryMessage);
    }
  }, [queryMessage]);

  useEffect(() => {
    if (!thread) return;
    setThreadId(thread.id);
    setMessages(thread.messages || []);
  }, [threadId, thread, setMessages, setThreadId]);

  useEffect(() => {
    if (messages.length == 0) {
      setThreadId(null);
    }
  }, [messages, setThreadId]);

  const { offlineMode } = useConfigStore();

  return (
    <>
      {messages.length > 0 || threadId || (queryMessage && hasRun.current) ? (
        isLoading ? (
          <div className="w-full flex justify-center items-center">
            <LoaderIcon className="animate-spin w-8 h-8" />
          </div>
        ) : (
          <div ref={messagesRef} className="pt-10 w-full relative">
            <MessagesList
              messages={messages}
              streamingMessage={streamingMessage}
              isStreamingMessage={isStreamingMessage}
              isStreamingProSearch={isStreamingProSearch}
              onRelatedQuestionSelect={handleSend}
            />
            <div ref={messageBottomRef} className="h-0" />
            <div
              className="bottom-12 fixed px-2 max-w-screen-md justify-center items-center md:px-2"
              style={{ width: `${width}px` }}
            >
              <AskInput isFollowingUp sendMessage={handleSend} />
            </div>
          </div>
        )
      ) : (
        <div className="w-full flex flex-col justify-center items-center">
            <div className="flex flex-col items-center gap-2 mb-8">
              <span className="text-3xl text-center">
                  {offlineMode ? "Ask anything locally without internet" : "Ask anything locally"}
              </span>
              <span className="text-sm text-muted-foreground text-center">
                  {offlineMode ? "Chase curiosity without internet" : "The first SLM-based search that works 100% locally in your browser."}
              </span>
            </div>
          <AskInput sendMessage={handleSend} />
          <div className="w-full flex flex-row px-3 justify-between space-y-2 pt-1">
            <StarterQuestionsList handleSend={handleSend} />
          </div>
        </div>
      )}
    </>
  );
};
