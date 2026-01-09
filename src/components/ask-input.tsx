import { useWebLLMStore } from "@/stores/webllm";
import { ArrowUp } from "lucide-react";
import { useState } from "react";
import TextareaAutosize from "react-textarea-autosize";
import { ModelSelection } from "./model-selection";
import ProToggle from "./pro-toggle";
import { Button } from "./ui/button";

const InputBar = ({
  input,
  setInput,
}: {
  input: string;
  setInput: (input: string) => void;
}) => {
  const { engine, progress, isLoading } = useWebLLMStore();
  const isReady = !!engine;

  return (
    <div className="w-full flex flex-col rounded-md focus:outline-none px-2 py-1 bg-card border-2 ">
      <div className="w-full">
        <TextareaAutosize
          className="w-full bg-transparent text-md resize-none focus:outline-none p-2"
          placeholder={isLoading ? `Initializing: ${progress}` : "Ask anything..."}
          onChange={(e) => setInput(e.target.value)}
          value={input}
          disabled={isLoading}
        />
      </div>
      <div className="flex justify-between">
        <div>
          <ModelSelection />
        </div>
        <div className="flex items-center gap-2">
          <ProToggle />
          <Button
            type="submit"
            variant="default"
            size="icon"
            className="rounded-full bg-tint aspect-square h-8 w-8 disabled:opacity-20 hover:bg-tint/80 overflow-hidden"
            disabled={input.trim().length < 2 && !isLoading}
          >
            <ArrowUp size={20} />
          </Button>
        </div>
      </div>
      {isLoading && (
        <div className="px-2 pb-1">
             <span className="text-xs text-muted-foreground animate-pulse">{progress || "Loading Model..."}</span>
        </div>
      )}
    </div>
  );
};
// ... similar update for FollowingUpInput ...
// Actually, to save complexity, update AskInput wrapper to check store? 
// But InputBar has the UI.
// Let's just update InputBar and FollowingUpInput to respect store state.

const FollowingUpInput = ({
  input,
  setInput,
}: {
  input: string;
  setInput: (input: string) => void;
}) => {
  const { engine, isLoading } = useWebLLMStore();
  const isReady = !!engine;
  
  return (
    <div className="w-full flex flex-row rounded-full focus:outline-none px-2 py-1 bg-card border-2 items-center ">
      <div className="w-full">
        <TextareaAutosize
          className="w-full bg-transparent text-md resize-none focus:outline-none p-2"
          placeholder="Ask anything..."
          onChange={(e) => setInput(e.target.value)}
          value={input}
          disabled={isLoading || !isReady}
        />
      </div>
      <div className="flex items-center gap-2">
        <ProToggle />
        <Button
          type="submit"
          variant="default"
          size="icon"
          className="rounded-full bg-tint aspect-square h-8 w-8 disabled:opacity-20 hover:bg-tint/80 overflow-hidden"
          disabled={input.trim().length < 5 || !isReady}
        >
          <ArrowUp size={20} />
        </Button>
      </div>
    </div>
  );
};

export const AskInput = ({
  sendMessage,
  isFollowingUp = false,
}: {
  sendMessage: (message: string) => void;
  isFollowingUp?: boolean;
}) => {
  const [input, setInput] = useState("");
  const { engine } = useWebLLMStore(); // Check here too to prevent Enter submission

  return (
    <>
      <form
        className="w-full overflow-hidden"
        onSubmit={(e) => {
          e.preventDefault();
          if (input.trim().length < 2) return;
          
          if (!engine) {
              const { loadModel, isLoading } = useWebLLMStore.getState();
              if (isLoading) return;
              // Auto-select first model if none selected
              // Actually, better to just let message pass and let hook handle auto-load, 
              // BUT hook handles logic better. 
              // Sending anyway.
          }
          sendMessage(input);
          setInput("");
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (input.trim().length < 2) return;
            sendMessage(input);
            setInput("");
          }
        }}
      >
        {isFollowingUp ? (
          <FollowingUpInput input={input} setInput={setInput} />
        ) : (
          <InputBar input={input} setInput={setInput} />
        )}
      </form>
    </>
  );
};
