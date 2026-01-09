"use client";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import { AVAILABLE_MODELS, useWebLLMStore } from "@/stores/webllm";
import { GlobeIcon, LaptopIcon } from "lucide-react";

export function ModelSelection() {
  const { currentModelId, loadModel, isLoading, progress } = useWebLLMStore();

  const selectedModel = AVAILABLE_MODELS.find(m => m.id === currentModelId) || AVAILABLE_MODELS[0];

  return (
    <div className="flex flex-col gap-1">
    <Select
      value={currentModelId}
      onValueChange={(val) => loadModel(val)}
      disabled={isLoading}
    >
      <SelectTrigger className="w-fit space-x-2 bg-transparent outline-none border-none select-none focus:ring-0 shadow-none transition-all duration-200 ease-in-out hover:scale-[1.05] text-sm">
        <SelectValue placeholder="Select Model">
           {currentModelId && (
             <div className="flex items-center space-x-2">
               {selectedModel.isLocal ? <LaptopIcon className="w-4 h-4 text-orange-500"/> : <GlobeIcon className="w-4 h-4 text-blue-500"/>}
               <span className="font-semibold">{selectedModel.name}</span>
             </div>
           )}
        </SelectValue>
      </SelectTrigger>
      <SelectContent className="w-[300px]">
          {AVAILABLE_MODELS.map((model) => (
            <SelectItem key={model.id} value={model.id} className="cursor-pointer">
              <div className="flex flex-col items-start py-1">
                 <div className="flex items-center gap-2 font-medium">
                    {model.name}
                 </div>
                 <span className="text-xs text-muted-foreground">
                    {model.provider} • {model.size} {model.isLocal ? "• Local API" : "• Client-Side"}
                 </span>
              </div>
            </SelectItem>
          ))}
      </SelectContent>
    </Select>
    {isLoading && <span className="text-[10px] text-muted-foreground animate-pulse pl-2">{progress || "Loading..."}</span>}
    </div>
  );
}
