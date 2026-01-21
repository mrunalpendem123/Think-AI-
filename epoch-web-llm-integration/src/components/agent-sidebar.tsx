"use client";

import { cleanupOverlay, showNumberedOverlay, tools } from '@/lib/agent-tools';
import { getAgentTask, saveAgentTask } from '@/lib/indexed-db';
import { useAppModeStore } from '@/stores/app-mode';
import { useWebLLMStore } from '@/stores/webllm';
import { Loader2, Play, StopCircle } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';

interface AgentTask {
    goal: string;
    logs: string[];
    status: 'idle' | 'running' | 'completed' | 'failed';
}

function BotIcon() {
    return (
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-5 w-5"
        >
            <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 12 2Z" />
            <path d="M9 13a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v1a2 2 0 0 1-2 2h-2a2 2 0 0 1-2-2v-1Z" />
            <path d="M15 9v2" />
            <path d="M9 9v2" />
        </svg>
    )
}

export function AgentSidebar() {
    const { mode } = useAppModeStore();
    const { engine } = useWebLLMStore();
    const [goal, setGoal] = useState("");
    const [logs, setLogs] = useState<string[]>([]);
    const [status, setStatus] = useState<AgentTask['status']>('idle');
    const abortControllerRef = useRef<AbortController | null>(null);

    const addLog = (msg: string) => setLogs(prev => [...prev, msg]);

    useEffect(() => {
        if (mode === 'agent') {
            getAgentTask('current').then(task => {
                if (task) {
                    setGoal(task.goal);
                    setLogs(task.logs || []);
                    if (task.status === 'completed') setStatus('completed');
                }
            });
        } else {
            cleanupOverlay();
        }
    }, [mode]);

    useEffect(() => {
        if (mode === 'agent' && goal) {
            saveAgentTask({
                id: 'current',
                goal,
                logs,
                status: status
            });
        }
    }, [goal, logs, status, mode]);

    const runAgent = async () => {
        if (!engine || !goal) return;
        setStatus('running');
        setLogs([]);
        addLog(`Goal: ${goal}`);
        abortControllerRef.current = new AbortController();

        try {
            let currentStep = 0;
            const maxSteps = 15;

            while (currentStep < maxSteps && !abortControllerRef.current.signal.aborted) {
                currentStep++;
                addLog(`\n--- Step ${currentStep} ---`);

                // 1. Scrape & Overlay
                const elementCount = await showNumberedOverlay();
                const pageContentRaw = await tools.scrape_active_tab();
                const pageContent = typeof pageContentRaw === 'string' ? pageContentRaw : "";
                addLog(`Scraped page. Found ${elementCount} interactive elements.`);

                // 2. Think
                const prompt = `You are a Browser Agent. Your goal is: "${goal}".
            
Current Browser Content (Simplified):
${pageContent.slice(0, 3000)}

Interactive Elements are marked with numbers on the screen.
Available Tools:
- click_element(id: number): Click on a numbered element.
- input_text(id: number, text: string): Type text into a numbered input. Use this for search bars or forms.
- navigate(url: string): Open a new website (e.g., "https://bookmyshow.com").
- finish_task(): Call this when the goal is achieved.

Choose the next action. Respond ONLY with a valid JSON object in this format:
{
  "thought": "Reasoning about what to do...",
  "tool_name": "click_element" | "input_text" | "finish_task",
  "tool_args": [1] | [1, "text"] | []
}
`;

                addLog("Thinking...");
                // Force strict JSON if possible, otherwise parse loosely
                const response = await engine.chat.completions.create({
                    messages: [{ role: 'user', content: prompt }],
                    temperature: 0.1,
                    max_tokens: 500
                });

                const content = response.choices[0].message.content || "{}";
                let action: any = {};
                try {
                    // simple json extraction if wrapped in code blocks
                    const jsonStr = content.replace(/```json/g, '').replace(/```/g, '').trim();
                    action = JSON.parse(jsonStr);
                } catch (e) {
                    console.error(e);
                    addLog(`Error parsing JSON response. Retrying step.`);
                    continue;
                }

                if (action.thought) addLog(`Thought: ${action.thought}`);

                // 3. Act
                if (action.tool_name === 'finish_task') {
                    addLog("Task Completed!");
                    setStatus('completed');
                    break;
                } else if (action.tool_name === 'click_element') {
                    const id = action.tool_args ? action.tool_args[0] : undefined;
                    if (!id) throw new Error("Missing ID for click_element");
                    addLog(`Clicking element ${id}...`);
                    const res = await tools.click_element(id);
                    addLog(res as string);
                    // Wait for navigation
                    await new Promise(r => setTimeout(r, 2000));
                } else if (action.tool_name === 'navigate') {
                    const url = action.tool_args ? action.tool_args[0] : undefined;
                    addLog(`Navigating to ${url}...`);
                    const res = await tools.navigate(url);
                    addLog(res as string);
                } else if (action.tool_name === 'input_text') {
                    const id = action.tool_args ? action.tool_args[0] : undefined;
                    const text = action.tool_args ? action.tool_args[1] : undefined;
                    addLog(`Inputting "${text}" into ${id}...`);
                    const res = await tools.input_text(id, text);
                    addLog(res as string);
                } else {
                    addLog(`Unknown tool: ${action.tool_name} or format error.`);
                }

                // Wait a bit
                await new Promise(r => setTimeout(r, 1000));
            }
        } catch (e: any) {
            addLog(`Error: ${e.message}`);
            setStatus('failed');
        } finally {
            if (status !== 'completed' && !abortControllerRef.current?.signal.aborted) setStatus('idle');
            cleanupOverlay();
        }
    };

    const stopAgent = () => {
        abortControllerRef.current?.abort();
        setStatus('idle');
        cleanupOverlay();
        addLog("Agent stopped by user.");
    };

    if (mode !== 'agent') return null;

    return (
        <div className="fixed right-0 top-[60px] bottom-0 w-96 bg-background border-l z-[50] flex flex-col shadow-2xl">
            <div className="p-4 border-b">
                <h2 className="font-semibold text-lg flex items-center gap-2">
                    <BotIcon /> Agent Co-worker
                </h2>
            </div>

            <div className="p-4 space-y-4 flex-1 overflow-hidden flex flex-col">
                <div className="space-y-2">
                    <label className="text-sm font-medium">Current Goal</label>
                    <Input
                        value={goal}
                        onChange={e => setGoal(e.target.value)}
                        placeholder="e.g. Find the definition of Quantum Computing"
                        disabled={status === 'running'}
                    />
                </div>

                <div className="flex gap-2">
                    {status === 'running' ? (
                        <Button onClick={stopAgent} variant="destructive" className="w-full">
                            <StopCircle className="mr-2 h-4 w-4" /> Stop
                        </Button>
                    ) : (
                        <Button onClick={runAgent} disabled={!goal || !engine} className="w-full">
                            <Play className="mr-2 h-4 w-4" /> Start Agent
                        </Button>
                    )}
                </div>

                <div className="flex-1 border rounded-md p-2 bg-muted/50 h-full overflow-y-auto">
                    {logs.length === 0 && <span className="text-muted-foreground text-sm">Waiting for instructions...</span>}
                    {logs.map((log, i) => (
                        <div key={i} className="mb-2 text-xs font-mono break-words whitespace-pre-wrap">
                            {log}
                        </div>
                    ))}
                    {status === 'running' && <Loader2 className="h-4 w-4 animate-spin mt-2" />}
                </div>
            </div>
        </div>
    );
}
