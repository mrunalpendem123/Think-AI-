import { Readability } from '@mozilla/readability';

let elementMap: Map<number, HTMLElement> = new Map();
let overlayContainer: HTMLElement | null = null;

export const cleanupOverlay = () => {
    if (overlayContainer) {
        overlayContainer.remove();
        overlayContainer = null;
    }
    elementMap.clear();
};

// Helper to check if we are in a Chrome Extension environment (Side Panel)
const isExtension = () => {
    return typeof chrome !== 'undefined' && chrome.tabs && chrome.scripting;
};

// Async messaging wrapper
const sendToActiveTab = async (action: string, payload: any = {}) => {
    return new Promise((resolve, reject) => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const activeTab = tabs[0];
            if (!activeTab || !activeTab.id) {
                reject("No active tab");
                return;
            }

            // Ensure content script is ready (optional, or just send)
            chrome.tabs.sendMessage(activeTab.id, { action, ...payload }, (response) => {
                if (chrome.runtime.lastError) {
                    // Try injecting script if missing?
                    // For now assume "activeTab" permission handles this or user installed it properly
                    reject(chrome.runtime.lastError.message);
                } else {
                    resolve(response);
                }
            });
        });
    });
};

export const showNumberedOverlay = async () => {
    if (isExtension()) {
        const res: any = await sendToActiveTab("show_overlay");
        return res ? res.count : 0;
    }

    // --- Legacy Iframe Fallback ---
    cleanupOverlay();
    // ... (rest of local overlay logic)
    overlayContainer = document.createElement('div');
    overlayContainer.id = 'agent-overlay';
    // ... setup styles
    overlayContainer.style.position = 'absolute';
    overlayContainer.style.top = '0';
    overlayContainer.style.left = '0';
    overlayContainer.style.width = '100%';
    overlayContainer.style.height = '100%';
    overlayContainer.style.pointerEvents = 'none';
    overlayContainer.style.zIndex = '9999';

    // Check if we are in Browser View
    const browserFrame = document.getElementById('agent-browser-frame') as HTMLIFrameElement;
    let targetDoc: Document = document;
    let targetBody: HTMLElement = document.body;

    if (browserFrame && browserFrame.contentDocument) {
        try {
            targetDoc = browserFrame.contentDocument;
            targetBody = targetDoc.body;
        } catch (e) {
            console.warn("Cross-origin iframe access blocked.");
        }
    }

    targetBody.appendChild(overlayContainer);

    const elements = targetDoc.querySelectorAll('button, a, input, textarea, select, [role="button"], [onclick]');
    let count = 0;
    elements.forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0 || window.getComputedStyle(el).display === 'none') return;
        count++;
        const id = count;
        elementMap.set(id, el as HTMLElement);
        const badge = document.createElement('div');
        badge.innerText = id.toString();
        badge.style.position = 'absolute';
        badge.style.left = `${rect.left + window.scrollX}px`;
        badge.style.top = `${rect.top + window.scrollY}px`;
        badge.style.backgroundColor = '#FFFF00';
        badge.style.color = '#000000';
        badge.style.border = '1px solid #000000';
        badge.style.borderRadius = '3px';
        badge.style.padding = '0px 4px';
        badge.style.fontSize = '12px';
        badge.style.fontWeight = 'bold';
        badge.style.zIndex = '10000';
        overlayContainer?.appendChild(badge);
    });
    return count;
};

export const tools = {
    scrape_active_tab: async () => {
        if (isExtension()) {
            const res: any = await sendToActiveTab("show_overlay"); // It extracts text too
            return res ? res.text : "";
        }

        // Iframe Fallback
        try {
            const browserFrame = document.getElementById('agent-browser-frame') as HTMLIFrameElement;
            const targetDoc = (browserFrame && browserFrame.contentDocument) ? browserFrame.contentDocument : document;
            // Use simple text for now
            return targetDoc.body.innerText;
        } catch (e) {
            return document.body.innerText;
        }
    },

    click_element: async (id: number) => {
        if (isExtension()) {
            const res: any = await sendToActiveTab("click_element", { id });
            return res ? res.result : "Failed to click";
        }

        const el = elementMap.get(id);
        if (el) {
            el.click();
            el.focus();
            return `Clicked element ${id}`;
        }
        return `Error: Element ${id} not found.`;
    },

    input_text: async (id: number, text: string) => {
        if (isExtension()) {
            const res: any = await sendToActiveTab("input_text", { id, text });
            return res ? res.result : "Failed to input";
        }

        const el = elementMap.get(id);
        if (el) {
            // ... (local logic)
            if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
                el.value = text;
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                return `Input "${text}" into element ${id}`;
            }
            return `Error: Element ${id} is not an input field.`;
        }
        return `Error: Element ${id} not found.`;
    },

    navigate: async (url: string) => {
        if (isExtension()) {
            // Use chrome.tabs.update
            return new Promise((resolve) => {
                chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                    if (tabs[0]?.id) {
                        chrome.tabs.update(tabs[0].id, { url });
                        resolve(`Navigated to ${url}`);
                    } else {
                        resolve("No active tab");
                    }
                });
            });
        }

        try {
            // @ts-ignore
            window._setBrowserUrl(url);
            return `Navigating to ${url}...`;
        } catch (e) {
            return "Navigation failed.";
        }
    }
};
