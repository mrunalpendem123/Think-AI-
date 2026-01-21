// Think AI Content Script
let elementMap = new Map();
let overlayContainer = null;

function cleanupOverlay() {
    if (overlayContainer) {
        overlayContainer.remove();
        overlayContainer = null;
    }
    elementMap.clear();
}

function showNumberedOverlay() {
    cleanupOverlay();

    overlayContainer = document.createElement('div');
    overlayContainer.id = 'agent-overlay';
    Object.assign(overlayContainer.style, {
        position: 'fixed',
        top: '0',
        left: '0',
        width: '100vw',
        height: '100vh',
        pointerEvents: 'none',
        zIndex: '2147483647' // Max z-index
    });
    document.body.appendChild(overlayContainer);

    const elements = document.querySelectorAll('button, a, input, textarea, select, [role="button"], [onclick]');
    let count = 0;

    elements.forEach((el) => {
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0 || window.getComputedStyle(el).display === 'none') return;

        // Basic viewport check
        if (rect.bottom < 0 || rect.top > window.innerHeight) return;

        count++;
        const id = count;
        elementMap.set(id, el);

        const badge = document.createElement('div');
        badge.innerText = id.toString();
        Object.assign(badge.style, {
            position: 'fixed',
            left: `${rect.left}px`,
            top: `${rect.top}px`,
            backgroundColor: '#FFFF00',
            color: '#000000',
            border: '1px solid #000000',
            borderRadius: '3px',
            padding: '0px 4px',
            fontSize: '12px',
            fontWeight: 'bold',
            zIndex: '2147483647',
            boxShadow: '0 1px 2px rgba(0,0,0,0.2)'
        });

        overlayContainer.appendChild(badge);
    });

    return count;
}

function scrapePage() {
    // Simple text extraction for now
    return document.body.innerText;
    // Ideally use Readability here too, but that requires bundling.
    // For now, innerText is faster and "good enough" for 3B models to get context.
}

function clickElement(id) {
    const el = elementMap.get(id);
    if (el) {
        el.click();
        el.focus();
        return `Clicked element ${id}`;
    }
    return `Error: Element ${id} not found.`;
}

function inputText(id, text) {
    const el = elementMap.get(id);
    if (el) {
        el.value = text;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        return `Input "${text}" into element ${id}`;
    }
    return `Error: Element ${id} not found.`;
}

// Message Listener
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "ping") {
        sendResponse({ status: "alive" });
        return;
    }

    try {
        if (request.action === "show_overlay") {
            const count = showNumberedOverlay();
            const text = scrapePage();
            sendResponse({ count, text });
        } else if (request.action === "click_element") {
            const result = clickElement(request.id);
            sendResponse({ result });
        } else if (request.action === "input_text") {
            const result = inputText(request.id, request.text);
            sendResponse({ result });
        } else {
            sendResponse({ error: "Unknown action" });
        }
    } catch (e) {
        sendResponse({ error: e.message });
    }
    return true; // Keep channel open for async response if needed
});
