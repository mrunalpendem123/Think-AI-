import { ServiceWorkerMLCEngineHandler } from "./web-llm.js";

const handler = new ServiceWorkerMLCEngineHandler();

self.addEventListener("activate", function (event) {
    event.waitUntil(self.clients.claim());
});

self.addEventListener("message", (event) => {
    handler.onmessage(event);
});
