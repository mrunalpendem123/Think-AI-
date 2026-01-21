declare namespace chrome {
    export namespace runtime {
        export const lastError: { message?: string } | undefined;
        export function sendMessage(extensionId: string, message: any, options?: any, responseCallback?: (response: any) => void): void;
        export function sendMessage(message: any, responseCallback?: (response: any) => void): void;
    }
    export namespace tabs {
        export function query(queryInfo: any, callback: (result: any[]) => void): void;
        export function sendMessage(tabId: number, message: any, responseCallback?: (response: any) => void): void;
        export function update(tabId: number, updateProperties: any): void;
    }
    export namespace scripting {
        export function executeScript(details: any, callback?: (result: any[]) => void): void;
    }
    export namespace sidePanel {
        export function setOptions(options: any, callback?: () => void): void;
    }
}
