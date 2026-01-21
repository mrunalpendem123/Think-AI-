import { SiGithub, SiX } from "react-icons/si";
import { Button } from "./ui/button";

export function Footer() {
  return (
    <footer className="w-full flex fixed bottom-0 right-0 p-1 z-50 bg-background/95">
      <div className="px-1 w-full flex flex-row justify-end space-x-1">
        <a href="https://github.com/mrunalpendem123/Think-AI-" target="_blank" rel="noopener noreferrer">
          <Button variant="ghost" size="icon" className="hover:bg-transparent" title="GitHub">
            <SiGithub size={16} />
          </Button>
        </a>
        <a href="https://x.com/MaxMill06" target="_blank" rel="noopener noreferrer">
          <Button variant="ghost" size="icon" className="hover:bg-transparent" title="X (Twitter)">
            <SiX size={16} />
          </Button>
        </a>
        <a href="https://buymeacoffee.com/mrunalpend7" target="_blank" rel="noopener noreferrer">
          <Button variant="ghost" size="sm" className="hover:bg-transparent text-xs text-muted-foreground hover:text-foreground transition-colors gap-2 px-2">
             <span>☕</span>
             <span>Buy me a coffee</span>
          </Button>
        </a>
      </div>
    </footer>
  );
}
