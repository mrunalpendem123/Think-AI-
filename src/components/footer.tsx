import { SiDiscord, SiGithub, SiX } from "react-icons/si";
import { Button } from "./ui/button";

export function Footer() {
  return (
    <footer className="w-full flex fixed bottom-0 right-0 p-1 z-50 bg-background/95">
      <div className="px-1 w-full flex flex-row justify-end space-x-1">
        <Button variant="ghost" size="icon" className="hover:bg-transparent">
          <SiDiscord size={16} />
        </Button>
        <Button variant="ghost" size="icon" className="hover:bg-transparent">
          <SiGithub size={16} />
        </Button>
        <Button variant="ghost" size="icon" className="hover:bg-transparent">
          <SiX size={16} />
        </Button>
      </div>
    </footer>
  );
}
