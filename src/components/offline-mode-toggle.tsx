"use client";

import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useConfigStore } from "@/stores";
import { Wifi, WifiOff } from "lucide-react";

export function OfflineModeToggle() {
  const { offlineMode, toggleOfflineMode } = useConfigStore();

  return (
    <div className="flex items-center space-x-2 border p-1 px-3 rounded-full hover:bg-accent/50 transition-colors">
      <Switch
        id="offline-mode"
        checked={!offlineMode} // Checked means ONLINE
        onCheckedChange={toggleOfflineMode}
        className="scale-75"
      />
      <Label htmlFor="offline-mode" className="flex items-center gap-2 cursor-pointer text-xs font-medium">
         {offlineMode ? (
            <>
                <WifiOff className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-muted-foreground">Offline</span>
            </>
         ) : (
            <>
                <Wifi className="w-3.5 h-3.5 text-green-500" />
                <span>Online</span>
            </>
         )}
      </Label>
    </div>
  );
}
