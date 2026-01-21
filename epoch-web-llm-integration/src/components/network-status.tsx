"use client";

import { useConfigStore } from "@/stores";
import { useEffect } from "react";
import { toast } from "./ui/use-toast";

export function NetworkStatusListener() {
  const { setOfflineMode, offlineMode } = useConfigStore();

  useEffect(() => {
    const handleOnline = () => {
      setOfflineMode(false);
      toast({
        title: "Back Online",
        description: "You are connected to the internet. Web search is enabled.",
      });
    };

    const handleOffline = () => {
      setOfflineMode(true);
      toast({
        title: "You are Offline",
        description: "Switching to offline mode. Web search is disabled.",
      });
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    // Initial check
    if (!navigator.onLine && !offlineMode) {
        setOfflineMode(true);
    } else if (navigator.onLine && offlineMode) {
        // Maybe don't auto-switch back if user manually set it? 
        // But for now, let's sync it.
        // Actually, if user manually toggles, we might overwrite it.
        // But the requirement is "if i trun off wifi", so auto-detection is key.
        setOfflineMode(false);
    }

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []); // Run once on mount

  return null;
}
