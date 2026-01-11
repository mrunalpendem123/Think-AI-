import { create } from "zustand";
import { persist } from "zustand/middleware";
import { ConfigStore, createConfigSlice } from "./slices/configSlice";
import { ChatStore, createMessageSlice } from "./slices/messageSlice";

type StoreState = ChatStore & ConfigStore;

export const useStore = create<StoreState>()(
  persist(
    (...a) => ({
      ...createMessageSlice(...a),
      ...createConfigSlice(...a),
    }),
    {
      name: "store",
      partialize: (state) => ({
        model: state.model,
        localMode: state.localMode,
        proMode: state.proMode,
        offlineMode: state.offlineMode,
      }),
    },
  ),
);

export const useChatStore = () =>
  useStore((state) => ({
    messages: state.messages,
    addMessage: state.addMessage,
    setMessages: state.setMessages,
    threadId: state.threadId,
    setThreadId: state.setThreadId,
  }));

export const useConfigStore = () =>
  useStore((state) => ({
    localMode: state.localMode,
    toggleLocalMode: state.toggleLocalMode,
    model: state.model,
    setModel: state.setModel,
    proMode: state.proMode,
    toggleProMode: state.toggleProMode,
    offlineMode: state.offlineMode,
    toggleOfflineMode: state.toggleOfflineMode,
  }));
