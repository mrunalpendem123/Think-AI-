import { env } from "@/env.mjs";
import { StateCreator } from "zustand";
import { ChatModel } from "../../../generated";

type State = {
  model: ChatModel;
  localMode: boolean;
  proMode: boolean;
  offlineMode: boolean;
  webSearch: boolean;
};


type Actions = {
  setModel: (model: ChatModel) => void;
  toggleLocalMode: () => void;
  toggleProMode: () => void;
  toggleOfflineMode: () => void;
  setOfflineMode: (offline: boolean) => void;
  toggleWebSearch: () => void;
};


export type ConfigStore = State & Actions;

export const createConfigSlice: StateCreator<
  ConfigStore,
  [],
  [],
  ConfigStore
> = (set) => ({
  model: ChatModel.GPT_4O_MINI,
  localMode: false,
  proMode: false,
  offlineMode: false,
  webSearch: false,
  setModel: (model: ChatModel) => set({ model }),
  toggleLocalMode: () =>
    set((state) => {
      const localModeEnabled = env.NEXT_PUBLIC_LOCAL_MODE_ENABLED;
      if (!localModeEnabled) {
        return { ...state, localMode: false };
      }

      const newLocalMode = !state.localMode;
      const newModel = newLocalMode
        ? ChatModel.LLAMA3
        : ChatModel.GPT_4O_MINI;
      return { localMode: newLocalMode, model: newModel };
    }),
  toggleProMode: () =>
    set((state) => {
      const proModeEnabled = env.NEXT_PUBLIC_PRO_MODE_ENABLED;
      if (!proModeEnabled) {
        return { ...state, proMode: false };
      }
      return { ...state, proMode: !state.proMode };
    }),
  toggleOfflineMode: () =>
    set((state) => ({ ...state, offlineMode: !state.offlineMode })),
  setOfflineMode: (offline: boolean) => set({ offlineMode: offline }),
  toggleWebSearch: () =>
    set((state) => ({ ...state, webSearch: !state.webSearch })),
});
