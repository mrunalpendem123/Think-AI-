import { create } from 'zustand';

export type AppMode = 'chat' | 'agent';

interface AppModeState {
    mode: AppMode;
    browserUrl: string | null;
    toggleMode: () => void;
    setMode: (mode: AppMode) => void;
    setBrowserUrl: (url: string | null) => void;
}

export const useAppModeStore = create<AppModeState>((set) => ({
    mode: 'chat',
    browserUrl: null,
    toggleMode: () => set((state) => ({ mode: state.mode === 'chat' ? 'agent' : 'chat' })),
    setMode: (mode) => set({ mode }),
    setBrowserUrl: (url) => set({ browserUrl: url }),
}));
