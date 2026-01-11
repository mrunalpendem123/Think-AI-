import { deleteThread, getAllThreads, getThread, saveThread, Thread } from "@/lib/indexed-db";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export const useChatThread = (threadId?: number) => {
  return useQuery({
    queryKey: ["thread", threadId],
    queryFn: async () => {
      if (!threadId) return null;
      return getThread(threadId);
    },
    enabled: !!threadId,
  });
};

export const useChatThreads = () => {
  return useQuery({
    queryKey: ["threads"],
    queryFn: async () => getAllThreads(),
  });
};

export const useSaveThread = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (thread: Omit<Thread, 'id' | 'updatedAt'> & { id?: number }) => saveThread(thread),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["threads"] });
        }
    });
};

export const useDeleteThread = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (id: number) => deleteThread(id),
        onSuccess: () => {
             queryClient.invalidateQueries({ queryKey: ["threads"] });
        }
    });
}
