import { DBSchema, openDB } from 'idb';
import { ChatMessage } from '../../generated';

export interface Thread {
  id: number;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
}

interface ThinkAIDB extends DBSchema {
  threads: {
    key: number;
    value: Thread;
    indexes: { 'by-date': number };
  };
  agent_tasks: {
    key: string;
    value: {
      id: string;
      goal: string;
      logs: string[];
      status: 'idle' | 'pending' | 'running' | 'completed' | 'failed';
      updatedAt: number;
    };
  };
}

const DB_NAME = 'think-ai-db';

export async function initDB() {
  return openDB<ThinkAIDB>(DB_NAME, 2, {
    upgrade(db, oldVersion, newVersion, transaction) {
      if (!db.objectStoreNames.contains('threads')) {
        const store = db.createObjectStore('threads', {
          keyPath: 'id',
          autoIncrement: true,
        });
        store.createIndex('by-date', 'updatedAt');
      }
      if (!db.objectStoreNames.contains('agent_tasks')) {
        const store = db.createObjectStore('agent_tasks', {
          keyPath: 'id'
        });
      }
    },
  });
}

export async function saveThread(thread: Omit<Thread, 'id' | 'updatedAt'> & { id?: number }) {
  const db = await initDB();
  const timestamp = Date.now();
  // If id is provided, update. If not, autoincrement default.
  // Wait, put will use the keyPath 'id' if present in value.
  const item = { ...thread, updatedAt: timestamp };
  return db.put('threads', item as Thread);
}

export async function getThread(id: number) {
  const db = await initDB();
  return db.get('threads', id);
}

export async function getAllThreads() {
  const db = await initDB();
  const threads = await db.getAllFromIndex('threads', 'by-date');
  return threads.reverse(); // Newest first
}

export async function deleteThread(id: number) {
  const db = await initDB();
  return db.delete('threads', id);
}

export async function saveAgentTask(task: { id: string, goal: string, logs: string[], status: 'idle' | 'running' | 'completed' | 'failed' }) {
  const db = await initDB();
  return db.put('agent_tasks', { ...task, updatedAt: Date.now() });
}

export async function getAgentTask(id: string) {
  const db = await initDB();
  return db.get('agent_tasks', id);
}
