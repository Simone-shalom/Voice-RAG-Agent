"use client";

import { useEffect, useState } from "react";

interface ThreadSummary {
  id: string;
  title: string;
  updated_at: string;
}

interface Props {
  activeThreadId: string;
  onSelectThread: (id: string) => void;
  onNewChat: () => void;
  refreshKey: number;
}

function relativeTime(iso: string): string {
  const diffMs = Date.now() - new Date(iso + "Z").getTime();
  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function ChatHistory({ activeThreadId, onSelectThread, onNewChat, refreshKey }: Props) {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let ignore = false;
    fetch("/api/agent/threads")
      .then((res) => (res.ok ? res.json() : []))
      .then((data: ThreadSummary[]) => {
        if (!ignore) setThreads(data);
      })
      .catch(() => {
        // backend may not be ready on first render
      })
      .finally(() => {
        if (!ignore) setLoaded(true);
      });
    return () => {
      ignore = true;
    };
  }, [refreshKey]);

  // The active thread may not be in `threads` yet (a brand new chat with no
  // messages sent) — show it as an unsaved placeholder at the top so it's
  // clear a new conversation is in progress, not that history vanished.
  const activeKnown = threads.some((t) => t.id === activeThreadId);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">Chats</p>
        <button
          onClick={onNewChat}
          className="text-[10px] font-semibold text-indigo-400 hover:text-indigo-300"
        >
          + New
        </button>
      </div>

      {!loaded ? (
        <p className="text-[11px] text-gray-600 text-center py-2">Loading…</p>
      ) : (
        <div className="flex flex-col gap-1">
          {!activeKnown && (
            <div className="text-left w-full rounded-lg border px-3 py-2 bg-indigo-950/60 border-indigo-600/70">
              <p className="text-xs font-medium text-indigo-200 truncate">New conversation</p>
              <p className="text-[10px] mt-0.5 text-indigo-400/70">not sent yet</p>
            </div>
          )}
          {threads.length === 0 && activeKnown && (
            <p className="text-[11px] text-gray-600 text-center py-2">No conversations yet.</p>
          )}
          {threads.map((t) => {
            const active = t.id === activeThreadId;
            return (
              <button
                key={t.id}
                onClick={() => onSelectThread(t.id)}
                aria-pressed={active}
                className={`text-left w-full rounded-lg border px-3 py-2 transition-colors ${
                  active
                    ? "bg-indigo-950/60 border-indigo-600/70"
                    : "bg-gray-900 border-gray-800 hover:border-gray-600"
                }`}
              >
                <p
                  className={`text-xs font-medium truncate ${active ? "text-indigo-200" : "text-gray-300"}`}
                  title={t.title}
                >
                  {t.title}
                </p>
                <p className={`text-[10px] mt-0.5 ${active ? "text-indigo-400/70" : "text-gray-600"}`}>
                  {relativeTime(t.updated_at)}
                </p>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
