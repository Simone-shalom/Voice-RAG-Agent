"use client";

import { useEffect, useRef, useState } from "react";

const CONFIRM_TIMEOUT_MS = 3000;

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
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const confirmTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (confirmTimeoutRef.current) clearTimeout(confirmTimeoutRef.current);
    };
  }, []);

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

  function handleDeleteClick(id: string) {
    if (confirmTimeoutRef.current) clearTimeout(confirmTimeoutRef.current);

    if (confirmingId !== id) {
      // first click arms the confirm state; auto-reverts if not confirmed
      setConfirmingId(id);
      confirmTimeoutRef.current = setTimeout(() => setConfirmingId(null), CONFIRM_TIMEOUT_MS);
      return;
    }

    setConfirmingId(null);
    setThreads((prev) => prev.filter((t) => t.id !== id));
    if (id === activeThreadId) onNewChat();
    fetch(`/api/agent/threads/${id}`, { method: "DELETE" }).catch(() => {
      // best-effort — already removed from the list optimistically
    });
  }

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
            const confirming = confirmingId === t.id;
            return (
              <div
                key={t.id}
                className={`group relative rounded-lg border transition-colors ${
                  confirming
                    ? "bg-red-950/40 border-red-800/70"
                    : active
                    ? "bg-indigo-950/60 border-indigo-600/70"
                    : "bg-gray-900 border-gray-800 hover:border-gray-600"
                }`}
              >
                <button
                  onClick={() => onSelectThread(t.id)}
                  aria-pressed={active}
                  disabled={confirming}
                  className="text-left w-full px-3 py-2 pr-14"
                >
                  <p
                    className={`text-xs font-medium truncate ${active ? "text-indigo-200" : "text-gray-300"}`}
                    title={t.title}
                  >
                    {t.title}
                  </p>
                  <p className={`text-[10px] mt-0.5 ${active ? "text-indigo-400/70" : "text-gray-600"}`}>
                    {confirming ? "Delete? Click × again to confirm" : relativeTime(t.updated_at)}
                  </p>
                </button>
                <button
                  onClick={() => handleDeleteClick(t.id)}
                  aria-label={confirming ? `Confirm delete "${t.title}"` : `Delete "${t.title}"`}
                  className={`absolute top-1.5 right-1.5 w-5 h-5 flex items-center justify-center rounded text-xs transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:outline-red-400 ${
                    confirming
                      ? "opacity-100 text-red-300 bg-red-900/60"
                      : "opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 hover:bg-red-950/40"
                  }`}
                >
                  ×
                </button>
                {confirming && (
                  <button
                    onClick={() => {
                      if (confirmTimeoutRef.current) clearTimeout(confirmTimeoutRef.current);
                      setConfirmingId(null);
                    }}
                    aria-label={`Cancel deleting "${t.title}"`}
                    className="absolute top-1.5 right-7 w-5 h-5 flex items-center justify-center rounded text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                  >
                    ↺
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
