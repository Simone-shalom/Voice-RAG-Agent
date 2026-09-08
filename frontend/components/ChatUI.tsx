"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import AudioRecorder from "./AudioRecorder";
import SourcePlayer from "./SourcePlayer";
import { Episode } from "./FileUploader";
import { friendlyErrorMessage } from "@/lib/errors";

interface Source {
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
}

interface Message {
  role: "user" | "assistant";
  text: string;
  sources?: Source[];
  streaming?: boolean;
}

type SearchMode = "semantic" | "bm25" | "hybrid" | "hybrid+rerank";

interface ThreadMessage {
  role: "user" | "assistant";
  text: string;
  sources?: Source[] | null;
  created_at: string;
}

interface Props {
  selectedEpisode: Episode | null;
  episodes: Episode[];
  threadId: string;
  onThreadUpdated?: () => void;
}

const SEARCH_MODES: { value: SearchMode; label: string }[] = [
  { value: "semantic", label: "Semantic" },
  { value: "bm25", label: "BM25" },
  { value: "hybrid", label: "Hybrid" },
  { value: "hybrid+rerank", label: "Hybrid + Rerank" },
];

export default function ChatUI({ selectedEpisode, episodes, threadId, onThreadUpdated }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<Promise<void>>(Promise.resolve());
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);

  const episodeById = useMemo(() => {
    const map: Record<string, Episode> = {};
    for (const ep of episodes) map[ep.id] = ep;
    return map;
  }, [episodes]);

  useEffect(() => {
    return () => {
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close().catch(() => {});
      }
    };
  }, []);

  // Loads the selected conversation's history whenever threadId changes —
  // covers both "switch to an older thread" (GET succeeds, populate
  // messages) and "start a new chat" (thread doesn't exist server-side
  // yet, GET 404s, so just clear to empty) through one code path.
  useEffect(() => {
    let ignore = false;
    setMessages([]);
    fetch(`/api/agent/threads/${threadId}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data: { messages: ThreadMessage[] } | null) => {
        if (ignore || !data) return;
        setMessages(
          data.messages.map((m) => ({
            role: m.role,
            text: m.text,
            sources: m.sources ?? undefined,
          }))
        );
      })
      .catch(() => {
        // thread history is a convenience — an unreachable backend just
        // leaves this conversation starting empty, same as a new chat
      });
    return () => {
      ignore = true;
    };
  }, [threadId]);

  // Re-runs on every token appended during streaming (each one is a new
  // `messages` array), so the view tracks the bottom live as text grows.
  // Setting scrollTop directly on the container (rather than
  // scrollIntoView, which walks up and can grab an unintended scrollable
  // ancestor) keeps this pinned to the message list itself.
  useEffect(() => {
    const el = messagesContainerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  function getAudioCtx(): AudioContext {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      audioCtxRef.current = new AudioContext();
    }
    return audioCtxRef.current;
  }

  function enqueueAudioChunk(base64: string) {
    audioQueueRef.current = audioQueueRef.current.then(async () => {
      try {
        const ctx = getAudioCtx();
        if (ctx.state === "suspended") await ctx.resume();
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
        await new Promise<void>((resolve) => {
          const src = ctx.createBufferSource();
          src.buffer = audioBuffer;
          src.connect(ctx.destination);
          src.onended = () => resolve();
          src.start();
        });
      } catch {
        // audio decode/playback failure is non-fatal — text response still shows
      }
    });
  }

  async function sendMessage(userText: string) {
    if (!userText.trim()) return;
    if (loading) return;
    setInput("");
    setLoading(true);

    // Create/resume the AudioContext synchronously, still inside the
    // triggering user-gesture call stack (before any await below) — browsers
    // may refuse to un-suspend an AudioContext once we've hopped past the
    // gesture window (e.g. after the fetch() call below resolves).
    const gestureCtx = getAudioCtx();
    if (gestureCtx.state === "suspended") {
      gestureCtx.resume().catch(() => {});
    }

    const messagePayload = selectedEpisode
      ? `[Focus on episode: "${selectedEpisode.filename}" (id: ${selectedEpisode.id})] ${userText}`
      : userText;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: userText },
      { role: "assistant", text: "", streaming: true },
    ]);

    try {
      const res = await fetch("/api/agent/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messagePayload, thread_id: threadId, mode }),
      });
      if (!res.ok || !res.body) throw new Error(await friendlyErrorMessage(res));

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let done = false;

      while (!done) {
        const { done: readerDone, value } = await reader.read();
        if (readerDone) break;
        buf += decoder.decode(value, { stream: true });

        const events = buf.split("\n\n");
        buf = events.pop() ?? "";

        for (const raw of events) {
          const line = raw.startsWith("data: ") ? raw.slice(6) : raw;
          if (!line.trim()) continue;

          let event: { type: string; text?: string; data?: string | Source[] };
          try {
            event = JSON.parse(line);
          } catch {
            continue;
          }

          if (event.type === "token") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  text: last.text + ((event.text as string) ?? ""),
                };
              }
              return next;
            });
          } else if (event.type === "audio" && typeof event.data === "string") {
            enqueueAudioChunk(event.data);
          } else if (event.type === "sources" && Array.isArray(event.data)) {
            const sources = event.data;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = { ...last, sources };
              }
              return next;
            });
          } else if (event.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                next[next.length - 1] = {
                  ...last,
                  text: last.text || `Error: ${event.text}`,
                  streaming: false,
                };
              }
              return next;
            });
            done = true;
          } else if (event.type === "done") {
            done = true;
          }
        }
      }

      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, streaming: false };
        }
        return next;
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant") {
          next[next.length - 1] = { ...last, text: last.text || `Error: ${message}`, streaming: false };
        } else {
          next.push({ role: "assistant", text: `Error: ${message}` });
        }
        return next;
      });
    } finally {
      setLoading(false);
      onThreadUpdated?.();
    }
  }

  return (
    <div className="flex flex-col h-full max-w-2xl mx-auto">
      {/* toolbar: episode focus chip (left) + search mode selector (right) — one row, always present */}
      <div className="shrink-0 px-4 pt-3 flex items-center justify-between gap-3">
        <div className="min-w-0">
          {selectedEpisode ? (
            <div className="inline-flex items-center gap-1.5 bg-indigo-950/70 border border-indigo-700/60 rounded-full px-3 py-1 text-xs text-indigo-300">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0" />
              <span className="truncate max-w-[200px]" title={selectedEpisode.filename}>
                Focused on: <span className="font-medium text-indigo-200">{selectedEpisode.filename}</span>
              </span>
              <span className="text-indigo-500 ml-0.5">·</span>
              <span className="text-indigo-500 text-[10px] whitespace-nowrap">
                deselect in sidebar
              </span>
            </div>
          ) : (
            <p className="text-[10px] text-gray-600">All episodes</p>
          )}
        </div>

        <div className="shrink-0 flex items-center gap-2">
          <label htmlFor="search-mode" className="text-[10px] uppercase tracking-widest text-gray-500">
            Search mode
          </label>
          <select
            id="search-mode"
            value={mode}
            onChange={(e) => setMode(e.target.value as SearchMode)}
            className="bg-gray-900 border border-gray-800 rounded-lg px-2 py-1 text-xs text-gray-300 focus:outline-none focus:border-indigo-500"
          >
            {SEARCH_MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* messages */}
      <div ref={messagesContainerRef} className="flex-1 min-h-0 overflow-y-auto no-scrollbar space-y-4 p-4">
        {messages.length === 0 && !loading && (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-gray-600 text-center max-w-xs">
              {selectedEpisode
                ? `Focused on "${selectedEpisode.filename}". Ask anything about it.`
                : "Ask a question about your audio library, or select an episode in the sidebar to focus on it."}
            </p>
          </div>
        )}

        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              aria-live={m.role === "assistant" && m.streaming ? "off" : "polite"}
              aria-atomic="true"
              className={`max-w-prose rounded-2xl px-4 py-3 text-sm ${
                m.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap">
                {m.text}
                {m.streaming && (
                  <span
                    aria-hidden="true"
                    className="inline-block w-1.5 h-4 ml-0.5 bg-gray-400 animate-pulse align-middle"
                  />
                )}
              </p>
              {m.sources && <SourcePlayer sources={m.sources} episodeById={episodeById} />}
            </div>
          </div>
        ))}
      </div>

      {/* input bar */}
      <div className="shrink-0 border-t border-gray-800 p-4 space-y-3">
        <AudioRecorder onTranscript={(t) => sendMessage(t)} />
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage(input)}
            placeholder={
              selectedEpisode
                ? `Ask about "${selectedEpisode.filename}"…`
                : "Type a question…"
            }
            disabled={loading}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 text-sm focus:outline-none focus:border-indigo-500"
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={loading}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-4 py-2 rounded-xl text-sm font-semibold focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-400"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
