"use client";

import { useCallback, useEffect, useState } from "react";
import ChatUI from "@/components/ChatUI";
import ChatHistory from "@/components/ChatHistory";
import FileUploader, { Episode } from "@/components/FileUploader";

export default function Home() {
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [episodesLoaded, setEpisodesLoaded] = useState(false);
  const [activeThreadId, setActiveThreadId] = useState(() => crypto.randomUUID());
  const [historyVersion, setHistoryVersion] = useState(0);

  const loadEpisodes = useCallback(async () => {
    try {
      const res = await fetch("/api/episodes");
      if (res.ok) setEpisodes(await res.json());
    } catch {
      // backend may not be ready on first render
    } finally {
      setEpisodesLoaded(true);
    }
  }, []);

  useEffect(() => {
    loadEpisodes();
  }, [loadEpisodes]);

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      {/* ── header — single line, centered over the chat column ── */}
      <header className="shrink-0 border-b border-gray-800 flex items-center">
        {/* spacer matches combined sidebar width so content centers over chat */}
        <div className="w-[480px] shrink-0" />
        <div className="flex-1 flex items-center justify-center gap-3 py-3 px-4">
          <span aria-hidden="true" className="text-xl leading-none">🎙️</span>
          <h1 className="text-base font-bold tracking-tight text-white whitespace-nowrap">
            Voice Knowledge Agent
          </h1>
          <span className="text-gray-700">·</span>
          <p className="text-xs text-gray-400 truncate">
            Ask questions about your audio library by text or voice
          </p>
        </div>
      </header>

      {/* ── body: chat history + library + chat ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* chat history — Messenger-style conversation list */}
        <aside className="w-56 shrink-0 border-r border-gray-800 overflow-y-auto no-scrollbar p-4">
          <ChatHistory
            activeThreadId={activeThreadId}
            onSelectThread={setActiveThreadId}
            onNewChat={() => setActiveThreadId(crypto.randomUUID())}
            refreshKey={historyVersion}
          />
        </aside>

        {/* episode library */}
        <aside className="w-64 shrink-0 border-r border-gray-800 overflow-y-auto no-scrollbar p-4">
          <FileUploader
            selectedEpisodeId={selectedEpisode?.id ?? null}
            onSelectEpisode={setSelectedEpisode}
            episodes={episodes}
            episodesLoaded={episodesLoaded}
            onIngested={loadEpisodes}
          />
        </aside>

        {/* chat */}
        <div className="flex-1 min-w-0 min-h-0 overflow-hidden">
          <ChatUI
            selectedEpisode={selectedEpisode}
            episodes={episodes}
            threadId={activeThreadId}
            onThreadUpdated={() => setHistoryVersion((v) => v + 1)}
          />
        </div>
      </div>
    </div>
  );
}
