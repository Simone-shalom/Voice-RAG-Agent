"use client";

import { useState } from "react";
import ChatUI from "@/components/ChatUI";
import FileUploader, { Episode } from "@/components/FileUploader";

export default function Home() {
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null);

  return (
    <div className="flex flex-col h-screen">
      {/* ── centered header ── */}
      <header className="shrink-0 border-b border-gray-800 py-4 px-6 text-center">
        <div className="flex items-center justify-center gap-2 mb-1">
          <span className="text-2xl leading-none">🎙️</span>
          <h1 className="text-xl font-bold tracking-tight text-white">
            Voice Knowledge Agent
          </h1>
        </div>
        <p className="text-xs text-gray-400 max-w-sm mx-auto leading-relaxed">
          Ask questions about your podcast or lecture library — by text or
          voice — and get spoken answers with timestamp citations.
        </p>
      </header>

      {/* ── body: sidebar + chat ── */}
      <div className="flex flex-1 overflow-hidden">
        {/* sidebar */}
        <aside className="w-64 shrink-0 border-r border-gray-800 overflow-y-auto p-4">
          <FileUploader
            selectedEpisodeId={selectedEpisode?.id ?? null}
            onSelectEpisode={setSelectedEpisode}
          />
        </aside>

        {/* chat */}
        <div className="flex-1 min-w-0 overflow-hidden">
          <ChatUI selectedEpisode={selectedEpisode} />
        </div>
      </div>
    </div>
  );
}
