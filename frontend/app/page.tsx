"use client";

import { useState } from "react";
import ChatUI from "@/components/ChatUI";
import FileUploader, { Episode } from "@/components/FileUploader";

export default function Home() {
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null);

  return (
    <div className="flex flex-col h-screen">
      {/* ── header — single line, centered over the chat column ── */}
      <header className="shrink-0 border-b border-gray-800 flex items-center">
        {/* spacer matches sidebar width so content centers over chat */}
        <div className="w-64 shrink-0" />
        <div className="flex-1 flex items-center justify-center gap-3 py-3 px-4">
          <span className="text-xl leading-none">🎙️</span>
          <h1 className="text-base font-bold tracking-tight text-white whitespace-nowrap">
            Voice Knowledge Agent
          </h1>
          <span className="text-gray-700">·</span>
          <p className="text-xs text-gray-400 truncate">
            Ask questions about your audio library by text or voice
          </p>
        </div>
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
