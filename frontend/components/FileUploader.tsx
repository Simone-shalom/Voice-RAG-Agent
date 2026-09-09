"use client";

import { useCallback, useRef, useState } from "react";
import { friendlyErrorMessage } from "@/lib/errors";

export interface Episode {
  id: string;
  filename: string;
  chunk_count: number;
  source_url?: string | null;
  summary?: string | null;
  chapters?: { start_ts: number; title: string }[] | null;
}

function formatChapterTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

interface IngestResult {
  id: string;
  filename: string;
  chunk_count: number;
}

type UploadStatus =
  | { kind: "idle" }
  | { kind: "uploading"; label: string }
  | { kind: "done"; result: IngestResult }
  | { kind: "error"; message: string };

interface Props {
  selectedEpisodeId: string | null;
  onSelectEpisode: (ep: Episode | null) => void;
  episodes: Episode[];
  episodesLoaded: boolean;
  onIngested: () => void | Promise<void>;
}

export default function FileUploader({ selectedEpisodeId, onSelectEpisode, episodes, episodesLoaded, onIngested }: Props) {
  const [tab, setTab] = useState<"file" | "url" | "rss">("file");
  const [status, setStatus] = useState<UploadStatus>({ kind: "idle" });
  const [dragging, setDragging] = useState(false);
  const [urlInput, setUrlInput] = useState("");
  const [rssUrl, setRssUrl] = useState("");
  const [rssLimit, setRssLimit] = useState(5);
  const [chapterJumpTs, setChapterJumpTs] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function uploadFile(file: File) {
    setStatus({ kind: "uploading", label: "Uploading…" });
    const timer = setTimeout(
      () => setStatus({ kind: "uploading", label: "Transcribing with Whisper (may take a few minutes)…" }),
      3000
    );
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/ingest/upload", { method: "POST", body: form });
      clearTimeout(timer);
      if (!res.ok) throw new Error(await friendlyErrorMessage(res));
      const result: IngestResult = await res.json();
      setStatus({ kind: "done", result });
      await onIngested();
    } catch (err) {
      clearTimeout(timer);
      setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }

  async function ingestUrl() {
    const url = urlInput.trim();
    if (!url) return;
    setStatus({ kind: "uploading", label: "Downloading & transcribing…" });
    try {
      const res = await fetch("/api/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!res.ok) throw new Error(await friendlyErrorMessage(res));
      const result: IngestResult = await res.json();
      setStatus({ kind: "done", result });
      setUrlInput("");
      await onIngested();
    } catch (err) {
      setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }

  async function ingestRss() {
    const url = rssUrl.trim();
    if (!url) return;
    setStatus({ kind: "uploading", label: `Fetching feed & ingesting up to ${rssLimit} episodes…` });
    try {
      const res = await fetch("/api/ingest/rss", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, limit: rssLimit }),
      });
      if (!res.ok) throw new Error(await friendlyErrorMessage(res));
      const result: { episodes: IngestResult[]; errors: { title: string; error: string }[] } = await res.json();
      const totalChunks = result.episodes.reduce((sum, e) => sum + e.chunk_count, 0);
      if (result.episodes.length === 0 && result.errors.length > 0) {
        setStatus({ kind: "error", message: `All ${result.errors.length} episode(s) failed to ingest.` });
      } else if (result.errors.length > 0) {
        setStatus({
          kind: "done",
          result: {
            id: "batch",
            filename: `${result.episodes.length} episode(s) ingested, ${result.errors.length} failed`,
            chunk_count: totalChunks,
          },
        });
        setRssUrl("");
      } else {
        setStatus({
          kind: "done",
          result: {
            id: "batch",
            filename: `${result.episodes.length} episode(s) ingested from feed`,
            chunk_count: totalChunks,
          },
        });
        setRssUrl("");
      }
      await onIngested();
    } catch (err) {
      setStatus({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  }

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) uploadFile(file);
    },
    [] // eslint-disable-line react-hooks/exhaustive-deps
  );

  const busy = status.kind === "uploading";

  function toggleEpisode(ep: Episode) {
    onSelectEpisode(selectedEpisodeId === ep.id ? null : ep);
  }

  function selectTab(next: "file" | "url" | "rss") {
    // Never clear an in-flight "uploading" status on tab switch — status also
    // gates `busy`, which prevents concurrent ingests; clearing it here would
    // let a switched-to tab start a second upload while the first is still running.
    setStatus((prev) => (prev.kind === "uploading" ? prev : { kind: "idle" }));
    setTab(next);
  }

  return (
    <div className="flex flex-col gap-4">
      {/* ── upload section ── */}
      <div className="flex flex-col gap-3">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">
          Add to Library
        </p>

        {/* tab toggle */}
        <div className="flex text-xs rounded-lg overflow-hidden border border-gray-800">
          <button
            onClick={() => selectTab("file")}
            aria-pressed={tab === "file"}
            className={`flex-1 py-1.5 font-medium transition-colors ${
              tab === "file"
                ? "bg-indigo-600 text-white"
                : "bg-gray-900 text-gray-400 hover:bg-gray-800"
            }`}
          >
            File
          </button>
          <button
            onClick={() => selectTab("url")}
            aria-pressed={tab === "url"}
            className={`flex-1 py-1.5 font-medium transition-colors ${
              tab === "url"
                ? "bg-indigo-600 text-white"
                : "bg-gray-900 text-gray-400 hover:bg-gray-800"
            }`}
          >
            URL
          </button>
          <button
            onClick={() => selectTab("rss")}
            aria-pressed={tab === "rss"}
            className={`flex-1 py-1.5 font-medium transition-colors ${
              tab === "rss"
                ? "bg-indigo-600 text-white"
                : "bg-gray-900 text-gray-400 hover:bg-gray-800"
            }`}
          >
            RSS
          </button>
        </div>

        {/* drop zone */}
        {tab === "file" && (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => !busy && fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-4 text-center transition-colors select-none ${
              busy
                ? "border-gray-800 opacity-40 cursor-not-allowed"
                : dragging
                ? "border-indigo-400 bg-indigo-950/50 cursor-copy"
                : "border-gray-700 hover:border-indigo-600/60 cursor-pointer"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,.mp3,.wav,.m4a,.webm,.ogg,.flac"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) uploadFile(f);
                e.target.value = "";
              }}
            />
            <span aria-hidden="true" className="text-xl block mb-1">🎵</span>
            <p className="text-xs text-gray-300">
              Drop audio or{" "}
              <span className="text-indigo-400 underline underline-offset-2">browse</span>
            </p>
            <p className="text-[10px] text-gray-600 mt-0.5">MP3 · WAV · M4A · WebM · FLAC</p>
          </div>
        )}

        {/* URL input */}
        {tab === "url" && (
          <div className="flex gap-2">
            <input
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !busy && ingestUrl()}
              placeholder="https://…/episode.mp3"
              disabled={busy}
              className="flex-1 min-w-0 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-xs placeholder-gray-600 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
            />
            <button
              onClick={ingestUrl}
              disabled={busy || !urlInput.trim()}
              className="shrink-0 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-3 py-1.5 rounded-lg text-xs font-semibold"
            >
              Add
            </button>
          </div>
        )}

        {/* RSS feed input */}
        {tab === "rss" && (
          <div className="flex flex-col gap-2">
            <input
              value={rssUrl}
              onChange={(e) => setRssUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !busy && ingestRss()}
              placeholder="https://…/podcast-feed.xml"
              disabled={busy}
              className="flex-1 min-w-0 bg-gray-900 border border-gray-800 rounded-lg px-3 py-2 text-xs placeholder-gray-600 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
            />
            <div className="flex items-center gap-2">
              <label className="text-[10px] text-gray-500 whitespace-nowrap">Max episodes</label>
              <input
                type="number"
                min={1}
                max={20}
                value={rssLimit}
                onChange={(e) => setRssLimit(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
                disabled={busy}
                className="w-16 bg-gray-900 border border-gray-800 rounded-lg px-2 py-1 text-xs focus:outline-none focus:border-indigo-500 disabled:opacity-50"
              />
              <button
                onClick={ingestRss}
                disabled={busy || !rssUrl.trim()}
                className="ml-auto shrink-0 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 px-3 py-1.5 rounded-lg text-xs font-semibold"
              >
                Ingest Feed
              </button>
            </div>
            <p className="text-[10px] text-gray-600">
              Downloads and transcribes up to {rssLimit} episode(s) from the feed. This can take several minutes.
            </p>
          </div>
        )}

        {/* status */}
        {status.kind === "uploading" && (
          <div className="flex items-start gap-2 text-xs text-indigo-300">
            <span className="mt-1 w-1.5 h-1.5 rounded-full bg-indigo-400 shrink-0 animate-ping" />
            <span>{status.label}</span>
          </div>
        )}
        {status.kind === "done" && (
          <p className="text-xs text-green-400 flex items-center gap-1">
            <span aria-hidden="true">✓</span>
            <span>
              <span className="font-medium">{status.result.filename}</span>
              {" · "}
              {status.result.chunk_count} chunks
            </span>
          </p>
        )}
        {status.kind === "error" && (
          <p className="text-xs text-red-400 break-words">{status.message}</p>
        )}
      </div>

      {/* ── episode list ── */}
      <div className="flex flex-col gap-2">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">
          Library{episodes.length > 0 ? ` · ${episodes.length}` : ""}
        </p>

        {!episodesLoaded ? (
          <p className="text-[11px] text-gray-600 text-center py-2">Loading…</p>
        ) : episodes.length === 0 ? (
          <p className="text-[11px] text-gray-600 text-center py-2">
            No audio ingested yet.
          </p>
        ) : (
          episodes.map((ep) => {
            const selected = selectedEpisodeId === ep.id;
            return (
              <div key={ep.id}>
                <button
                  onClick={() => toggleEpisode(ep)}
                  aria-pressed={selected}
                  title={selected ? "Click to deselect" : "Click to focus queries on this episode"}
                  className={`text-left w-full rounded-lg border px-3 py-2 transition-colors group ${
                    selected
                      ? "bg-indigo-950/60 border-indigo-600/70"
                      : "bg-gray-900 border-gray-800 hover:border-gray-600"
                  }`}
                >
                  <div className="flex items-start justify-between gap-1">
                    <p
                      className={`text-xs font-medium truncate ${
                        selected ? "text-indigo-200" : "text-gray-300"
                      }`}
                      title={ep.filename}
                    >
                      {ep.filename}
                    </p>
                    {selected && (
                      <span className="shrink-0 text-indigo-400 text-[10px] mt-0.5">
                        <span aria-hidden="true">✓</span> focused
                      </span>
                    )}
                  </div>
                  <p className={`text-[10px] mt-0.5 ${selected ? "text-indigo-400/70" : "text-gray-600"}`}>
                    {ep.chunk_count} chunks
                  </p>
                </button>
                {selected && (ep.summary || ep.chapters?.length || ep.source_url) && (
                  <div className="mt-1 mb-2 px-3 py-2 bg-gray-950/60 border border-indigo-900/40 rounded-lg text-xs space-y-2">
                    {ep.summary && <p className="text-gray-300">{ep.summary}</p>}
                    {ep.chapters && ep.chapters.length > 0 && (
                      <div className="space-y-1">
                        <p className="text-[10px] uppercase tracking-wider text-gray-500">Chapters</p>
                        {ep.chapters.map((ch, i) => (
                          <button
                            key={i}
                            onClick={() => setChapterJumpTs(ch.start_ts)}
                            className="flex items-center gap-2 w-full text-left hover:bg-gray-800/60 rounded px-1 py-0.5"
                          >
                            <span className="text-indigo-400 font-mono text-[10px] shrink-0">
                              {formatChapterTime(ch.start_ts)}
                            </span>
                            <span className="text-gray-300 truncate">{ch.title}</span>
                          </button>
                        ))}
                      </div>
                    )}
                    {ep.source_url && (
                      <audio
                        data-testid="episode-audio-player"
                        controls
                        preload="none"
                        src={
                          chapterJumpTs !== null
                            ? `${ep.source_url.split("#")[0]}#t=${chapterJumpTs}`
                            : ep.source_url
                        }
                        className="w-full h-8"
                      >
                        Your browser does not support audio playback.
                      </audio>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
