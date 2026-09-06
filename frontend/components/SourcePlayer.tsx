"use client";

import { Episode } from "./FileUploader";

interface Source {
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
}

interface Props {
  sources: Source[];
  episodeById?: Record<string, Episode>;
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function SourcePlayer({ sources, episodeById = {} }: Props) {
  if (!sources.length) return null;

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs text-gray-400 uppercase tracking-wider">Sources</p>
      {sources.map((src, i) => {
        const episode = episodeById[src.episode_id];
        const label = episode?.filename ?? src.episode_id;
        const audioSrc = episode?.source_url
          ? `${episode.source_url.split("#")[0]}#t=${src.start_ts},${src.end_ts}`
          : null;

        return (
          <div
            key={i}
            className="bg-gray-800 rounded-lg p-3 text-sm border border-gray-700"
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="text-indigo-400 font-mono text-xs">
                {formatTime(src.start_ts)} – {formatTime(src.end_ts)}
              </span>
              <span className="text-gray-500 text-xs truncate max-w-xs" title={label}>
                {label}
              </span>
            </div>
            <p className="text-gray-300 line-clamp-3">{src.text}</p>
            {audioSrc ? (
              <audio controls preload="none" src={audioSrc} className="mt-2 w-full h-8">
                Your browser does not support audio playback.
              </audio>
            ) : (
              <p className="mt-2 text-[10px] text-gray-600 italic">
                Audio unavailable for uploaded files.
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
