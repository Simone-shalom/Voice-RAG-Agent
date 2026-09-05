"use client";

interface Source {
  episode_id: string;
  start_ts: number;
  end_ts: number;
  text: string;
}

interface Props {
  sources: Source[];
}

function formatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function SourcePlayer({ sources }: Props) {
  if (!sources.length) return null;

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs text-gray-400 uppercase tracking-wider">Sources</p>
      {sources.map((src, i) => (
        <div
          key={i}
          className="bg-gray-800 rounded-lg p-3 text-sm border border-gray-700"
        >
          <div className="flex items-center gap-2 mb-1">
            <span className="text-indigo-400 font-mono text-xs">
              {formatTime(src.start_ts)} – {formatTime(src.end_ts)}
            </span>
            <span className="text-gray-500 text-xs truncate max-w-xs">
              {src.episode_id}
            </span>
          </div>
          <p className="text-gray-300 line-clamp-3">{src.text}</p>
        </div>
      ))}
    </div>
  );
}
